using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Effects.Execution;
using VoiceBox.Assistant.Runtime.Effects.Handlers;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>
/// The only writer of the committed Turn snapshot. Callers may request a decision, but the single
/// AssistantControlLoop consumer is the only one permitted to commit it, and every new value comes
/// from the pure Kernel reduction of the current snapshot.
/// </summary>
internal sealed class AssistantControlLoopTurnWriter : ITurnSemanticCommitPort
{
    private readonly TurnSnapshotStore _snapshots;
    private readonly TurnStateProjection _projection;
    private readonly List<ICommittedTurnObserver> _committedObservers;
    private readonly AssistantRuntimePublicEventEffectHandler _events;
    private readonly AssistantControlLoopHost _loopHost;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;
    private AssistantEffectRunner? _effects;
    private long _lastPublicationSequence;

    public AssistantControlLoopTurnWriter(
        TurnSnapshotStore snapshots,
        TurnStateProjection projection,
        IEnumerable<ICommittedTurnObserver> committedObservers,
        AssistantRuntimePublicEventEffectHandler events,
        AssistantControlLoopHost loopHost,
        VoiceAssistantBenchmarkProbe? benchmarkProbe)
    {
        _snapshots = snapshots ?? throw new ArgumentNullException(nameof(snapshots));
        _projection = projection ?? throw new ArgumentNullException(nameof(projection));
        _committedObservers = [.. committedObservers ?? throw new ArgumentNullException(nameof(committedObservers))];
        _events = events ?? throw new ArgumentNullException(nameof(events));
        _loopHost = loopHost ?? throw new ArgumentNullException(nameof(loopHost));
        _benchmarkProbe = benchmarkProbe;
    }

    public TurnSnapshot Snapshot => _snapshots.Snapshot;

    /// <summary>
    /// Composition binds the closed effect runner once its typed handlers exist. Kept separate
    /// from construction because those handlers need a commit port of their own.
    /// </summary>
    /// <summary>
    /// Registers a projection that must be refreshed inside the commit. Composition adds them
    /// before the loop starts; the list is never mutated afterwards.
    /// </summary>
    public void AddCommittedObserver(ICommittedTurnObserver observer)
        => _committedObservers.Add(observer ?? throw new ArgumentNullException(nameof(observer)));

    public void BindEffects(AssistantEffectRunner effects)
    {
        if (_effects is not null)
            throw new InvalidOperationException("Assistant effect runner was already bound.");

        _effects = effects ?? throw new ArgumentNullException(nameof(effects));
    }

    public KernelCommitResult Commit(TurnSignal signal)
    {
        if (!_loopHost.IsInMutation)
        {
            throw new InvalidOperationException(
                "TurnKernel decisions must be committed by AssistantControlLoop.");
        }

        TurnSnapshot previous = _snapshots.Snapshot;
        TurnDecision decision = TurnKernel.Reduce(previous, signal);
        _benchmarkProbe?.ObserveControlTrace(
            "kernel.reduce",
            signal.TurnId,
            signal.OperationId,
            signal.Kind.ToString());
        _benchmarkProbe?.ObserveKernelDecision(signal, previous, decision);
        if (!decision.Accepted)
            return default;

        AssistantState previousState = previous.ConversationState;
        Volatile.Write(ref _lastPublicationSequence, 0);
        // Capture/data-plane callbacks classify facts through these projections. Publish them at
        // the same commit point as the Kernel snapshot, before any cancellation or playback effect
        // becomes externally visible. Otherwise ClearQueue or a cancelled response can be observed
        // while callbacks still route using the old state.
        for (int i = 0; i < _committedObservers.Count; i++)
            _committedObservers[i].OnTurnCommitted(signal, decision.Next.ConversationState);
        _projection.Publish(decision.Next.ConversationState);
        // Snapshot is the public commit marker. Publish it only after every private/data-plane
        // projection has applied the decision, but before executing externally visible effects.
        _snapshots.Publish(decision.Next);

        foreach (AssistantEffect effect in decision.Effects)
        {
            ValueTask execution = _effects!.ExecuteAsync(effect);
            if (!execution.IsCompletedSuccessfully)
            {
                throw new InvalidOperationException(
                    $"Kernel effect '{effect.Kind}' performed asynchronous work on the control loop.");
            }

            execution.GetAwaiter().GetResult();
            if (effect is PublishStatefulEventEffect)
                Volatile.Write(ref _lastPublicationSequence, _events.LastPublicationSequence);
        }

        return new KernelCommitResult(
            Accepted: true,
            StateChanged: previousState != decision.Next.ConversationState,
            PublicationSequence: Volatile.Read(ref _lastPublicationSequence));
    }

}
