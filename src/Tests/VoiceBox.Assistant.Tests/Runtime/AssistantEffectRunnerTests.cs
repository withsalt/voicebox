using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Publications;
using Xunit;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Execution;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Handlers;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.Control.State;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void ControlLoopTurnWriter_RejectsOutOfLoopCommits()
    {
        AssistantControlLoopTurnWriter writer = CreateTurnWriter(
            isControlLoopMutation: static () => false,
            new RecordingEffectPorts());

        Assert.Throws<InvalidOperationException>(() => writer.Commit(
            TurnSignal.InputGateChanged(1, 1, isOpen: false)));
    }

    [Fact]
    public void ControlLoopTurnWriter_ReportsStatefulPublicationSequenceAndUnchangedConversationState()
    {
        var events = new RecordingEffectPorts();
        AssistantControlLoopTurnWriter writer = CreateTurnWriter(
            isControlLoopMutation: static () => true,
            events);

        KernelCommitResult result = writer.Commit(TurnSignal.WakeupObserved(
            operationId: 1,
            fromMode: AssistantWakeMode.Disabled,
            toMode: AssistantWakeMode.WaitingForWakeWord,
            hasWakeup: false,
            keyword: string.Empty,
            score: 0,
            threshold: 0,
            durationMs: 0,
            startTimeMs: 0,
            KernelWakeupTriggerKind.None));

        Assert.True(result.Accepted);
        Assert.False(result.StateChanged);
        Assert.True(result.PublicationSequence > 0);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, writer.Snapshot.WakeMode);
    }

    [Fact]
    public void ControlLoopTurnWriter_RejectsAsyncEffectOnControlThread()
    {
        var events = new RecordingEffectPorts { AsyncInputGate = true };
        AssistantControlLoopTurnWriter writer = CreateTurnWriter(
            isControlLoopMutation: static () => true,
            events);

        Assert.Throws<InvalidOperationException>(() => writer.Commit(
            TurnSignal.InputGateChanged(1, 1, isOpen: false)));
    }

    [Fact]
    public void ControlLoopTurnWriter_ProjectsCommittedStateBeforeCancellationEffects()
    {
        AssistantState projected = AssistantState.Idle;
        AssistantState stateObservedByCancellation = AssistantState.Stopped;
        AssistantState snapshotObservedByCancellation = AssistantState.Stopped;
        AssistantControlLoopTurnWriter? writer = null;
        var effects = new RecordingEffectPorts
        {
            CancelOperationObserved = () =>
            {
                stateObservedByCancellation = projected;
                snapshotObservedByCancellation = writer!.Snapshot.ConversationState;
            },
        };
        writer = CreateTurnWriter(
            isControlLoopMutation: static () => true,
            effects,
            (_, state) => projected = state);

        Assert.True(writer.Commit(TurnSignal.SpeechStarted(1, 1)).Accepted);
        Assert.Equal(AssistantState.Listening, projected);

        Assert.True(writer.Commit(TurnSignal.CancelRequested(1, 2, "test-cancel")).Accepted);

        Assert.Equal(AssistantState.Idle, stateObservedByCancellation);
        Assert.Equal(AssistantState.Idle, snapshotObservedByCancellation);
        Assert.Equal(AssistantState.Idle, projected);
    }

    [Fact]
    public async Task ClosedEffectSwitch_DispatchesEveryCatalogedEffectExactlyOnce()
    {
        var ports = new RecordingEffectPorts();
        var runner = new AssistantEffectRunner(ports.ToHandlers());
        AssistantEffect[] effects =
        [
            new SetInputGateEffect(1, true),
            new PrepareRecognitionEffect(1, 2),
            new FinishRecognitionEffect(1, 3, "session"),
            new GenerateResponseEffect(1, 4, "text"),
            new CancelResponseEffect(1),
            new ClearPlaybackQueueEffect(1),
            new StopPlaybackEffect(1),
            new CommitHistoryEffect(1, "text"),
            new RollbackHistoryEffect(1),
            new PublishStateChangedEffect(1, AssistantState.Idle, AssistantState.Listening),
            new PublishUserSpokeEffect(1, "user", TimeSpan.Zero),
            new PublishAssistantResponseGeneratedEffect(1, "assistant"),
            new PublishAssistantRepliedEffect(1, "assistant", TimeSpan.Zero),
            new PublishFaultedEffect(1, new KernelFailure("code", "category", "component", "message")),
            new PublishBargeInValidationEffect(1, BargeInValidationKind.Heard, null, null),
            new CancelOperationEffect(1, 2),
            new CompleteTurnEffect(1),
        ];

        foreach (AssistantEffect effect in effects)
            await runner.ExecuteAsync(effect);

        Assert.Equal(
            [
                "SetInputGate", "PrepareRecognition", "FinishRecognition", "GenerateResponse",
                "CancelResponse", "ClearPlaybackQueue",
                "StopPlayback", "CommitHistory", "RollbackHistory", "PublishStateChanged",
                "PublishUserSpoke", "PublishAssistantResponseGenerated", "PublishAssistantReplied",
                "PublishFaulted", "PublishBargeInValidation",
                "CancelOperation", "CompleteTurn",
            ],
            ports.Calls);
    }

    [Fact]
    public void EffectRunner_RequiresACompleteRuntimeHandlerBundle()
    {
        Assert.Throws<ArgumentNullException>(
            () => new AssistantEffectRunner(null!));
    }

    private static AssistantControlLoopTurnWriter CreateTurnWriter(
        Func<bool> isControlLoopMutation,
        RecordingEffectPorts ports,
        Action<TurnSignal, AssistantState>? projectCommittedState = null)
    {
        var effectRunner = new AssistantEffectRunner(ports.ToHandlers());
        var publications = new OrderedPublicationBoundary(
            new AssistantEventDispatcher(new object(), NullLogger.Instance));
        var publicEvents = new AssistantRuntimePublicEventEffectHandler(
            publications,
            NullLogger.Instance,
            new AlwaysClaimFaultPublication(),
            new TurnStateProjection());
        ports.PublicEvents = publicEvents;
        // The host is the production latch that proves a caller runs on the single consumer.
        var loopHost = new AssistantControlLoopHost(new AssistantControlIngress(), TimeProvider.System);
        if (isControlLoopMutation())
            loopHost.BeginMutation("test");
        var writer = new AssistantControlLoopTurnWriter(
            new TurnSnapshotStore(),
            new TurnStateProjection(),
            [new DelegateCommittedTurnObserver(projectCommittedState ?? ((_, _) => { }))],
            publicEvents,
            loopHost,
            benchmarkProbe: null);
        writer.BindEffects(effectRunner);
        return writer;
    }

    private sealed class DelegateCommittedTurnObserver(Action<TurnSignal, AssistantState> observe)
        : ICommittedTurnObserver
    {
        public void OnTurnCommitted(TurnSignal signal, AssistantState state) => observe(signal, state);
    }

    private sealed class AlwaysClaimFaultPublication : IRuntimeFaultPublicationGate
    {
        public bool TryClaim(out Exception? fatalException)
        {
            fatalException = null;
            return true;
        }
    }

    private sealed class RecordingEffectPorts :
        IInputGateEffectHandler,
        IRecognitionEffectHandler,
        IResponseEffectHandler,
        IPlaybackEffectHandler,
        IConversationEffectHandler,
        IPublicEventEffectHandler,
        IRuntimeEffectHandler
    {
        public List<string> Calls { get; } = [];
        public AssistantRuntimePublicEventEffectHandler? PublicEvents { get; set; }
        public bool AsyncInputGate { get; set; }
        public Action? CancelOperationObserved { get; set; }

        public AssistantEffectHandlers ToHandlers()
            => new(this, this, this, this, this, this, this);

        public ValueTask SetInputGateAsync(SetInputGateEffect effect, CancellationToken cancellationToken)
            => AsyncInputGate
                ? new ValueTask(Task.FromException(new InvalidCastException("async test effect")))
                : Record("SetInputGate");
        public ValueTask PrepareRecognitionAsync(PrepareRecognitionEffect effect, CancellationToken cancellationToken) => Record("PrepareRecognition");
        public ValueTask FinishRecognitionAsync(FinishRecognitionEffect effect, CancellationToken cancellationToken) => Record("FinishRecognition");
        public ValueTask GenerateResponseAsync(GenerateResponseEffect effect, CancellationToken cancellationToken) => Record("GenerateResponse");
        public ValueTask CancelResponseAsync(CancelResponseEffect effect, CancellationToken cancellationToken) => Record("CancelResponse");
        public ValueTask ClearPlaybackQueueAsync(ClearPlaybackQueueEffect effect, CancellationToken cancellationToken) => Record("ClearPlaybackQueue");
        public ValueTask StopPlaybackAsync(StopPlaybackEffect effect, CancellationToken cancellationToken) => Record("StopPlayback");
        public ValueTask CommitHistoryAsync(CommitHistoryEffect effect, CancellationToken cancellationToken) => Record("CommitHistory");
        public ValueTask RollbackHistoryAsync(RollbackHistoryEffect effect, CancellationToken cancellationToken) => Record("RollbackHistory");
        public ValueTask PublishWakeModeChangedAsync(PublishWakeModeChangedEffect effect, CancellationToken cancellationToken)
            => PublicEvents?.PublishWakeModeChangedAsync(effect, cancellationToken) ?? Record("PublishWakeModeChanged");
        public ValueTask PublishWakeupDetectedAsync(PublishWakeupDetectedEffect effect, CancellationToken cancellationToken)
            => PublicEvents?.PublishWakeupDetectedAsync(effect, cancellationToken) ?? Record("PublishWakeupDetected");
        public ValueTask PublishStateChangedAsync(PublishStateChangedEffect effect, CancellationToken cancellationToken)
            => PublicEvents?.PublishStateChangedAsync(effect, cancellationToken) ?? Record("PublishStateChanged");
        public ValueTask PublishBargeInValidationAsync(PublishBargeInValidationEffect effect, CancellationToken cancellationToken)
            => PublicEvents?.PublishBargeInValidationAsync(effect, cancellationToken) ?? Record("PublishBargeInValidation");
        public ValueTask PublishUserSpokeAsync(PublishUserSpokeEffect effect, CancellationToken cancellationToken) => Record("PublishUserSpoke");
        public ValueTask PublishAssistantResponseGeneratedAsync(PublishAssistantResponseGeneratedEffect effect, CancellationToken cancellationToken) => Record("PublishAssistantResponseGenerated");
        public ValueTask PublishAssistantRepliedAsync(PublishAssistantRepliedEffect effect, CancellationToken cancellationToken) => Record("PublishAssistantReplied");
        public ValueTask PublishFaultedAsync(PublishFaultedEffect effect, CancellationToken cancellationToken)
            => PublicEvents?.PublishFaultedAsync(effect, cancellationToken) ?? Record("PublishFaulted");
        public ValueTask CancelOperationAsync(CancelOperationEffect effect, CancellationToken cancellationToken)
        {
            CancelOperationObserved?.Invoke();
            return Record("CancelOperation");
        }
        public ValueTask CompleteTurnAsync(CompleteTurnEffect effect, CancellationToken cancellationToken) => Record("CompleteTurn");

        private ValueTask Record(string name)
        {
            Calls.Add(name);
            return ValueTask.CompletedTask;
        }
    }
}
