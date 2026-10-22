using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Modules.Wakeup.Controllers;

/// <summary>
/// Owns the wakeup observation queue and the in-session wake-word interruption. Capture callbacks
/// only admit a typed observation; the interruption itself is decided by the Kernel on the control
/// consumer, and the abandoned turn's context is released by the runtime's turn owner.
/// </summary>
internal sealed class WakeupInterruptOwner : IWakeupDecisionSink
{
    private const int ObservationCapacity = 8;

    private readonly AssistantControlIngress _ingress;
    private readonly ITurnStateView _turnState;
    private readonly IWakeupModule? _wakeup;
    private readonly ITurnBoundarySource _turnBoundary;
    private readonly TurnCompletionCoordinator _turns;
    private readonly IBargeInDataPlaneResetPort _bargeInDataPlane;
    private readonly INearEndResetPort _nearEndReset;
    private readonly IConversationRollbackPort _history;
    private readonly IPlaybackInterruptionPort _playbackInterruption;
    private readonly IRecognitionSessionCleanupPort _recognition;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly InputGate _gate;
    private readonly ILogger _logger;
    private readonly CorrelatedObservationQueue<WakeupAudioDecision> _observations;
    private long _nextObservationId;

    public WakeupInterruptOwner(
        AssistantControlIngress ingress,
        ITurnStateView turnState,
        IWakeupModule? wakeup,
        ITurnBoundarySource turnBoundary,
        TurnCompletionCoordinator turns,
        IBargeInDataPlaneResetPort bargeInDataPlane,
        INearEndResetPort nearEndReset,
        IConversationRollbackPort history,
        IPlaybackInterruptionPort playbackInterruption,
        IRecognitionSessionCleanupPort recognition,
        AssistantTaskSupervisor tasks,
        InputGate gate,
        ILogger logger)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _wakeup = wakeup;
        _turnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        _turns = turns ?? throw new ArgumentNullException(nameof(turns));
        _bargeInDataPlane = bargeInDataPlane ?? throw new ArgumentNullException(nameof(bargeInDataPlane));
        _nearEndReset = nearEndReset ?? throw new ArgumentNullException(nameof(nearEndReset));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _playbackInterruption = playbackInterruption ?? throw new ArgumentNullException(nameof(playbackInterruption));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _observations = new CorrelatedObservationQueue<WakeupAudioDecision>(
            ingress,
            ObservationCapacity,
            "wakeup-data-plane",
            overflowCode: "VBX-CTRL-008",
            mismatchCode: "VBX-CTRL-009",
            mismatchMessageKey: "wakeup-operation-correlation");
    }

    /// <summary>Admits a wakeup observation from the capture callback. Allocation-free and non-blocking.</summary>
    public bool TryEnqueueObservation(WakeupAudioDecision decision)
    {
        long operationId = Interlocked.Increment(ref _nextObservationId);
        if (!_observations.TryWrite(operationId, decision, "wakeup-observation-overflow"))
            return false;

        return _ingress.TryEnqueueCritical(
            AssistantSignal.WakeupObserved(operationId),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.CaptureCallback,
            out _);
    }

    public WakeupSemanticPlan ProcessObservation(long operationId)
    {
        if (!_observations.TryTake(operationId, out WakeupAudioDecision decision))
            return default;

        TurnSignal signal = TurnSignal.WakeupObserved(
            operationId,
            decision.OldMode,
            decision.NewMode,
            decision.Wakeup is not null,
            decision.Wakeup?.Keyword ?? string.Empty,
            decision.Wakeup?.Score ?? 0,
            decision.Wakeup?.Threshold ?? 0,
            decision.Wakeup?.DurationMs ?? 0,
            decision.Wakeup?.StartTimeMs ?? 0,
            decision.TriggerKind switch
            {
                WakeupTriggerKind.InitialWakeup => KernelWakeupTriggerKind.InitialWakeup,
                WakeupTriggerKind.InSessionWakeup => KernelWakeupTriggerKind.InSessionWakeup,
                _ => KernelWakeupTriggerKind.None,
            });
        return new WakeupSemanticPlan(
            WakeupSemanticPlanKind.Observed,
            signal,
            decision,
            default);
    }

    public WakeupSemanticPlan CommitObservation(
        WakeupSemanticPlan plan,
        KernelCommitResult kernel)
    {
        if (!kernel.Accepted || plan.Kind != WakeupSemanticPlanKind.Observed)
            return default;

        if (plan.Decision.TriggerKind == WakeupTriggerKind.InSessionWakeup)
            return ObserveInterrupt(plan.Decision);

        return default;
    }

    private WakeupSemanticPlan ObserveInterrupt(WakeupAudioDecision decision)
    {
        if (decision.Wakeup is null)
            return default;

        // Wake word is a high-confidence command to start over, even when regular barge-in is enabled.
        if (!_turns.TryObserveWakeupInterrupt(
                _turnState.NextTurnOrdinal,
                out WakeupInterruptDecision interrupt))
        {
            return default;
        }

        _logger.LogDebug(
            "会话内唤醒词命中, 以唤醒词打断当前回复: keyword={Keyword}, oldState={OldState}, interruptedTurnId={InterruptedTurnId}, newTurnId={TurnId}.",
            decision.Wakeup.Keyword,
            interrupt.OldState,
            interrupt.InterruptedTurnId,
            interrupt.TurnId);

        return new WakeupSemanticPlan(
            WakeupSemanticPlanKind.Interrupt,
            TurnSignal.CancelRequested(
                interrupt.InterruptedTurnId,
                interrupt.TurnId,
                "wakeup-interrupt"),
            decision,
            interrupt);
    }

    public void CommitInterrupt(WakeupSemanticPlan plan, KernelCommitResult kernel)
    {
        if (!kernel.Accepted || plan.Kind != WakeupSemanticPlanKind.Interrupt)
            return;

        WakeupInterruptDecision interrupt = plan.Interrupt;
        _wakeup?.MarkUserActivity();
        _turnBoundary.Reset();
        _bargeInDataPlane.ResetAll();
        _nearEndReset.Reset();
        if (plan.Decision.ReplayAudio is { Length: > 0 } replay)
            _turnBoundary.Observe(replay);
        if (!_turns.CommitWakeupInterrupt(interrupt))
        {
            throw new InvalidOperationException(
                "Wakeup interruption observation could not be committed after Kernel acceptance.");
        }

        // The physical stop of the interrupted reply was issued by the Kernel's owner-scoped
        // StopPlayback/ClearPlaybackQueue effects during the commit above; only the borrowed
        // response source is cancelled here so the response pipeline unwinds promptly.
        try { interrupt.ResponseToCancel?.Cancel(); }
        catch (ObjectDisposedException) { }
        _history.RemoveUnansweredUserMessage(interrupt.InterruptedTurnId);
        if (interrupt.AsrSessionIdToCleanup is not null)
        {
            _tasks.Track(
                _recognition.CleanupAbandonedSessionAsync(interrupt.AsrSessionIdToCleanup),
                "recognition.cleanup-abandoned-session",
                interrupt.TurnId,
                AssistantTaskFaultPolicy.ObserveAndLog);
        }

        _gate.Open();
    }

}
