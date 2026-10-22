using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Kernel.Signals;

internal readonly record struct TurnSignal
{
    private TurnSignal(
        TurnSignalKind kind,
        long sourceSequence,
        long sourceTick,
        long turnId,
        long operationId,
        bool flag,
        long durationTicks,
        string? text,
        string? reason,
        KernelFailure failure,
        KernelWakeupObservation wakeup = default,
        AssistantState? restoreState = null,
        bool hasBargeInContext = false,
        BargeInTriggerSource bargeInSource = BargeInTriggerSource.Vad,
        BargeInDecisionDiagnostics bargeInDiagnostics = default,
        TurnAuthorizationStatus authorization = TurnAuthorizationStatus.NotRequired,
        long candidateId = 0)
    {
        Kind = kind;
        SourceSequence = sourceSequence;
        SourceTick = sourceTick;
        TurnId = turnId;
        OperationId = operationId;
        Flag = flag;
        DurationTicks = durationTicks;
        Text = text;
        Reason = reason;
        Failure = failure;
        Wakeup = wakeup;
        RestoreState = restoreState;
        HasBargeInContext = hasBargeInContext;
        BargeInSource = bargeInSource;
        BargeInDiagnostics = bargeInDiagnostics;
        Authorization = authorization;
        CandidateId = candidateId;
    }

    public TurnSignalKind Kind { get; }
    public long SourceSequence { get; }
    public long SourceTick { get; }
    public long TurnId { get; }
    public long OperationId { get; }
    public bool Flag { get; }
    public long DurationTicks { get; }
    public string? Text { get; }
    public string? Reason { get; }
    public KernelFailure Failure { get; }
    public KernelWakeupObservation Wakeup { get; }
    public AssistantState? RestoreState { get; }
    public bool HasBargeInContext { get; }
    public BargeInTriggerSource BargeInSource { get; }
    public BargeInDecisionDiagnostics BargeInDiagnostics { get; }

    /// <summary>
    /// For <see cref="TurnSignalKind.InterruptionRequested"/> and <see cref="TurnSignalKind.SpeechStarted"/>:
    /// the identity mode the candidate/turn enters with (<c>NotRequired</c> or <c>PendingSpeaker</c>).
    /// For <see cref="TurnSignalKind.AuthorizationResolved"/>: the resolved status.
    /// </summary>
    public TurnAuthorizationStatus Authorization { get; }

    /// <summary>Speaker candidate correlation; 0 when the signal carries none.</summary>
    public long CandidateId { get; }

    internal TurnSignal WithSourceMetadata(long sourceSequence, long sourceTick)
        => new(
            Kind,
            sourceSequence,
            sourceTick,
            TurnId,
            OperationId,
            Flag,
            DurationTicks,
            Text,
            Reason,
            Failure,
            Wakeup,
            RestoreState,
            HasBargeInContext,
            BargeInSource,
            BargeInDiagnostics,
            Authorization,
            CandidateId);

    public static TurnSignal InputGateChanged(long sourceSequence, long sourceTick, bool isOpen)
        => new(TurnSignalKind.InputGateChanged, sourceSequence, sourceTick, 0, 0, isOpen, 0, null, null, default);

    /// <summary>
    /// Opens a Listening turn. <paramref name="authorization"/> is <c>NotRequired</c> for the plain
    /// single-talk path and <c>PendingSpeaker</c> when the effective SingleTalk purpose requires the
    /// utterance's own owner Match before its recognition result may start a semantic turn.
    /// </summary>
    public static TurnSignal SpeechStarted(
        long turnId,
        long operationId,
        long sourceSequence = 0,
        long sourceTick = 0,
        TurnAuthorizationStatus authorization = TurnAuthorizationStatus.NotRequired)
        => new(
            TurnSignalKind.SpeechStarted,
            sourceSequence,
            sourceTick,
            turnId,
            operationId,
            false,
            0,
            null,
            null,
            default,
            authorization: authorization);

    public static TurnSignal SpeechEnded(
        long turnId,
        long operationId,
        TimeSpan duration,
        string sessionId,
        long sourceSequence = 0,
        long sourceTick = 0)
        => new(
            TurnSignalKind.SpeechEnded,
            sourceSequence,
            sourceTick,
            turnId,
            operationId,
            false,
            duration.Ticks,
            sessionId,
            null,
            default);

    public static TurnSignal WakeupObserved(
        long operationId,
        AssistantWakeMode fromMode,
        AssistantWakeMode toMode,
        bool hasWakeup,
        string keyword,
        int score,
        int threshold,
        long durationMs,
        long startTimeMs,
        KernelWakeupTriggerKind triggerKind)
        => new(
            TurnSignalKind.WakeupDetected,
            0,
            0,
            0,
            operationId,
            false,
            0,
            null,
            null,
            default,
            new KernelWakeupObservation(
                fromMode,
                toMode,
                fromMode != toMode,
                hasWakeup,
                keyword,
                score,
                threshold,
                durationMs,
                startTimeMs,
                triggerKind));

    /// <summary>
    /// Speaker verdict for the pending candidate. <paramref name="turnId"/> is the validation turn
    /// an interruption candidate opened, or the current Listening/Recognizing turn of a pending
    /// single-talk utterance; <paramref name="status"/> is <c>Authorized</c> or one of the
    /// <c>Denied*</c> values. Only a Match of the same candidate can authorize a stop or let the
    /// utterance's recognition result start a semantic turn.
    /// </summary>
    public static TurnSignal AuthorizationResolved(
        long turnId,
        long operationId,
        TurnAuthorizationStatus status,
        long candidateId,
        string? reason = null,
        long sourceSequence = 0,
        long sourceTick = 0)
        => new(
            TurnSignalKind.AuthorizationResolved,
            sourceSequence,
            sourceTick,
            turnId,
            operationId,
            status == TurnAuthorizationStatus.Authorized,
            0,
            null,
            reason,
            default,
            authorization: status,
            candidateId: candidateId);

    public static TurnSignal RecognitionSessionReady(
        long turnId,
        long operationId,
        long sourceSequence = 0,
        long sourceTick = 0)
        => new(
            TurnSignalKind.RecognitionSessionReady,
            sourceSequence,
            sourceTick,
            turnId,
            operationId,
            false,
            0,
            null,
            null,
            default);

    public static TurnSignal RecognitionCompleted(
        long turnId,
        long operationId,
        string text,
        TimeSpan speechDuration = default)
        => new(
            TurnSignalKind.RecognitionCompleted,
            0,
            0,
            turnId,
            operationId,
            false,
            speechDuration.Ticks,
            text,
            null,
            default);

    public static TurnSignal RecognitionFailed(long turnId, long operationId, KernelFailure failure)
        => new(TurnSignalKind.RecognitionFailed, 0, 0, turnId, operationId, false, 0, null, null, failure);

    /// <summary>
    /// Establishes an interruption candidate against the current turn. <paramref name="authorization"/>
    /// must be <c>NotRequired</c> (KWS disabled: the acoustic interrupt stops playback now) or
    /// <c>PendingSpeaker</c> (KWS enabled: no playback side effect until AuthorizationResolved).
    /// </summary>
    public static TurnSignal InterruptionRequested(
        long turnId,
        long operationId,
        bool publishHeard = false,
        BargeInTriggerSource source = BargeInTriggerSource.Vad,
        BargeInDecisionDiagnostics diagnostics = default,
        bool hasBargeInContext = false,
        TurnAuthorizationStatus authorization = TurnAuthorizationStatus.NotRequired,
        long candidateId = 0)
        => new(
            TurnSignalKind.InterruptionRequested,
            0,
            0,
            turnId,
            operationId,
            publishHeard,
            0,
            null,
            null,
            default,
            hasBargeInContext: hasBargeInContext,
            bargeInSource: source,
            bargeInDiagnostics: diagnostics,
            authorization: authorization,
            candidateId: candidateId);

    public static TurnSignal InterruptionResolved(
        long turnId,
        long operationId,
        bool accepted,
        string? reason = null,
        string? text = null,
        AssistantState? restoreState = null,
        long sourceSequence = 0,
        long sourceTick = 0)
        => new(
            TurnSignalKind.InterruptionResolved,
            sourceSequence,
            sourceTick,
            turnId,
            operationId,
            accepted,
            0,
            text,
            reason,
            default,
            restoreState: restoreState);

    public static TurnSignal ResponseGenerated(long turnId, long operationId, string text)
        => new(TurnSignalKind.ResponseGenerated, 0, 0, turnId, operationId, false, 0, text, null, default);

    public static TurnSignal PlaybackStarted(long turnId, long operationId)
        => new(TurnSignalKind.PlaybackStarted, 0, 0, turnId, operationId, false, 0, null, null, default);

    public static TurnSignal BackgroundResponseStarted(long turnId, long operationId)
        => new(TurnSignalKind.BackgroundResponseStarted, 0, 0, turnId, operationId, false, 0, null, null, default);

    public static TurnSignal PlaybackCompleted(long turnId, long operationId, TimeSpan elapsed = default)
        => new(TurnSignalKind.PlaybackCompleted, 0, 0, turnId, operationId, false, elapsed.Ticks, null, null, default);

    public static TurnSignal PlaybackCompleted(
        long turnId,
        long operationId,
        string text,
        TimeSpan elapsed)
        => new(
            TurnSignalKind.PlaybackCompleted,
            0,
            0,
            turnId,
            operationId,
            false,
            elapsed.Ticks,
            text,
            null,
            default);

    public static TurnSignal OperationFailed(
        long turnId,
        long operationId,
        KernelFailure failure,
        TurnSignalKind failedKind = TurnSignalKind.OperationFailed)
        => new(failedKind, 0, 0, turnId, operationId, false, 0, null, null, failure);

    public static TurnSignal CancelRequested(long turnId, long operationId, string? reason = null)
        => new(TurnSignalKind.CancelRequested, 0, 0, turnId, operationId, false, 0, null, reason, default);

    public static TurnSignal RuntimeFaultObserved(KernelFailure failure)
        => new(TurnSignalKind.RuntimeFaultObserved, 0, 0, 0, 0, false, 0, null, null, failure);
}
