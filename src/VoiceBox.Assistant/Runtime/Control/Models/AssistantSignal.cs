using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct AssistantSignal
{
    private AssistantSignal(
        AssistantSignalKind kind,
        long turnId,
        long operationId,
        long durationTicks,
        bool flag,
        AssistantState state,
        string? text,
        string? reason,
        AssistantFailure failure,
        AssistantWakeupObservation wakeup = default,
        long admissionEpoch = 0,
        long sourceSequence = 0,
        long sourceTimestamp = 0,
        bool suppressKernelProjection = false)
    {
        Kind = kind;
        TurnId = turnId;
        OperationId = operationId;
        DurationTicks = durationTicks;
        Flag = flag;
        State = state;
        Text = text;
        Reason = reason;
        Failure = failure;
        Wakeup = wakeup;
        AdmissionEpoch = admissionEpoch;
        SourceSequence = sourceSequence;
        SourceTimestamp = sourceTimestamp;
        SuppressKernelProjection = suppressKernelProjection;
    }

    public AssistantSignalKind Kind { get; }

    public long TurnId { get; }

    public long OperationId { get; }

    public long DurationTicks { get; }

    public bool Flag { get; }

    public AssistantState State { get; }

    public string? Text { get; }

    public string? Reason { get; }

    public AssistantFailure Failure { get; }

    public AssistantWakeupObservation Wakeup { get; }

    public long AdmissionEpoch { get; }

    /// <summary>Ingress sequence and source timestamp captured before queue admission.</summary>
    public long SourceSequence { get; }

    public long SourceTimestamp { get; }

    /// <summary>
    /// Marks a provider completion that is intentionally module-private because an equivalent
    /// Kernel transition was already committed (for example an immediately accepted barge-in
    /// that deferred recognition readiness).  The envelope still traverses the normal control
    /// loop; only the pure projection yields no second semantic signal.
    /// </summary>
    public bool SuppressKernelProjection { get; }

    internal AssistantSignal WithIngressMetadata(long sourceSequence, long sourceTimestamp)
        => new(
            Kind,
            TurnId,
            OperationId,
            DurationTicks,
            Flag,
            State,
            Text,
            Reason,
            Failure,
            Wakeup,
            AdmissionEpoch,
            sourceSequence,
            sourceTimestamp,
            SuppressKernelProjection);

    internal AssistantSignal WithKernelProjectionSuppressed()
        => new(
            Kind,
            TurnId,
            OperationId,
            DurationTicks,
            Flag,
            State,
            Text,
            Reason,
            Failure,
            Wakeup,
            AdmissionEpoch,
            SourceSequence,
            SourceTimestamp,
            suppressKernelProjection: true);

    public static AssistantSignal SpeechStarted(bool enableBargeIn)
        => new(AssistantSignalKind.SpeechStarted, 0, 0, 0, false, default, null, null, default);

    public static AssistantSignal SpeechStartedWithEndpointObservation(
        bool endpointObservationApplied,
        long admissionEpoch = 0)
        => new(
            AssistantSignalKind.SpeechStarted,
            0,
            0,
            0,
            endpointObservationApplied,
            default,
            null,
            null,
            default,
            admissionEpoch: admissionEpoch);

    public static AssistantSignal SpeechEnded(
        long turnId,
        TimeSpan duration,
        bool endpointObservationApplied = false,
        long admissionEpoch = 0)
        => new(
            AssistantSignalKind.SpeechEnded,
            turnId,
            0,
            duration.Ticks,
            endpointObservationApplied,
            default,
            null,
            null,
            default,
            admissionEpoch: admissionEpoch);

    public static AssistantSignal SpeechAudioAvailable()
        => new(
            AssistantSignalKind.SpeechAudioAvailable,
            0,
            0,
            0,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal WakeupObserved(long operationId)
        => new(AssistantSignalKind.WakeupObserved, 0, operationId, 0, false, default, null, null, default, default);

    public static AssistantSignal WakeupModeChanged(
        long operationId,
        AssistantWakeMode fromMode,
        AssistantWakeMode toMode)
        => new(
            AssistantSignalKind.WakeupModeChanged,
            0,
            operationId,
            0,
            false,
            default,
            null,
            null,
            default,
            new AssistantWakeupObservation(fromMode, toMode));

    public static AssistantSignal InterruptionRequested(long turnId, long operationId)
        => new(AssistantSignalKind.InterruptionRequested, turnId, operationId, 0, false, default, null, null, default);

    public static AssistantSignal BackgroundResponseStarted(long turnId, long operationId)
        => new(AssistantSignalKind.BackgroundResponseStarted, turnId, operationId, 0, false, default, null, null, default);

    public static AssistantSignal RecognitionCompleted(
        long turnId,
        long operationId,
        string text,
        TimeSpan speechDuration = default)
        => new(
            AssistantSignalKind.RecognitionCompleted,
            turnId,
            operationId,
            speechDuration.Ticks,
            false,
            default,
            text,
            null,
            default);

    public static AssistantSignal ActiveNearEndInterruptionRequested(long operationId)
        => new(
            AssistantSignalKind.ActiveNearEndInterruptionRequested,
            0,
            operationId,
            0,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal ActiveNearEndSpeechEnded(long turnId, TimeSpan duration)
        => new(
            AssistantSignalKind.ActiveNearEndSpeechEnded,
            turnId,
            0,
            duration.Ticks,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal ActiveNearEndSpeechDrainCompleted(
        long turnId,
        TimeSpan duration)
        => new(
            AssistantSignalKind.ActiveNearEndSpeechDrainCompleted,
            turnId,
            0,
            duration.Ticks,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal PlaybackTransportStarted(long turnId)
        => new(
            AssistantSignalKind.PlaybackTransportStarted,
            turnId,
            0,
            0,
            false,
            default,
            null,
            null,
            default);


    public static AssistantSignal SpeakerGateResolved(long operationId)
        => new(
            AssistantSignalKind.SpeakerGateResolved,
            0,
            operationId,
            0,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal RecognitionStarted(
        long turnId,
        long operationId,
        string sessionId)
        => new(
            AssistantSignalKind.RecognitionStarted,
            turnId,
            operationId,
            0,
            false,
            default,
            sessionId,
            null,
            default);

    public static AssistantSignal ResponseStarted(long turnId, long operationId)
        => new(
            AssistantSignalKind.ResponseStarted,
            turnId,
            operationId,
            0,
            false,
            AssistantState.Thinking,
            null,
            null,
            default);

    public static AssistantSignal ResponseSpeakableStarted(long turnId, long operationId)
        => new(
            AssistantSignalKind.ResponseSpeakableStarted,
            turnId,
            operationId,
            0,
            false,
            AssistantState.Speaking,
            null,
            null,
            default);

    public static AssistantSignal ResponseCompleted(long turnId, long operationId, string text)
        => new(
            AssistantSignalKind.ResponseCompleted,
            turnId,
            operationId,
            0,
            false,
            default,
            text,
            null,
            default);

    public static AssistantSignal PlaybackCompleted(
        long turnId,
        long operationId,
        string text,
        TimeSpan elapsed)
        => new(
            AssistantSignalKind.PlaybackCompleted,
            turnId,
            operationId,
            elapsed.Ticks,
            false,
            default,
            text,
            null,
            default);

    public static AssistantSignal OperationCancelled(long turnId, long operationId)
        => new(
            AssistantSignalKind.OperationCanceled,
            turnId,
            operationId,
            0,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal ValidationResolved(
        long turnId,
        long operationId,
        string? text,
        string reason,
        bool meaningful)
        => new(
            AssistantSignalKind.InterruptionResolved,
            turnId,
            operationId,
            0,
            meaningful,
            default,
            text,
            reason,
            default);

    public static AssistantSignal ValidationTurnCompleted(
        long turnId,
        long operationId,
        bool endConversationRequested)
        => new(
            AssistantSignalKind.TurnCompleted,
            turnId,
            operationId,
            0,
            endConversationRequested,
            default,
            null,
            null,
            default);

    public static AssistantSignal ValidationCancelled(long turnId, long operationId)
        => new(
            AssistantSignalKind.ValidationCancelled,
            turnId,
            operationId,
            0,
            false,
            default,
            null,
            null,
            default);

    public static AssistantSignal OperationFailed(
        AssistantSignalKind kind,
        long turnId,
        long operationId,
        AssistantFailure failure)
    {
        if (kind is not (AssistantSignalKind.RecognitionFailed or
            AssistantSignalKind.ResponseFailed or
            AssistantSignalKind.PlaybackFailed or
            AssistantSignalKind.RuntimeFault))
        {
            throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
        }

        return new AssistantSignal(kind, turnId, operationId, 0, false, default, null, null, failure);
    }
}
