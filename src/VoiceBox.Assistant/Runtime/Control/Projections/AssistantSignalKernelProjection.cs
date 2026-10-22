using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Projections;

/// <summary>
/// Projects a runtime control signal onto the protocol-independent Kernel signal. It is a pure
/// mapping: signals with no global Turn meaning project to nothing and never reach the reducer.
/// </summary>
internal static class AssistantSignalKernelProjection
{
    public static TurnSignal? Project(
        AssistantSignal signal,
        AssistantState conversationState,
        long sourceSequence = 0,
        long sourceTimestamp = 0)
    {
        TurnSignal? projected = signal.Kind switch
        {
            AssistantSignalKind.WakeupModeChanged => TurnSignal.WakeupObserved(
                signal.OperationId,
                signal.Wakeup.FromMode,
                signal.Wakeup.ToMode,
                hasWakeup: false,
                keyword: string.Empty,
                score: 0,
                threshold: 0,
                durationMs: 0,
                startTimeMs: 0,
                KernelWakeupTriggerKind.None),
            AssistantSignalKind.RecognitionStarted when !signal.SuppressKernelProjection => TurnSignal.RecognitionSessionReady(
                signal.TurnId,
                signal.OperationId),
            AssistantSignalKind.RecognitionCompleted when !signal.SuppressKernelProjection => TurnSignal.RecognitionCompleted(
                signal.TurnId,
                signal.OperationId,
                signal.Text ?? string.Empty,
                TimeSpan.FromTicks(signal.DurationTicks)),
            AssistantSignalKind.RecognitionFailed => TurnSignal.RecognitionFailed(
                signal.TurnId,
                signal.OperationId,
                ProjectFailure(signal.Failure)),
            AssistantSignalKind.ResponseSpeakableStarted => TurnSignal.PlaybackStarted(
                signal.TurnId,
                signal.OperationId),
            AssistantSignalKind.BackgroundResponseStarted => TurnSignal.BackgroundResponseStarted(
                signal.TurnId,
                signal.OperationId),
            AssistantSignalKind.ResponseCompleted => TurnSignal.ResponseGenerated(
                signal.TurnId,
                signal.OperationId,
                signal.Text ?? string.Empty),
            AssistantSignalKind.PlaybackCompleted => TurnSignal.PlaybackCompleted(
                signal.TurnId,
                signal.OperationId,
                signal.Text ?? string.Empty,
                TimeSpan.FromTicks(signal.DurationTicks)),
            AssistantSignalKind.ResponseFailed => TurnSignal.OperationFailed(
                signal.TurnId,
                signal.OperationId,
                ProjectFailure(signal.Failure)),
            AssistantSignalKind.OperationCanceled => TurnSignal.CancelRequested(
                signal.TurnId,
                signal.OperationId,
                "response-cancelled"),
            AssistantSignalKind.ValidationCancelled => TurnSignal.CancelRequested(
                signal.TurnId,
                signal.OperationId,
                "barge-in-validation-cancelled"),
            AssistantSignalKind.TurnCompleted when conversationState == AssistantState.Thinking =>
                TurnSignal.CancelRequested(
                    signal.TurnId,
                    signal.OperationId,
                    "response-completed-without-playback"),
            AssistantSignalKind.TurnCompleted
                when conversationState is not AssistantState.Idle and not AssistantState.Speaking =>
                TurnSignal.CancelRequested(signal.TurnId, signal.OperationId, "turn-completed"),
            AssistantSignalKind.InterruptionResolved => TurnSignal.InterruptionResolved(
                signal.TurnId,
                signal.OperationId,
                signal.Flag,
                signal.Reason),
            AssistantSignalKind.RuntimeFault => TurnSignal.RuntimeFaultObserved(ProjectFailure(signal.Failure)),
            _ => null,
        };

        return projected is { } value
            ? value.WithSourceMetadata(sourceSequence, sourceTimestamp)
            : null;
    }

    public static KernelFailure ProjectFailure(AssistantFailure failure)
        => new(failure.Code, failure.Category, failure.Component, failure.MessageKey);
}
