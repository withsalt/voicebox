using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>
/// Narrow turn-boundary port for recognition capture ownership. It exposes only the observations
/// and lifecycle transitions needed by the boundary owner; ASR provider/session state and pending
/// PCM remain private to <see cref="Modules.RecognitionModule"/>.
/// </summary>
internal interface IRecognitionCaptureOwner
{
    string? GetCaptureSessionId(long activeTurnId, long validationTurnId);

    SpeakingDecision ObserveCaptureAudio(
        long activeTurnId,
        AssistantState conversationState,
        ReadOnlySpan<float> audio,
        BargeInValidationCaptureSnapshot validation);

    SpeechEndedDecision ObserveCaptureSpeechEnded(
        long activeTurnId,
        long validationTurnId,
        AssistantState conversationState,
        TimeSpan duration);

    bool TryAttachCaptureSession(
        long turnId,
        string sessionId,
        bool isValidationTurn,
        IReadOnlyList<float[]> bootstrapAudio,
        out bool speechAlreadyEnded,
        out TimeSpan speechDuration);

    bool CanEnterCaptureRecognizing(
        long turnId,
        string asrSessionId,
        AssistantState conversationState,
        out bool isValidationTurn);

    bool CommitEnterCaptureRecognizing(
        long turnId,
        string asrSessionId,
        bool isValidationTurn);

    bool CanObserveCaptureStartFailure(long turnId, AssistantState conversationState);

    bool CommitCaptureStartFailure(long turnId);

    void PrepareCaptureTurn(long turnId);

    void ClearCaptureSessionId(long turnId);

    string? TakeAbandonedCaptureSession(long activeTurnId, long validationTurnId);
}
