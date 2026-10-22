using VoiceBox.Assistant.Modules.Recognition.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

internal interface IRecognitionCompletionPort
{
    Task<RecognitionOutcome> FinishAsync(
        long turnId,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken);
}
