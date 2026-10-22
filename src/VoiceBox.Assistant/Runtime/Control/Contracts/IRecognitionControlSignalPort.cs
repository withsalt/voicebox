using VoiceBox.Assistant.Modules.Recognition.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

internal interface IRecognitionControlSignalPort
{
    ValueTask<bool> PublishRecognitionStartedAsync(
        long turnId,
        long operationId,
        string sessionId,
        CancellationToken cancellationToken);

    ValueTask PublishRecognitionFailedAsync(
        RecognitionOutcome outcome,
        CancellationToken cancellationToken);
}
