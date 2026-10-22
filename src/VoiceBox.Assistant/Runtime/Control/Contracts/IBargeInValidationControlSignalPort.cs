using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Control-plane admission for the whole-utterance recognition outcome of an accepted interruption.
/// There is no timeout entry: recognition never authorizes or times a physical stop.
/// </summary>
internal interface IBargeInValidationControlSignalPort
{
    ValueTask<BargeInValidationControlResolution> ResolveValidationAsync(
        long turnId,
        string? text,
        string reason,
        bool meaningful,
        CancellationToken cancellationToken);

    ValueTask PublishValidatedRecognitionCompletedAsync(
        long turnId,
        string text,
        string? rawText,
        TimeSpan speechDuration,
        CancellationToken cancellationToken);

    ValueTask PublishValidationCancelledAsync(
        long turnId,
        CancellationToken cancellationToken);
}
