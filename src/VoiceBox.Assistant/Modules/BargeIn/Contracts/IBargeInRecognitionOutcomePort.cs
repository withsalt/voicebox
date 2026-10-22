namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Finishes recognition after an accepted interruption. The recognized utterance decides whether a
/// new response starts; playback has already stopped before this call.
/// </summary>
internal interface IBargeInRecognitionOutcomePort
{
    Task RunRecognitionOutcomeAsync(
        long turnId,
        string asrSessionId,
        TimeSpan speechDuration,
        CancellationToken runToken);
}
