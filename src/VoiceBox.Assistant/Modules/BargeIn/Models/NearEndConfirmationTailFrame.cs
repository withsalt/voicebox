namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// One bounded confirmation-tail range. Only timeline metadata is queued; the PCM stays in the
/// timeline reservoir and is snapshotted when the replay is drained.
/// </summary>
internal readonly record struct NearEndConfirmationTailFrame(
    long Generation,
    long StreamEpoch,
    long StartSample,
    long EndSampleExclusive,
    bool HasDiscontinuity,
    bool LatestSpeechActive,
    double DurationMs);
