namespace VoiceBox.DTD.RenderAware.Models;

/// <summary>One render-aware near-end speech observation.</summary>
/// <param name="Probability">Raw model probability in [0, 1].</param>
/// <param name="SmoothedProbability">Moving-average probability in [0, 1].</param>
/// <param name="IsNearEndSpeech">
/// Whether <paramref name="SmoothedProbability"/> meets the configured decision threshold.
/// </param>
/// <param name="FrameIndex">One-based output frame index since construction or reset.</param>
/// <param name="Timestamp">
/// Start time of the newest input hop represented by this observation.
/// </param>
/// <param name="FrameDuration">Duration advanced by each observation.</param>
public readonly record struct RenderAwareDtdObservation(
    float Probability,
    float SmoothedProbability,
    bool IsNearEndSpeech,
    long FrameIndex,
    TimeSpan Timestamp,
    TimeSpan FrameDuration);
