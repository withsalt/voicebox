namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Per-segment quality statistics; every value is finite and every ratio lies in [0, 1].</summary>
public sealed record SpeakerEnrollmentSegmentQuality(
    TimeSpan Duration,
    double Rms,
    double Peak,
    double ClippingRatio,
    double DcOffset,
    double SpeechRatio,
    double RenderActiveRatio,
    double NearEndAcceptedRatio,
    int EmbeddingWindowCount,
    double WithinSegmentSimilarityMedian,
    double SimilarityToBankMedoid)
{
    public static SpeakerEnrollmentSegmentQuality Empty { get; } =
        new(TimeSpan.Zero, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}
