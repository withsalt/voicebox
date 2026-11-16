namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Enrollment statistics kept for diagnostics; never used for authorization.</summary>
internal sealed record OwnerSpeakerProfileQuality(
    int SegmentCount,
    int ConfirmationCount,
    double PairwiseP10,
    double PairwiseMedian,
    double ConfirmationMinScore,
    double WithinSpeakerSpread)
{
    public static OwnerSpeakerProfileQuality Empty { get; } = new(0, 0, 0, 0, 0, 0);
}
