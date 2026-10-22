namespace VoiceBox.Assistant.Modules.Speaker.Models;

public sealed record SpeakerEnrollmentResult(
    string ProfilePath,
    string VariantId,
    string FrontendFingerprint,
    int CleanPrototypeCount,
    int DoubleTalkPrototypeCount,
    int CleanSegmentsRetained,
    int DoubleTalkSegmentsRetained,
    int RetriedSegments,
    int DiscardedSegments,
    double PairwiseMedian,
    double PairwiseP10,
    double ConfirmationMinScore,
    double WithinSpeakerSpread,
    float Threshold);
