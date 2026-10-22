namespace VoiceBox.Assistant.Modules.Speaker.Models;

public sealed class SpeakerEnrollmentProgressEventArgs : EventArgs
{
    public SpeakerEnrollmentProgressKind Kind { get; init; }
    public required SpeakerEnrollmentStage Stage { get; init; }
    public SpeakerEnrollmentCondition Condition { get; init; }
    public int SegmentIndex { get; init; }
    public int SegmentCount { get; init; }
    public int Attempt { get; init; }
    public TimeSpan Collected { get; init; }
    public TimeSpan Required { get; init; }
    public SpeakerEnrollmentSegmentQuality? QualitySnapshot { get; init; }
    public SpeakerEnrollmentReasonCode ReasonCode { get; init; }
}
