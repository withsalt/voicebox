namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Stable reason codes reported with progress and failures; applications map them to prompts.</summary>
public enum SpeakerEnrollmentReasonCode
{
    None,
    SegmentAccepted,
    LowEnergy,
    Clipping,
    DcOffset,
    InsufficientSpeech,
    TimelineGap,
    NoRenderDuringDoubleTalk,
    InsufficientNearEndDuringDoubleTalk,
    InvalidEmbedding,
    LowWithinSegmentConsistency,
    OutlierSegment,
    SegmentTimeout,
    TooManyRetries,
    NotEnoughSegments,
    ConfirmationFailed,
    FrontendNotSupported,
    StimulusPlayerRequired,
}
