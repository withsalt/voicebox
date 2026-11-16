using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal readonly record struct SpeakerEnrollmentSegmentEvaluation(
    SpeakerEnrollmentReasonCode Reason,
    SpeakerEnrollmentSegmentQuality Quality,
    SpeakerEnrollmentSegmentEmbedding? Embedding)
{
    public bool Accepted => Reason == SpeakerEnrollmentReasonCode.SegmentAccepted;
}
