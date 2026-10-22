using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Exceptions;

/// <summary>Enrollment ended without a saved profile; <see cref="Reason"/> is the stable code.</summary>
public sealed class SpeakerEnrollmentException : InvalidOperationException
{
    public SpeakerEnrollmentException(SpeakerEnrollmentReasonCode reason, string message)
        : this(reason, message, SpeakerEnrollmentStage.Failed, 0)
    {
    }

    public SpeakerEnrollmentException(
        SpeakerEnrollmentReasonCode reason,
        string message,
        SpeakerEnrollmentStage stage,
        int segmentIndex)
        : base(message)
    {
        Reason = reason;
        Stage = stage;
        SegmentIndex = segmentIndex;
    }

    public SpeakerEnrollmentReasonCode Reason { get; }
    public SpeakerEnrollmentStage Stage { get; }
    public int SegmentIndex { get; }
}
