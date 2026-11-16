using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Exceptions;

/// <summary>A segment attempt failed for a typed, retryable reason.</summary>
internal sealed class SpeakerEnrollmentSegmentException : Exception
{
    public SpeakerEnrollmentSegmentException(SpeakerEnrollmentReasonCode reason, string message)
        : base(message)
        => Reason = reason;

    public SpeakerEnrollmentReasonCode Reason { get; }
}
