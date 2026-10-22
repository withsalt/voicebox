using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Models;

/// <summary>
/// One identity-tap frame: capture audio after the capture processor / AEC / APM stage and before
/// any independent DNR. <see cref="Audio"/> is borrowed from the duplex callback and is only valid
/// while the synchronous router dispatch runs; the speaker timeline copies it immediately.
/// </summary>
internal readonly record struct SpeakerCaptureFrame(
    CaptureRange SourceRange,
    uint SampleRate,
    uint Channels,
    ReadOnlyMemory<float> Audio,
    bool HasDiscontinuity,
    string FrontendFingerprint)
{
    public bool IsValid =>
        SourceRange.IsValid &&
        !Audio.IsEmpty &&
        SourceRange.Length == Audio.Length;
}
