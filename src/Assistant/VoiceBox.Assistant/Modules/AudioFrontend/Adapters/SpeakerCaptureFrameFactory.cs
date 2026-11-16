using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Adapters;

/// <summary>
/// Builds identity frames from the duplex callback without allocating per frame. The descriptor is
/// frozen at Build time so the fingerprint string is shared by every frame.
/// </summary>
internal sealed class SpeakerCaptureFrameFactory
{
    private readonly SpeakerCaptureDescriptor _descriptor;

    public SpeakerCaptureFrameFactory(SpeakerCaptureDescriptor descriptor)
        => _descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));

    public SpeakerCaptureDescriptor Descriptor => _descriptor;

    /// <summary>
    /// Slices the after-processor signal of a duplex frame onto its own source timeline. Returns
    /// false when the provider exposed no valid after-AEC timeline for this callback.
    /// </summary>
    public bool TryCreate(in DuplexAudioFrame frame, out SpeakerCaptureFrame identity)
    {
        CaptureSignalTimeline timeline = frame.CaptureAfterAecTimeline;
        ReadOnlyMemory<float> audio = frame.CaptureAfterAecValid;
        if (!timeline.IsValid || audio.IsEmpty || timeline.SourceRange.Length != audio.Length)
        {
            identity = default;
            return false;
        }

        identity = new SpeakerCaptureFrame(
            timeline.SourceRange,
            frame.SampleRate,
            frame.Channels,
            audio,
            frame.HasDiscontinuity,
            _descriptor.FrontendFingerprint);
        return true;
    }
}
