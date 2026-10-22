using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Optional online speaker-evidence module. It consumes only the identity tap
/// (<see cref="ISpeakerCaptureFrameSink"/>), never the final processed capture, and owns the
/// timeline ring, prefetch gate, inference scheduler, evidence cache and verification policy.
/// </summary>
internal interface ISpeakerModule :
    ISpeakerEvidenceProvider,
    ISpeakerVerificationStatePort,
    ISpeakerCaptureFrameSink,
    IDisposable
{
    AssistantTimelineRequirement TimelineRequirement { get; }

    SpeakerCaptureDescriptor Descriptor { get; }

    SpeakerVerificationPolicy Policy { get; }

    void SetPlaybackActive(bool active);

    void Start();

    Task StopAsync(CancellationToken cancellationToken = default);
}
