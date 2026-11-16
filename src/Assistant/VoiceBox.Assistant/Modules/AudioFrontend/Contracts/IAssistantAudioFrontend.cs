using VoiceBox.Assistant.Modules.AudioFrontend.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>
/// Static audio frontend selected during Build. It is deliberately not a service locator and
/// exposes no capability lookup: the Runtime receives typed input/output ports plus whatever
/// optional timeline sources the provider genuinely supports.
/// </summary>
internal interface IAssistantAudioFrontend
{
    IAssistantAudioInput Input { get; }

    IAssistantAudioOutput Output { get; }

    /// <summary>Processed capture timeline; present only when declared by the provider.</summary>
    IProcessedCaptureSource? ProcessedCapture { get; }

    /// <summary>Render reference timeline; present only when declared by the provider.</summary>
    IRenderReferenceSource? RenderReference { get; }

    /// <summary>
    /// Duplex frames carrying the identity tap (after capture processor / AEC / APM, before any
    /// independent DNR) together with their own source timeline; present only when declared.
    /// </summary>
    IDuplexAudioFrameSource? DuplexFrames { get; }

    /// <summary>Frozen description of the identity tap; null when the tap is not supported.</summary>
    SpeakerCaptureDescriptor? SpeakerCapture { get; }

    bool SupportsSimultaneousCapturePlayback { get; }

    bool SupportsRenderReference { get; }

    bool SupportsProcessedCaptureTimeline { get; }

    bool SupportsSpeakerCaptureTimeline { get; }

    Task StartAsync(CancellationToken cancellationToken = default);

    Task StopAsync(CancellationToken cancellationToken = default);
}
