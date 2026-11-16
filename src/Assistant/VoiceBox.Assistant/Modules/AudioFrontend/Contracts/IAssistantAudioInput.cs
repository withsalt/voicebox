using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>
/// Typed capture port. The Runtime observes ordered frames and the negotiated format through this
/// boundary and never reaches a concrete audio provider.
/// </summary>
internal interface IAssistantAudioInput
{
    uint SampleRate { get; }

    uint ChannelCount { get; }

    SampleFormat SampleFormat { get; }

    bool IsCapturing { get; }
}
