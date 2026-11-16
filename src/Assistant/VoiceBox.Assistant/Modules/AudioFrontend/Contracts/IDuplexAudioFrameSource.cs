using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>
/// The one duplex-frame subscription a capable frontend exposes. The router fans each frame out to
/// the identity tap first and the render-aware detector second; there is never a second copy of
/// the subscription per consumer.
/// </summary>
internal interface IDuplexAudioFrameSource
{
    event DuplexAudioFrameHandler? OnDuplexAudioFrame;
}
