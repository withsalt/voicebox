using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

internal interface IRenderReferenceFrameSink
{
    void OnRenderReference(in DuplexAudioFrame frame);
}
