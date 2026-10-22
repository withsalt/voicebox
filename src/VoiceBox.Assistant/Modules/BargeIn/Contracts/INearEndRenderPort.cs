using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Render-reference operations used by the duplex callback.</summary>
internal interface INearEndRenderPort
{
    ActiveNearEndProcessingState GetProcessingState();

    ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot();

    void ObserveDuplexAudio(
        in DuplexAudioFrame frame,
        bool playbackStartupGuardActive,
        bool latestNearEndSpeech = false);
}
