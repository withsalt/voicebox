using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Duplex-callback facet of the near-end data plane.</summary>
internal sealed class NearEndRenderAdapter : INearEndRenderPort
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndRenderAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public ActiveNearEndProcessingState GetProcessingState()
        => _dataPlane.GetProcessingState();

    public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot()
        => _dataPlane.GetDiagnosticSnapshot();

    public void ObserveDuplexAudio(
        in DuplexAudioFrame frame,
        bool playbackStartupGuardActive,
        bool latestNearEndSpeech = false)
        => _dataPlane.ObserveDuplexAudio(in frame, playbackStartupGuardActive, latestNearEndSpeech);
}
