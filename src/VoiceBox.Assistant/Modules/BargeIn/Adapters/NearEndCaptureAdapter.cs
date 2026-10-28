using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Hot-capture facet of the near-end data plane.</summary>
internal sealed class NearEndCaptureAdapter : INearEndCapturePort
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndCaptureAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame)
        => _dataPlane.ObserveRecognitionAudio(in frame);

    public void FeedFromRecognitionAudio(
        ReadOnlySpan<float> audio,
        MainAudioObservation observation)
        => _dataPlane.FeedFromRecognitionAudio(audio, observation);
}
