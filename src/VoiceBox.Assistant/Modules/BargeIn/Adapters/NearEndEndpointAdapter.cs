using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Endpoint-VAD facet of the near-end data plane.</summary>
internal sealed class NearEndEndpointAdapter : INearEndEndpointPort
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndEndpointAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public ActiveNearEndProcessingState GetProcessingState()
        => _dataPlane.GetProcessingState();

    public void ObserveEndpointSpeechStarted()
        => _dataPlane.ObserveEndpointSpeechStarted();

    public void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio)
        => _dataPlane.ObserveEndpointSpeechAudio(audio);

    public void ObserveUnownedEndpointSpeechEnded()
        => _dataPlane.ObserveUnownedEndpointSpeechEnded();

    public void ObserveEndpointSpeechEnded(bool captureCommittedAfterEndpoint = false)
        => _dataPlane.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint);
}
