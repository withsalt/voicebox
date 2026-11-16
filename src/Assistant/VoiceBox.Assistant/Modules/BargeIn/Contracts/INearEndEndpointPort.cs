using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Endpoint observations emitted by the VAD ingress and consumed on the control loop.</summary>
internal interface INearEndEndpointPort
{
    ActiveNearEndProcessingState GetProcessingState();

    void ObserveEndpointSpeechStarted();

    void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio);

    void ObserveUnownedEndpointSpeechEnded();

    void ObserveEndpointSpeechEnded(bool captureCommittedAfterEndpoint = false);
}
