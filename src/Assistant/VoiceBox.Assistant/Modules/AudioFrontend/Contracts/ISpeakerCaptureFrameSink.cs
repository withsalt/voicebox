using VoiceBox.Assistant.Modules.AudioFrontend.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>Identity-tap consumer. Called synchronously on the duplex dispatch thread; must only copy.</summary>
internal interface ISpeakerCaptureFrameSink
{
    void OnSpeakerCapture(in SpeakerCaptureFrame frame);
}
