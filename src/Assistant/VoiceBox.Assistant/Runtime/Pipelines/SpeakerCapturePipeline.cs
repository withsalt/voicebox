using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Runtime.Pipelines;

/// <summary>Samples real playback ownership before identity prefetch, including TTS sentence gaps.</summary>
internal sealed class SpeakerCapturePipeline(ISpeakerModule speaker, IPlaybackActivityView playback) : ISpeakerCaptureFrameSink
{
    public void OnSpeakerCapture(in SpeakerCaptureFrame frame)
    {
        speaker.SetPlaybackActive(playback.GetActivitySnapshot().IsActive);
        speaker.OnSpeakerCapture(in frame);
    }
}
