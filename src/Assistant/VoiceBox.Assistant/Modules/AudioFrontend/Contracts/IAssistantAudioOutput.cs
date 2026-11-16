using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>
/// Typed playback port: negotiated format, bounded enqueue, drain observation, queue clearing and
/// gain. It is the only playback surface the Runtime and the playback effect handler consume.
/// </summary>
internal interface IAssistantAudioOutput
{
    uint SampleRate { get; }

    uint ChannelCount { get; }

    SampleFormat SampleFormat { get; }

    bool IsPlaying { get; }

    int CurrentQueueSize { get; }

    AudioServiceState State { get; }

    bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples);

    void ClearQueue();

    void SetGain(float gain);
}
