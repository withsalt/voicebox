using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Playback.Adapters;

internal sealed class BoundedPlaybackPortAdapter(IAssistantAudioOutput playback) :
    IBoundedPlaybackPort,
    IPlaybackDeviceQueuePort
{
    private readonly IAssistantAudioOutput _playback = playback
        ?? throw new ArgumentNullException(nameof(playback));

    public bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples)
        => _playback.TryEnqueue(samples, out droppedSamples);

    public bool IsActive => _playback.State == AudioServiceState.Active;

    public string StateName => _playback.State.ToString();

    public int CurrentQueueSize => _playback.CurrentQueueSize;

    public bool IsPlaying => _playback.IsPlaying;

    public void ClearQueue() => _playback.ClearQueue();

    public void SetGain(float gain) => _playback.SetGain(gain);
}
