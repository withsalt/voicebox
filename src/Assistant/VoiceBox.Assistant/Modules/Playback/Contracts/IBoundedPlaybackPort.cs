namespace VoiceBox.Assistant.Modules.Playback.Contracts;

internal interface IBoundedPlaybackPort
{
    bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples);

    bool IsActive { get; }

    string StateName { get; }

    int CurrentQueueSize { get; }

    bool IsPlaying { get; }
}
