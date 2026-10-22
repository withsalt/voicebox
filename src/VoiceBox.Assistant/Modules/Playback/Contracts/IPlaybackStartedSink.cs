namespace VoiceBox.Assistant.Modules.Playback.Contracts;

internal interface IPlaybackStartedSink
{
    void OnPlaybackStarted(long turnId);
}
