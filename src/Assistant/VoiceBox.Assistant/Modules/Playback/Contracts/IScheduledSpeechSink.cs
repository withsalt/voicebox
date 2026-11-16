namespace VoiceBox.Assistant.Modules.Playback.Contracts;

internal interface IScheduledSpeechSink
{
    void OnSpeechScheduled(long turnId, string text);
}
