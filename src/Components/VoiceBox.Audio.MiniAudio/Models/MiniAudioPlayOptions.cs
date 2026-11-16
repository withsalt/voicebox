namespace VoiceBox.Audio.MiniAudio.Models;

public sealed class MiniAudioPlayOptions : MiniAudioOptionsBase
{
    public uint MaxQueueSize { get; set; } = 300;
}
