namespace VoiceBox.Audio.MiniAudio.Models;

public sealed class MiniAudioRecordOptions : MiniAudioOptionsBase
{
    public uint BufferCapacity { get; set; } = 100;
}
