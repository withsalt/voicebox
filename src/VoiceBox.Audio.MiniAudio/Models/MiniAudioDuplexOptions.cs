namespace VoiceBox.Audio.MiniAudio.Models;

public sealed class MiniAudioDuplexOptions : MiniAudioOptionsBase
{
    public uint RecordBufferCapacity { get; set; } = 100;

    public uint PlayMaxQueueSize { get; set; } = 300;

    public uint DuplexFrameBufferCapacity { get; set; } = 128;

    public bool EnableAec { get; set; }

    public bool EnableDnr { get; set; }

    public bool EnableHpf { get; set; }
}
