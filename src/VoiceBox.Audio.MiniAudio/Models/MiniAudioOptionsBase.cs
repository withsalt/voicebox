using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio.Models;

public abstract class MiniAudioOptionsBase : AudioOptionsBase
{
    protected MiniAudioOptionsBase()
    {
        BackendType = BackendType.MiniAudio;
    }

    public int MaxFrameSize { get; set; } = -1;

    public double? SuggestedLatency { get; set; }

    public bool EnableWasapiExclusiveMode { get; set; }
}
