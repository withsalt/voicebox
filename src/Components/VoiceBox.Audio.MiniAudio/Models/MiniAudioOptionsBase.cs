using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio.Models;

public abstract class MiniAudioOptionsBase : AudioOptionsBase
{
    protected MiniAudioOptionsBase()
    {
        BackendType = BackendType.MiniAudio;
    }

    /// <summary>
    /// Gets or sets the requested device period in frames. When <see cref="FixedSizedCallback"/>
    /// is false, this is a hint, not a fixed callback size or an upper bound.
    /// Use -1 to derive the period from <see cref="SuggestedLatency"/> (10 ms by default),
    /// or 0 to use the backend default when no latency is specified.
    /// </summary>
    public int MaxFrameSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets whether callbacks must use a fixed frame count.
    /// Defaults to false, allowing the backend to choose each callback's frame count.
    /// </summary>
    /// <remarks>
    /// Applies to playback, capture, and duplex devices when the device is initialized.
    /// The default bypasses miniaudio's fixed-size intermediary buffers, including the
    /// duplex restart buffering issue in miniaudio 0.11.26.
    /// </remarks>
    public bool FixedSizedCallback { get; set; }

    public double? SuggestedLatency { get; set; }

    public bool EnableWasapiExclusiveMode { get; set; }
}
