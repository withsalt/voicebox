using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.DTD.RenderAware;

/// <summary>
/// Stateful streaming double-talk detector that observes microphone and render audio together.
/// One instance represents one audio stream.
/// </summary>
public interface IRenderAwareDtdService : IDisposable
{
    /// <summary>Required input hop size in samples.</summary>
    int HopLength { get; }

    /// <summary>Model analysis window size in samples.</summary>
    int WindowLength { get; }

    /// <summary>Required audio sample rate.</summary>
    uint SampleRate { get; }

    /// <summary>Required audio channel count.</summary>
    uint ChannelCount { get; }

    /// <summary>
    /// Adds one synchronized microphone/render hop and returns the resulting observation.
    /// Returns <see langword="null"/> while the overlap window is not ready; this normally occurs
    /// on the first call after construction or reset.
    /// </summary>
    RenderAwareDtdObservation? Process(
        ReadOnlySpan<float> microphone,
        ReadOnlySpan<float> render,
        uint channels,
        uint sampleRate);

    /// <summary>Clears overlap, recurrent, smoothing, and timeline state.</summary>
    void Reset();
}
