namespace VoiceBox.HPF.Butterworth;

/// <summary>
/// Configures a Butterworth high-pass filter.
/// </summary>
public sealed class ButterworthHighPassFilterOptions
{
    /// <summary>
    /// Gets or sets the initial sample rate in Hz. The default is 16000.
    /// </summary>
    public uint SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the initial number of interleaved channels. The default is one.
    /// </summary>
    public uint ChannelCount { get; set; } = 1;

    /// <summary>
    /// Gets or sets the cutoff frequency in Hz. The default is 100.
    /// </summary>
    public double CutoffFrequency { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets the filter order. Supported values are even numbers from 2 through 8.
    /// The default is 2.
    /// </summary>
    public int Order { get; set; } = 2;

    internal void Validate()
        => ButterworthHighPassFilterValidation.ValidateOptions(this);
}
