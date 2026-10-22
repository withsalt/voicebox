using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.HPFService;

namespace VoiceBox.HPF.Butterworth;

/// <summary>
/// Applies an even-order Butterworth high-pass filter to interleaved floating-point audio.
/// </summary>
/// <remarks>
/// The filter is implemented as cascaded RBJ biquads in transposed direct form II. Processing
/// is allocation-free while the audio format remains unchanged. Instances are stateful and must
/// not be processed, reset, or disposed concurrently.
/// </remarks>
public sealed class ButterworthHighPassFilterService : IHighPassFilterService
{
    private const double DenormalThreshold = 1e-300;

    private readonly double _cutoffFrequency;
    private readonly int _order;

    private uint _sampleRate;
    private uint _channelCount;
    private int _sectionCount;

    private double[] _b0 = Array.Empty<double>();
    private double[] _b1 = Array.Empty<double>();
    private double[] _b2 = Array.Empty<double>();
    private double[] _a1 = Array.Empty<double>();
    private double[] _a2 = Array.Empty<double>();
    private double[] _z1 = Array.Empty<double>();
    private double[] _z2 = Array.Empty<double>();

    private bool _disposed;

    /// <summary>
    /// Initializes a Butterworth high-pass filter.
    /// </summary>
    public ButterworthHighPassFilterService(IOptions<ButterworthHighPassFilterOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);

        ButterworthHighPassFilterOptions value = options.Value
            ?? throw new ArgumentException("High-pass filter options cannot be null.", nameof(options));
        value.Validate();

        _cutoffFrequency = value.CutoffFrequency;
        _order = value.Order;
        Configure(value.SampleRate, value.ChannelCount);
    }

    /// <inheritdoc />
    public void Process(Span<float> capture, uint channels, uint sampleRate)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ButterworthHighPassFilterValidation.ValidateRuntimeFormat(sampleRate, channels, _cutoffFrequency);

        int channelCount = checked((int)channels);
        if (capture.Length % channelCount != 0)
        {
            throw new ArgumentException(
                "The sample count must be divisible by the number of interleaved channels.",
                nameof(capture));
        }

        if (capture.IsEmpty)
        {
            return;
        }

        if (channels != _channelCount || sampleRate != _sampleRate)
        {
            Configure(sampleRate, channels);
        }

        int frameCount = capture.Length / channelCount;
        for (int section = 0; section < _sectionCount; section++)
        {
            double b0 = _b0[section];
            double b1 = _b1[section];
            double b2 = _b2[section];
            double a1 = _a1[section];
            double a2 = _a2[section];
            int stateOffset = section * channelCount;

            for (int channel = 0; channel < channelCount; channel++)
            {
                double z1 = _z1[stateOffset + channel];
                double z2 = _z2[stateOffset + channel];

                for (int frame = 0; frame < frameCount; frame++)
                {
                    int sampleIndex = frame * channelCount + channel;
                    double input = capture[sampleIndex];
                    double output = b0 * input + z1;
                    z1 = b1 * input - a1 * output + z2;
                    z2 = b2 * input - a2 * output;
                    capture[sampleIndex] = (float)output;
                }

                _z1[stateOffset + channel] = ClearDenormal(z1);
                _z2[stateOffset + channel] = ClearDenormal(z2);
            }
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Array.Clear(_z1);
        Array.Clear(_z2);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _disposed = true;
    }

    private void Configure(uint sampleRate, uint channels)
    {
        int sectionCount = _order / 2;
        int channelCount = checked((int)channels);

        if (_b0.Length != sectionCount)
        {
            _b0 = new double[sectionCount];
            _b1 = new double[sectionCount];
            _b2 = new double[sectionCount];
            _a1 = new double[sectionCount];
            _a2 = new double[sectionCount];
        }

        int stateLength = checked(sectionCount * channelCount);
        if (_z1.Length != stateLength)
        {
            _z1 = new double[stateLength];
            _z2 = new double[stateLength];
        }
        else
        {
            Array.Clear(_z1);
            Array.Clear(_z2);
        }

        double angularCutoff = 2.0 * Math.PI * _cutoffFrequency / sampleRate;
        double cosine = Math.Cos(angularCutoff);
        double sine = Math.Sin(angularCutoff);

        for (int section = 0; section < sectionCount; section++)
        {
            double q = 1.0 / (2.0 * Math.Cos(Math.PI * (2.0 * section + 1.0) / (2.0 * _order)));
            double alpha = sine / (2.0 * q);
            double a0 = 1.0 + alpha;

            _b0[section] = ((1.0 + cosine) / 2.0) / a0;
            _b1[section] = -(1.0 + cosine) / a0;
            _b2[section] = ((1.0 + cosine) / 2.0) / a0;
            _a1[section] = (-2.0 * cosine) / a0;
            _a2[section] = (1.0 - alpha) / a0;
        }

        _sectionCount = sectionCount;
        _sampleRate = sampleRate;
        _channelCount = channels;
    }

    private static double ClearDenormal(double value)
        => Math.Abs(value) < DenormalThreshold ? 0.0 : value;
}
