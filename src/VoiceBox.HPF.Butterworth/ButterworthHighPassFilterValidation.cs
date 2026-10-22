namespace VoiceBox.HPF.Butterworth;

internal static class ButterworthHighPassFilterValidation
{
    private const uint MinimumSampleRate = 8000;
    private const uint MaximumSampleRate = 192000;
    private const uint MaximumChannelCount = 8;

    public static void ValidateOptions(ButterworthHighPassFilterOptions options)
    {
        ValidateSampleRate(options.SampleRate, nameof(options.SampleRate));
        ValidateChannelCount(options.ChannelCount, nameof(options.ChannelCount));
        ValidateCutoffFrequency(options.CutoffFrequency, options.SampleRate);

        if (options.Order is < 2 or > 8 || options.Order % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options.Order),
                options.Order,
                "Order must be an even number between 2 and 8.");
        }
    }

    public static void ValidateRuntimeFormat(uint sampleRate, uint channelCount, double cutoffFrequency)
    {
        ValidateSampleRate(sampleRate, nameof(sampleRate));
        ValidateChannelCount(channelCount, nameof(channelCount));

        if (cutoffFrequency >= sampleRate / 2.0)
        {
            throw new InvalidOperationException(
                $"The configured cutoff frequency ({cutoffFrequency} Hz) must be below the " +
                $"Nyquist frequency for the runtime sample rate ({sampleRate} Hz).");
        }
    }

    private static void ValidateSampleRate(uint sampleRate, string parameterName)
    {
        if (sampleRate is < MinimumSampleRate or > MaximumSampleRate)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                sampleRate,
                $"Sample rate must be between {MinimumSampleRate} and {MaximumSampleRate} Hz.");
        }
    }

    private static void ValidateChannelCount(uint channelCount, string parameterName)
    {
        if (channelCount is < 1 or > MaximumChannelCount)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                channelCount,
                $"Channel count must be between 1 and {MaximumChannelCount}.");
        }
    }

    private static void ValidateCutoffFrequency(double cutoffFrequency, uint sampleRate)
    {
        double nyquistFrequency = sampleRate / 2.0;
        if (!double.IsFinite(cutoffFrequency) || cutoffFrequency <= 0 || cutoffFrequency >= nyquistFrequency)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ButterworthHighPassFilterOptions.CutoffFrequency),
                cutoffFrequency,
                $"Cutoff frequency must be finite and between 0 and {nyquistFrequency} Hz.");
        }
    }
}
