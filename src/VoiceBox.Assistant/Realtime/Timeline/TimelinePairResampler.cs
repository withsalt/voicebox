namespace VoiceBox.Assistant.Realtime.Timeline;

/// <summary>
/// Stateful mono pair resampler for microphone/render evidence. The state is stream-local;
/// callers reset it on format, epoch, or discontinuity changes.
/// </summary>
internal sealed class TimelinePairResampler
{
    internal delegate void ResampledPairHandler(float microphone, float render);

    private uint _sampleRate;
    private uint _channels;
    private long _inputPosition;
    private double _nextOutputPosition;
    private float _previousMicrophone;
    private float _previousRender;
    private bool _hasPrevious;

    public void Append(
        ReadOnlySpan<float> microphone,
        ReadOnlySpan<float> render,
        uint channels,
        uint sampleRate,
        ResampledPairHandler handler)
    {
        if (channels == 0 || sampleRate == 0 || microphone.Length != render.Length || microphone.Length % channels != 0)
        {
            throw new ArgumentException("Duplex signals must have matching interleaved frames.");
        }
        if (_sampleRate != 0 && (_sampleRate != sampleRate || _channels != channels))
        {
            throw new InvalidOperationException("The resampler must be reset before changing input format.");
        }
        if (_sampleRate == 0)
        {
            _sampleRate = sampleRate;
            _channels = channels;
        }
        int num = microphone.Length / (int)channels;
        long num2 = checked(_inputPosition + num);
        double num3 = (double)sampleRate / 16000.0;
        while (_nextOutputPosition + 1.0 < (double)num2)
        {
            long num4 = (long)Math.Floor(_nextOutputPosition);
            double num5 = _nextOutputPosition - (double)num4;
            float num6 = Read(microphone, num4, _inputPosition, channels, _previousMicrophone, _hasPrevious);
            float num7 = Read(render, num4, _inputPosition, channels, _previousRender, _hasPrevious);
            float num8 = Read(microphone, num4 + 1, _inputPosition, channels, 0f, hasPrevious: false);
            float num9 = Read(render, num4 + 1, _inputPosition, channels, 0f, hasPrevious: false);
            handler(num6 + (num8 - num6) * (float)num5, num7 + (num9 - num7) * (float)num5);
            _nextOutputPosition += num3;
        }
        if (num > 0)
        {
            _previousMicrophone = Mix(microphone, num - 1, channels);
            _previousRender = Mix(render, num - 1, channels);
            _hasPrevious = true;
        }
        _inputPosition = num2;
    }

    public void Reset()
    {
        _sampleRate = 0u;
        _channels = 0u;
        _inputPosition = 0L;
        _nextOutputPosition = 0.0;
        _previousMicrophone = 0f;
        _previousRender = 0f;
        _hasPrevious = false;
    }

    private static float Read(
        ReadOnlySpan<float> samples,
        long position,
        long inputStart,
        uint channels,
        float previous,
        bool hasPrevious)
    {
        if (position == inputStart - 1 && hasPrevious)
        {
            return previous;
        }
        int frameIndex = checked((int)(position - inputStart));
        return Mix(samples, frameIndex, channels);
    }

    private static float Mix(ReadOnlySpan<float> samples, int frameIndex, uint channels)
    {
        // Mono is the dominant capture format and the general path costs a loop, a widen
        // and a divide per sample. Dividing by exactly 1.0 is exact and the float
        // round-trip through double is lossless, so this returns the identical bits.
        if (channels == 1)
        {
            float mono = samples[frameIndex];
            return float.IsFinite(mono) ? mono : 0f;
        }

        int num = checked(frameIndex * (int)channels);
        double num2 = 0.0;
        for (int i = 0; i < channels; i++)
        {
            float num3 = samples[num + i];
            num2 += (double)(float.IsFinite(num3) ? num3 : 0f);
        }
        return (float)(num2 / (double)channels);
    }
}
