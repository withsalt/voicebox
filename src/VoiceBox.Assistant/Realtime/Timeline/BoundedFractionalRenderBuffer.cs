namespace VoiceBox.Assistant.Realtime.Timeline;

internal sealed class BoundedFractionalRenderBuffer
{
    private readonly float[] _samples;

    private readonly bool[] _validity;

    private int _writeIndex;

    private int _count;

    private long _startSample;

    private long _endSampleExclusive;

    public int Count => _count;

    public long StartSample => _startSample;

    public long EndSampleExclusive => _endSampleExclusive;

    public BoundedFractionalRenderBuffer(int capacity)
    {
        if (capacity < 2)
        {
            throw new ArgumentOutOfRangeException("capacity", "Capacity must be at least two samples.");
        }
        _samples = new float[capacity];
        _validity = new bool[capacity];
    }

    public void Append(ReadOnlySpan<float> samples, long startSample, bool isValid = true)
    {
        if (startSample < 0)
        {
            throw new ArgumentOutOfRangeException("startSample");
        }
        if (samples.IsEmpty)
        {
            return;
        }
        if (_count != 0 && startSample != _endSampleExclusive)
        {
            throw new InvalidOperationException("Render samples must be appended on a continuous timeline.");
        }
        if (_count == 0)
        {
            _startSample = startSample;
            _endSampleExclusive = startSample;
            _writeIndex = 0;
        }
        int i = 0;
        if (samples.Length > _samples.Length)
        {
            i = samples.Length - _samples.Length;
            Reset();
            _startSample = checked(startSample + i);
            _endSampleExclusive = _startSample;
        }
        for (; i < samples.Length; i++)
        {
            _samples[_writeIndex] = samples[i];
            _validity[_writeIndex] = isValid;
            _writeIndex = (_writeIndex + 1) % _samples.Length;
            if (_count < _samples.Length)
            {
                _count++;
            }
            else
            {
                _startSample++;
            }
            _endSampleExclusive++;
        }
    }

    public void Append(float sample, long startSample, bool isValid = true)
    {
        if (startSample < 0)
        {
            throw new ArgumentOutOfRangeException("startSample");
        }
        if (_count != 0 && startSample != _endSampleExclusive)
        {
            throw new InvalidOperationException("Render samples must be appended on a continuous timeline.");
        }
        if (_count == 0)
        {
            _startSample = startSample;
            _endSampleExclusive = startSample;
            _writeIndex = 0;
        }
        _samples[_writeIndex] = sample;
        _validity[_writeIndex] = isValid;
        // This runs once per resampled sample (16 kHz), so the wrap is a compare rather
        // than an integer division. The resulting index sequence is identical.
        if (++_writeIndex == _samples.Length)
        {
            _writeIndex = 0;
        }
        if (_count < _samples.Length)
        {
            _count++;
        }
        else
        {
            _startSample++;
        }
        _endSampleExclusive++;
    }

    public bool TryRead(double startSample, Span<float> destination)
        => TryRead(startSample, destination, out _);

    public bool TryRead(
        double startSample,
        Span<float> destination,
        out bool intersectsInvalidSamples)
    {
        intersectsInvalidSamples = false;
        if (!double.IsFinite(startSample) || destination.IsEmpty || _count == 0)
        {
            return false;
        }
        long num;
        double num2;
        checked
        {
            num = (long)Math.Floor(startSample);
            num2 = startSample - (double)num;
            long num3 = num + destination.Length - 1;
            long num4 = num3 + ((num2 > 1E-09) ? 2 : 1);
            if (num < _startSample || num4 > _endSampleExclusive)
            {
                return false;
            }
        }
        // The window start is resolved once instead of re-deriving the oldest index (two
        // integer divisions) for every sample and neighbour. Indices advance by compare,
        // so the visited slots and the produced values are identical to the per-sample
        // Read/IsValid form.
        int capacity = _samples.Length;
        int oldest = (_writeIndex - _count + capacity) % capacity;
        int index = oldest + checked((int)(num - _startSample));
        if (index >= capacity)
        {
            index -= capacity;
        }
        bool interpolate = num2 > 1E-09;
        float[] samples = _samples;
        bool[] validity = _validity;
        for (int i = 0; i < destination.Length; i++)
        {
            int next = index + 1 == capacity ? 0 : index + 1;
            if (!validity[index] || (interpolate && !validity[next]))
            {
                intersectsInvalidSamples = true;
                return false;
            }
            float num6 = samples[index];
            if (!interpolate)
            {
                destination[i] = num6;
            }
            else
            {
                float num7 = samples[next];
                destination[i] = num6 + (num7 - num6) * (float)num2;
            }
            index = next;
        }
        return true;
    }

    public void Reset()
    {
        _writeIndex = 0;
        _count = 0;
        _startSample = 0L;
        _endSampleExclusive = 0L;
    }

    private float Read(long sample)
    {
        long num = sample - _startSample;
        int num2 = (_writeIndex - _count + _samples.Length) % _samples.Length;
        int num3 = (num2 + checked((int)num)) % _samples.Length;
        return _samples[num3];
    }

    private bool IsValid(long sample)
    {
        long offset = sample - _startSample;
        int oldest = (_writeIndex - _count + _samples.Length) % _samples.Length;
        int index = (oldest + checked((int)offset)) % _samples.Length;
        return _validity[index];
    }
}
