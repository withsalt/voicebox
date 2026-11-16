using System.Numerics;

namespace VoiceBox.TTS.Hojo.Internal;

/// <summary>1920-point mixed-radix inverse FFT and the model's periodic-Hann, same-padding ISTFT.</summary>
internal sealed class HojoIstft
{
    private const int Size = 1920;
    public const int Hop = 480;
    private const int Bins = Size / 2 + 1;
    private const int Pad = (Size - Hop) / 2;
    private readonly Complex[] _roots = new Complex[Size];
    private readonly Complex[] _spectrum = new Complex[Size];
    private readonly Complex[] _time = new Complex[Size];
    private readonly float[] _window = new float[Size];

    public HojoIstft()
    {
        for (int i = 0; i < Size; i++)
        {
            double angle = 2 * Math.PI * i / Size;
            _roots[i] = new(Math.Cos(angle), Math.Sin(angle));
            _window[i] = (float)(0.5 - 0.5 * Math.Cos(angle));
        }
    }

    public float[] Decode(ReadOnlySpan<float> magnitude, ReadOnlySpan<float> phase, int frames, int startSample, int count, CancellationToken ct)
    {
        if (frames < 1 || magnitude.Length != frames * Bins || phase.Length != magnitude.Length
            || startSample < 0 || count < 0 || startSample + count > frames * Hop)
            throw new InvalidDataException("Invalid codec spectrum or ISTFT slice.");
        var sum = new double[count];
        var envelope = new double[count];
        for (int frame = 0; frame < frames; frame++)
        {
            ct.ThrowIfCancellationRequested();
            int offset = frame * Hop - Pad - startSample;
            int begin = Math.Max(0, -offset), end = Math.Min(Size, count - offset);
            if (begin >= end) continue;
            for (int bin = 0; bin < Bins; bin++)
            {
                float mag = magnitude[bin * frames + frame], angle = phase[bin * frames + frame];
                if (!float.IsFinite(mag) || !float.IsFinite(angle) || mag < 0)
                    throw new InvalidDataException("Codec returned an invalid spectrum.");
                mag = Math.Min(mag, 100);
                // Match numpy's float32 complex construction before the double precision inverse FFT.
                _spectrum[bin] = new((float)(mag * MathF.Cos(angle)), (float)(mag * MathF.Sin(angle)));
            }
            _spectrum[0] = new(_spectrum[0].Real, 0);
            _spectrum[Size / 2] = new(_spectrum[Size / 2].Real, 0);
            for (int bin = 1; bin < Size / 2; bin++) _spectrum[Size - bin] = Complex.Conjugate(_spectrum[bin]);
            Transform(_time, 0, 1, Size);
            for (int i = begin; i < end; i++)
            {
                sum[offset + i] += _time[i].Real / Size * _window[i];
                envelope[offset + i] += (float)(_window[i] * _window[i]);
            }
        }
        var result = new float[count];
        for (int i = 0; i < result.Length; i++)
        {
            if (envelope[i] <= 1e-11) throw new InvalidDataException("Invalid ISTFT window envelope.");
            result[i] = (float)(sum[i] / envelope[i]);
            if (!float.IsFinite(result[i])) throw new InvalidDataException("Non-finite audio sample.");
        }
        return result;
    }

    private void Transform(Span<Complex> output, int inputStart, int stride, int size)
    {
        if (size == 1) { output[0] = _spectrum[inputStart]; return; }
        int radix = size % 2 == 0 ? 2 : size % 3 == 0 ? 3 : 5;
        int part = size / radix;
        for (int i = 0; i < radix; i++) Transform(output.Slice(i * part, part), inputStart + i * stride, stride * radix, part);
        Span<Complex> values = stackalloc Complex[5];
        int rootStep = Size / size;
        for (int k = 0; k < part; k++)
        {
            for (int j = 0; j < radix; j++) values[j] = output[j * part + k];
            for (int q = 0; q < radix; q++)
            {
                Complex sum = values[0];
                int index = k + q * part;
                for (int j = 1; j < radix; j++) sum += values[j] * _roots[j * index * rootStep % Size];
                output[index] = sum;
            }
        }
    }
}
