using System.Numerics;

namespace VoiceBox.DNR.GRENet.Internal;

/// <summary>
/// Small real DFT used by the 400-point GRE-Net analysis window.
/// The precomputed twiddle table keeps the realtime path allocation-free and
/// avoids taking a dependency on a general purpose FFT package.
/// </summary>
internal sealed class RealDft
{
    private readonly int _size;
    private readonly int _binCount;
    private readonly float[] _cos;
    private readonly float[] _sin;

    public RealDft(int size)
    {
        if (size <= 0 || (size & 1) != 0)
        {
            throw new ArgumentException("FFT size must be a positive even number.", nameof(size));
        }

        _size = size;
        _binCount = size / 2 + 1;
        _cos = new float[_binCount * size];
        _sin = new float[_binCount * size];

        double step = 2.0 * Math.PI / size;
        for (int k = 0; k < _binCount; k++)
        {
            int offset = k * size;
            for (int i = 0; i < size; i++)
            {
                double angle = step * k * i;
                _cos[offset + i] = (float)Math.Cos(angle);
                _sin[offset + i] = (float)Math.Sin(angle);
            }
        }
    }

    public int BinCount => _binCount;

    public void Forward(ReadOnlySpan<float> frame, Span<float> real, Span<float> imaginary)
    {
        if (frame.Length != _size || real.Length != _binCount || imaginary.Length != _binCount)
        {
            throw new ArgumentException("Unexpected DFT buffer length.");
        }

        int vectorSize = Vector<float>.Count;
        for (int k = 0; k < _binCount; k++)
        {
            int offset = k * _size;
            Vector<float> vectorReal = Vector<float>.Zero;
            Vector<float> vectorImaginary = Vector<float>.Zero;
            int i = 0;
            for (; i <= _size - vectorSize; i += vectorSize)
            {
                Vector<float> samples = new(frame.Slice(i, vectorSize));
                Vector<float> cosines = new(_cos, offset + i);
                Vector<float> sines = new(_sin, offset + i);
                vectorReal += samples * cosines;
                vectorImaginary -= samples * sines;
            }

            float sumReal = 0f;
            float sumImaginary = 0f;
            for (int lane = 0; lane < vectorSize; lane++)
            {
                sumReal += vectorReal[lane];
                sumImaginary += vectorImaginary[lane];
            }

            for (; i < _size; i++)
            {
                float sample = frame[i];
                sumReal += sample * _cos[offset + i];
                sumImaginary -= sample * _sin[offset + i];
            }

            real[k] = sumReal;
            imaginary[k] = sumImaginary;
        }
    }

    public void Inverse(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, Span<float> frame)
    {
        if (real.Length != _binCount || imaginary.Length != _binCount || frame.Length != _size)
        {
            throw new ArgumentException("Unexpected inverse DFT buffer length.");
        }

        int nyquist = _size / 2;
        int vectorSize = Vector<float>.Count;
        frame.Fill(real[0]);
        for (int k = 1; k < nyquist; k++)
        {
            int offset = k * _size;
            Vector<float> realPart = new(real[k] * 2f);
            Vector<float> imaginaryPart = new(imaginary[k] * -2f);
            int i = 0;
            for (; i <= _size - vectorSize; i += vectorSize)
            {
                Vector<float> values = new(frame.Slice(i, vectorSize));
                Vector<float> cosines = new(_cos, offset + i);
                Vector<float> sines = new(_sin, offset + i);
                (values + realPart * cosines + imaginaryPart * sines).CopyTo(frame.Slice(i, vectorSize));
            }

            for (; i < _size; i++)
            {
                frame[i] += 2f * (real[k] * _cos[offset + i] - imaginary[k] * _sin[offset + i]);
            }
        }

        int nyquistOffset = nyquist * _size;
        for (int i = 0; i < _size; i++)
        {
            frame[i] = (frame[i] + real[nyquist] * _cos[nyquistOffset + i]) / _size;
        }
    }
}
