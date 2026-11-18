namespace VoiceBox.DNR.GRENet.Internal;

internal sealed class StreamingIstft
{
    private readonly int _windowSize;
    private readonly int _hopSize;
    private readonly float[] _window;
    private readonly float[] _windowSquared;
    private readonly RealDft _dft;
    private readonly float[] _ola;
    private readonly float[] _olaNorm;
    private readonly float[] _frame;

    public StreamingIstft(int windowSize, int hopSize, RealDft dft)
    {
        if (dft.BinCount != windowSize / 2 + 1)
        {
            throw new ArgumentException("DFT size does not match ISTFT size.", nameof(dft));
        }

        _windowSize = windowSize;
        _hopSize = hopSize;
        _dft = dft;
        _window = new float[windowSize];
        _windowSquared = new float[windowSize];
        for (int i = 0; i < windowSize; i++)
        {
            float value = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / windowSize);
            _window[i] = value;
            _windowSquared[i] = value * value;
        }

        _ola = new float[windowSize];
        _olaNorm = new float[windowSize];
        _frame = new float[windowSize];
    }

    public void Push(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, Span<float> output)
    {
        if (real.Length != _dft.BinCount || imaginary.Length != _dft.BinCount || output.Length != _hopSize)
        {
            throw new ArgumentException("Unexpected ISTFT buffer length.");
        }

        _dft.Inverse(real, imaginary, _frame);
        Array.Copy(_ola, _hopSize, _ola, 0, _windowSize - _hopSize);
        Array.Copy(_olaNorm, _hopSize, _olaNorm, 0, _windowSize - _hopSize);
        Array.Clear(_ola, _windowSize - _hopSize, _hopSize);
        Array.Clear(_olaNorm, _windowSize - _hopSize, _hopSize);

        for (int i = 0; i < _windowSize; i++)
        {
            _ola[i] += _frame[i] * _window[i];
            _olaNorm[i] += _windowSquared[i];
        }

        for (int i = 0; i < _hopSize; i++)
        {
            output[i] = _ola[i] / MathF.Max(_olaNorm[i], 1e-4f);
        }
    }

    public void Reset()
    {
        Array.Clear(_ola);
        Array.Clear(_olaNorm);
    }
}
