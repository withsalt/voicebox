namespace VoiceBox.DNR.GRENet.Internal;

internal sealed class StreamingStft
{
    private readonly int _windowSize;
    private readonly int _hopSize;
    private readonly float[] _window;
    private readonly RealDft _dft;
    private readonly float[] _ring;
    private readonly float[] _windowed;

    public StreamingStft(int windowSize, int hopSize, RealDft dft)
    {
        if (windowSize <= 0 || hopSize <= 0 || hopSize > windowSize)
        {
            throw new ArgumentOutOfRangeException(nameof(hopSize));
        }

        ArgumentNullException.ThrowIfNull(dft);
        if (dft.BinCount != windowSize / 2 + 1)
        {
            throw new ArgumentException("DFT size does not match STFT size.", nameof(dft));
        }

        _windowSize = windowSize;
        _hopSize = hopSize;
        _window = new float[windowSize];
        for (int i = 0; i < windowSize; i++)
        {
            _window[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / windowSize);
        }

        _dft = dft;
        _ring = new float[windowSize];
        _windowed = new float[windowSize];
    }

    public int BinCount => _dft.BinCount;

    public void Push(ReadOnlySpan<float> hop, Span<float> real, Span<float> imaginary)
    {
        if (hop.Length != _hopSize)
        {
            throw new ArgumentException($"Hop must contain {_hopSize} samples.", nameof(hop));
        }

        Array.Copy(_ring, _hopSize, _ring, 0, _windowSize - _hopSize);
        hop.CopyTo(_ring.AsSpan(_windowSize - _hopSize));
        for (int i = 0; i < _windowSize; i++)
        {
            _windowed[i] = _ring[i] * _window[i];
        }

        _dft.Forward(_windowed, real, imaginary);
    }

    public void Reset() => Array.Clear(_ring);
}
