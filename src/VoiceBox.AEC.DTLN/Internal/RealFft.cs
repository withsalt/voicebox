namespace VoiceBox.AEC.DTLN.Internal
{
    internal sealed class RealFft
    {
        private readonly Radix2Fft _fft;
        private readonly float[] _re;
        private readonly float[] _im;

        public int FftSize { get; }
        public int BinCount => FftSize / 2 + 1;

        public RealFft(int fftSize)
        {
            FftSize = fftSize;
            _fft = new Radix2Fft(fftSize);
            _re = new float[fftSize];
            _im = new float[fftSize];
        }

        public void Forward(ReadOnlySpan<float> input, Span<float> re, Span<float> im)
        {
            if (input.Length != FftSize)
                throw new ArgumentException($"input length must be {FftSize}.", nameof(input));
            if (re.Length != BinCount || im.Length != BinCount)
                throw new ArgumentException($"re/im length must be {BinCount}.");

            input.CopyTo(_re);
            Array.Clear(_im);
            _fft.Forward(_re, _im);
            _re.AsSpan(0, BinCount).CopyTo(re);
            _im.AsSpan(0, BinCount).CopyTo(im);
        }

        public void ForwardMagnitude(ReadOnlySpan<float> input, Span<float> re, Span<float> im, Span<float> magnitude)
        {
            if (input.Length != FftSize)
                throw new ArgumentException($"input length must be {FftSize}.", nameof(input));
            if (re.Length != BinCount || im.Length != BinCount || magnitude.Length != BinCount)
                throw new ArgumentException($"re/im/magnitude length must be {BinCount}.");

            input.CopyTo(_re);
            Array.Clear(_im);
            _fft.Forward(_re, _im);
            for (int i = 0; i < BinCount; i++)
            {
                float real = _re[i];
                float imaginary = _im[i];
                re[i] = real;
                im[i] = imaginary;
                magnitude[i] = MathF.Sqrt(real * real + imaginary * imaginary);
            }
        }

        public void Inverse(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> output)
        {
            if (re.Length != BinCount || im.Length != BinCount)
                throw new ArgumentException($"re/im length must be {BinCount}.");
            if (output.Length != FftSize)
                throw new ArgumentException($"output length must be {FftSize}.", nameof(output));

            _re[0] = re[0];
            _im[0] = 0;
            int nyquist = FftSize / 2;
            for (int k = 1; k < nyquist; k++)
            {
                _re[k] = re[k];
                _im[k] = im[k];
                _re[FftSize - k] = re[k];
                _im[FftSize - k] = -im[k];
            }
            _re[nyquist] = re[nyquist];
            _im[nyquist] = 0;

            _fft.Inverse(_re, _im);
            _re.AsSpan().CopyTo(output);
        }
    }
}
