using System.Numerics.Tensors;

namespace VoiceBox.AEC.DFSMN.Internal
{
    /// <summary>
    /// Direct real DFT / inverse real DFT for a fixed even length, using precomputed
    /// cos/sin tables. The model's mask STFT uses <c>n_fft = 640</c> (not a power of two),
    /// so a radix-2 FFT does not apply; the transforms are evaluated directly.
    ///
    /// Both directions use contiguous-span <see cref="TensorPrimitives"/> operations. The
    /// forward bins are dot products of the frame with each cos/sin row, and the inverse is a
    /// running multiply-add of each row scaled by its spectral coefficient.
    /// </summary>
    internal sealed class RealDft
    {
        private readonly int _n;
        private readonly int _bins;
        private readonly float[] _cos;
        private readonly float[] _sin;
        private readonly float[] _acc;

        public int FftSize => _n;
        public int BinCount => _bins;

        public RealDft(int nFft)
        {
            if (nFft <= 0 || (nFft & 1) != 0)
                throw new ArgumentException($"nFft must be a positive even number, got {nFft}.", nameof(nFft));

            _n = nFft;
            _bins = nFft / 2 + 1;
            _cos = new float[_bins * _n];
            _sin = new float[_bins * _n];
            _acc = new float[_n];

            double scale = 2.0 * Math.PI / _n;
            for (int k = 0; k < _bins; k++)
            {
                int baseIndex = k * _n;
                for (int i = 0; i < _n; i++)
                {
                    double angle = scale * k * i;
                    _cos[baseIndex + i] = (float)Math.Cos(angle);
                    _sin[baseIndex + i] = (float)Math.Sin(angle);
                }
            }
        }

        public void Forward(ReadOnlySpan<float> frame, Span<float> re, Span<float> im)
        {
            if (frame.Length != _n)
                throw new ArgumentException($"frame length must be {_n}.", nameof(frame));
            if (re.Length != _bins || im.Length != _bins)
                throw new ArgumentException($"re/im length must be {_bins}.");

            ReadOnlySpan<float> cos = _cos;
            ReadOnlySpan<float> sin = _sin;
            for (int k = 0; k < _bins; k++)
            {
                int baseIndex = k * _n;
                re[k] = TensorPrimitives.Dot<float>(frame, cos.Slice(baseIndex, _n));
                im[k] = -TensorPrimitives.Dot<float>(frame, sin.Slice(baseIndex, _n));
            }
        }

        public void Inverse(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> frame)
        {
            if (re.Length != _bins || im.Length != _bins)
                throw new ArgumentException($"re/im length must be {_bins}.");
            if (frame.Length != _n)
                throw new ArgumentException($"frame length must be {_n}.", nameof(frame));

            int nyquist = _n / 2;
            Span<float> acc = _acc;
            ReadOnlySpan<float> cos = _cos;
            ReadOnlySpan<float> sin = _sin;

            // k = 0: real, weight 1; cos row 0 is all ones, so this is a constant offset.
            acc.Fill(re[0]);

            // k = 1 .. nyquist-1: Hermitian pair, weight 2.
            for (int k = 1; k < nyquist; k++)
            {
                int baseIndex = k * _n;
                TensorPrimitives.MultiplyAdd<float>(cos.Slice(baseIndex, _n), 2f * re[k], acc, acc);
                TensorPrimitives.MultiplyAdd<float>(sin.Slice(baseIndex, _n), -2f * im[k], acc, acc);
            }

            // k = nyquist: real, weight 1.
            TensorPrimitives.MultiplyAdd<float>(cos.Slice(nyquist * _n, _n), re[nyquist], acc, acc);

            TensorPrimitives.Multiply<float>(acc, 1f / _n, frame);
        }
    }
}
