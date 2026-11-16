using System;
using System.Numerics.Tensors;

namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// Direct DFT for real inputs of arbitrary length, returning the power spectrum
    /// over the first <c>N/2+1</c> bins. Uses precomputed twiddle tables.
    /// </summary>
    /// <remarks>
    /// Whisper feature extraction uses <c>n_fft = 400</c> (not a power of two). Rather
    /// than introduce Bluestein machinery for one fixed size, we evaluate the DFT
    /// directly: cost is <c>O(N · bins)</c> per frame which is ~80 K mults for N=400,
    /// bins=201 — fast enough for offline encoder feeding and trivially correct.
    /// The two inner dot products are evaluated with <see cref="TensorPrimitives.Dot{T}"/>.
    /// </remarks>
    internal sealed class RealFft
    {
        private readonly int _size;
        private readonly int _bins;
        private readonly float[] _cos;
        private readonly float[] _sin;

        public int Size => _size;
        public int Bins => _bins;

        public RealFft(int size)
        {
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
            _size = size;
            _bins = size / 2 + 1;

            _cos = new float[_bins * _size];
            _sin = new float[_bins * _size];
            double scale = -2.0 * Math.PI / _size;
            for (int k = 0; k < _bins; k++)
            {
                int rowOffset = k * _size;
                for (int n = 0; n < _size; n++)
                {
                    double angle = scale * k * n;
                    _cos[rowOffset + n] = (float)Math.Cos(angle);
                    _sin[rowOffset + n] = (float)Math.Sin(angle);
                }
            }
        }

        /// <summary>
        /// Forward DFT of a real input of length <see cref="Size"/>. Writes the
        /// magnitude-squared spectrum to <paramref name="power"/>: <c>power[k] = |X[k]|²</c>.
        /// Only the first <see cref="Bins"/> bins are produced.
        /// </summary>
        public void ForwardPower(ReadOnlySpan<float> real, Span<float> power)
        {
            if (real.Length != _size)
                throw new ArgumentException($"Input must be exactly {_size} samples.", nameof(real));
            if (power.Length < _bins)
                throw new ArgumentException($"power buffer must be ≥ {_bins} floats.", nameof(power));

            ReadOnlySpan<float> cosAll = _cos;
            ReadOnlySpan<float> sinAll = _sin;
            int n = _size;
            for (int k = 0; k < _bins; k++)
            {
                int row = k * n;
                float re = TensorPrimitives.Dot(real, cosAll.Slice(row, n));
                float im = TensorPrimitives.Dot(real, sinAll.Slice(row, n));
                power[k] = re * re + im * im;
            }
        }
    }
}
