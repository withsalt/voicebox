using System;
using System.Numerics.Tensors;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Mixed-radix FFT for real inputs of arbitrary composite length, returning the
    /// power spectrum over the first <c>N/2+1</c> bins.
    /// </summary>
    /// <remarks>
    /// Confucius4 always uses <c>n_fft = 400 = 25 × 16</c>. A direct DFT costs 160,800
    /// multiply-adds per frame even though the input is real. This implementation
    /// performs 25 radix-2 16-point FFTs, then 25-point recombination DFTs. It is the
    /// same unnormalised forward transform as the previous implementation, but reduces
    /// the hot feature-extraction work to roughly 5,000 complex multiply-adds per frame.
    /// The generic direct-DFT path remains for non-400 configurations used by callers.
    /// </remarks>
    internal sealed class RealFft
    {
        private readonly int _size;
        private readonly int _bins;
        private readonly float[] _cos;
        private readonly float[] _sin;
        private readonly float[]? _mixedCos;
        private readonly float[]? _mixedSin;
        private readonly float[]? _stageReal;
        private readonly float[]? _stageImag;

        public int Size => _size;
        public int Bins => _bins;

        public RealFft(int size)
        {
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
            _size = size;
            _bins = size / 2 + 1;

            if (size == 400)
            {
                // [k1, k0, n0], where output bin k = k1 + 16*k0 and
                // input sample n = n0 + 25*n1. Store the combined twiddle so
                // ForwardPower does no trigonometry or intermediate complex multiplies.
                _mixedCos = new float[16 * 25 * 25];
                _mixedSin = new float[_mixedCos.Length];
                for (int k1 = 0; k1 < 16; k1++)
                {
                    for (int k0 = 0; k0 < 25; k0++)
                    {
                        for (int n0 = 0; n0 < 25; n0++)
                        {
                            int index = ((k1 * 25) + k0) * 25 + n0;
                            double angle = -2.0 * Math.PI * n0 * (k1 + 16.0 * k0) / size;
                            _mixedCos[index] = (float)Math.Cos(angle);
                            _mixedSin[index] = (float)Math.Sin(angle);
                        }
                    }
                }
                _stageReal = new float[size];
                _stageImag = new float[size];
                _cos = Array.Empty<float>();
                _sin = Array.Empty<float>();
                return;
            }

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

            if (_size == 400)
            {
                ForwardPower400(real, power);
                return;
            }

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

        private void ForwardPower400(ReadOnlySpan<float> real, Span<float> power)
        {
            Span<float> stageReal = _stageReal!;
            Span<float> stageImag = _stageImag!;

            // First stage: 25 independent 16-point DIT FFTs over samples n0 + 25*n1.
            for (int n0 = 0; n0 < 25; n0++)
            {
                Span<float> re = stageReal.Slice(n0 * 16, 16);
                Span<float> im = stageImag.Slice(n0 * 16, 16);
                for (int n1 = 0; n1 < 16; n1++)
                {
                    int reversed = Reverse4(n1);
                    re[reversed] = real[n0 + 25 * n1];
                    im[reversed] = 0f;
                }
                Fft16(re, im);
            }

            ReadOnlySpan<float> cos = _mixedCos!;
            ReadOnlySpan<float> sin = _mixedSin!;
            for (int k1 = 0; k1 < 16; k1++)
            {
                for (int k0 = 0; k0 < 25; k0++)
                {
                    int output = k1 + 16 * k0;
                    if (output >= _bins) break;

                    int twiddleOffset = ((k1 * 25) + k0) * 25;
                    float sumRe = 0f;
                    float sumIm = 0f;
                    for (int n0 = 0; n0 < 25; n0++)
                    {
                        int stageOffset = n0 * 16 + k1;
                        float re = stageReal[stageOffset];
                        float im = stageImag[stageOffset];
                        float twRe = cos[twiddleOffset + n0];
                        float twIm = sin[twiddleOffset + n0];
                        sumRe += re * twRe - im * twIm;
                        sumIm += re * twIm + im * twRe;
                    }
                    power[output] = sumRe * sumRe + sumIm * sumIm;
                }
            }
        }

        private static void Fft16(Span<float> re, Span<float> im)
        {
            for (int length = 2; length <= 16; length <<= 1)
            {
                int half = length >> 1;
                int twiddleStep = 16 / length;
                for (int start = 0; start < 16; start += length)
                {
                    for (int j = 0; j < half; j++)
                    {
                        int even = start + j;
                        int odd = even + half;
                        int twiddle = j * twiddleStep;
                        GetFft16Twiddle(twiddle, out float twRe, out float twIm);
                        float oddRe = re[odd] * twRe - im[odd] * twIm;
                        float oddIm = re[odd] * twIm + im[odd] * twRe;
                        float evenRe = re[even];
                        float evenIm = im[even];
                        re[even] = evenRe + oddRe;
                        im[even] = evenIm + oddIm;
                        re[odd] = evenRe - oddRe;
                        im[odd] = evenIm - oddIm;
                    }
                }
            }
        }

        private static int Reverse4(int value)
            => ((value & 1) << 3) | ((value & 2) << 1) | ((value & 4) >> 1) | ((value & 8) >> 3);

        private static void GetFft16Twiddle(int index, out float re, out float im)
        {
            // The radix-2 stages only request 0, 1, 2, 3, 4, 6, or 7.
            // Keeping these constants avoids sin/cos calls in the per-frame hot path.
            const float S = 0.7071067811865476f;
            switch (index)
            {
                case 0: re = 1f; im = 0f; return;
                case 1: re = 0.9238795325112867f; im = -0.3826834323650898f; return;
                case 2: re = S; im = -S; return;
                case 3: re = 0.3826834323650898f; im = -0.9238795325112867f; return;
                case 4: re = 0f; im = -1f; return;
                case 5: re = -0.3826834323650898f; im = -0.9238795325112867f; return;
                case 6: re = -S; im = -S; return;
                default: re = -0.9238795325112867f; im = -0.3826834323650898f; return;
            }
        }
    }
}
