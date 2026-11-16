// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures.Utils
{
    internal class FeatureFunctions
    {
        public static int RoundUpToNearestPowerOfTwo(int n)
        {
            // copied from kaldi/src/base/kaldi-math.cc
            if (n <= 0)
            {
                throw new ArgumentException("n must be greater than 0.");
            }
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return n + 1;
        }

        /// <summary>
        /// In-place power spectrum on a span: input is the complex layout
        /// [real0, realN/2, real1, im1, real2, im2, ...]; output writes per-bin
        /// energies into the first <c>n/2 + 1</c> positions.
        /// </summary>
        public static void ComputePowerSpectrum(Span<float> complexFft)
        {
            int dim = complexFft.Length;
            int halfDim = dim / 2;
            float firstEnergy = complexFft[0] * complexFft[0];
            float lastEnergy = complexFft[1] * complexFft[1];

            for (int i = 1; i < halfDim; i++)
            {
                float real = complexFft[i * 2];
                float im = complexFft[i * 2 + 1];
                complexFft[i] = real * real + im * im;
            }
            complexFft[0] = firstEnergy;
            complexFft[halfDim] = lastEnergy;
        }

        public static void RemoveDcOffset(Span<float> d, int n)
        {
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += d[i];
            }

            float mean = sum / n;

            for (int i = 0; i < n; i++)
            {
                d[i] -= mean;
            }
        }

        public static float InnerProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n)
        {
            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        public static void Preemphasize(Span<float> d, int n, float preemphCoeff)
        {
            if (preemphCoeff == 0.0f)
            {
                return;
            }

            if (preemphCoeff < 0.0f || preemphCoeff > 1.0f)
            {
                throw new ArgumentException("Preemphasis coefficient must be between 0 and 1.");
            }

            for (int i = n - 1; i > 0; i--)
            {
                d[i] -= preemphCoeff * d[i - 1];
            }
            d[0] -= preemphCoeff * d[0];
        }
    }
}
