// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

using VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures.Utils;

namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures
{

    public class Rfft
    {
        private readonly RfftImpl _impl;
        private readonly double[] _scratch;

        public Rfft(int n)
        {
            _impl = new RfftImpl(n);
            _scratch = new double[n];
        }

        public int Size => _scratch.Length;

        /// <summary>
        /// In-place real FFT on a float span. The underlying FFTSG routine works in double
        /// precision; a reusable double[] scratch avoids per-call allocations and turns the
        /// legacy 4-trip float/double/list round-trip into a single up-cast and a single
        /// down-cast.
        /// </summary>
        public void Compute(Span<float> inOut)
        {
            int n = inOut.Length;
            if (n != _scratch.Length)
                throw new ArgumentException($"FFT input size {n} does not match planned size {_scratch.Length}.", nameof(inOut));
            for (int i = 0; i < n; i++) _scratch[i] = inOut[i];
            _impl.Compute(_scratch);
            for (int i = 0; i < n; i++) inOut[i] = (float)_scratch[i];
        }
    }

    public class RfftImpl
    {
        private int n;
        private int[] ip;
        private double[] w;

        public RfftImpl(int n)
        {
            this.n = n;
            // Check if n is a power of 2.
            if ((n & (n - 1)) != 0)
                throw new Exception("n must be a power of 2.");

            int ipSize = 2 + (int)Math.Sqrt(n / 2);
            ip = new int[ipSize];
            w = new double[n / 2];
        }

        /// <summary>
        /// Forward FFT sign is 1.
        /// </summary>
        public unsafe void Compute(double[] inOut)
        {
            if (inOut.Length != n)
                throw new ArgumentException($"FFT input size {inOut.Length} does not match planned size {n}.", nameof(inOut));
            // 1 means forward fft
            fixed (double* ptrInOut = inOut)
            fixed (int* ptrIp = ip)
            fixed (double* ptrW = w)
            {
                Fftsg.rdft(n, 1, ptrInOut, ptrIp, ptrW);
            }
        }
    }
}
