namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class Radix2Fft
    {
        private readonly int _n;
        private readonly int[] _bitReverse;
        private readonly float[] _twiddleRe;
        private readonly float[] _twiddleIm;

        public int Size => _n;

        public Radix2Fft(int size)
        {
            if (size <= 1 || (size & (size - 1)) != 0)
                throw new ArgumentException("FFT size must be a power of two.", nameof(size));

            _n = size;
            _bitReverse = BuildBitReverse(size);
            _twiddleRe = new float[size / 2];
            _twiddleIm = new float[size / 2];
            for (int i = 0; i < _twiddleRe.Length; i++)
            {
                double angle = -2.0 * Math.PI * i / size;
                _twiddleRe[i] = (float)Math.Cos(angle);
                _twiddleIm[i] = (float)Math.Sin(angle);
            }
        }

        public void Forward(Span<float> re, Span<float> im)
        {
            if (re.Length != _n || im.Length != _n)
                throw new ArgumentException($"FFT buffers must have length {_n}.");

            BitReverse(re, im);
            for (int len = 2; len <= _n; len <<= 1)
            {
                int half = len >> 1;
                int step = _n / len;
                for (int start = 0; start < _n; start += len)
                {
                    for (int j = 0; j < half; j++)
                    {
                        int tw = j * step;
                        float wr = _twiddleRe[tw];
                        float wi = _twiddleIm[tw];
                        int even = start + j;
                        int odd = even + half;

                        float tr = re[odd] * wr - im[odd] * wi;
                        float ti = re[odd] * wi + im[odd] * wr;
                        float er = re[even];
                        float ei = im[even];

                        re[even] = er + tr;
                        im[even] = ei + ti;
                        re[odd] = er - tr;
                        im[odd] = ei - ti;
                    }
                }
            }
        }

        private void BitReverse(Span<float> re, Span<float> im)
        {
            for (int i = 0; i < _n; i++)
            {
                int j = _bitReverse[i];
                if (j <= i) continue;
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        private static int[] BuildBitReverse(int n)
        {
            int bits = 0;
            for (int v = n; v > 1; v >>= 1) bits++;

            var map = new int[n];
            for (int i = 0; i < n; i++)
            {
                int x = i;
                int r = 0;
                for (int b = 0; b < bits; b++)
                {
                    r = (r << 1) | (x & 1);
                    x >>= 1;
                }
                map[i] = r;
            }
            return map;
        }
    }
}
