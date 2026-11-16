namespace VoiceBox.DNR.DeepFilterNet.Internal
{
    /// <summary>
    /// 实数信号的离散傅里叶变换 (rfft / irfft)，匹配 numpy <c>np.fft.rfft</c> / <c>np.fft.irfft</c> 的语义。
    ///
    /// DeepFilterNet 当前使用 n_fft=960 (= 15 × 64)。该长度走专用混合基 FFT；
    /// 其它偶数长度保留原来的预计算旋转因子表 + double 累加直接 DFT fallback。
    /// 数学公式不变，960 路径改变的是求值算法与浮点求和顺序。
    ///
    /// 频点数 K = n_fft/2 + 1。<see cref="Forward"/> 为<b>未归一化</b> rfft；
    /// <see cref="Inverse"/> 为归一化 irfft (含 1/N，与 numpy 一致)。
    /// libDF 合成端使用<b>未归一化</b>逆变换，因此调用方 (DfState) 在 Inverse 之后再乘 N 还原。
    /// 960 路径复用内部工作缓冲；调用方需避免并发复用同一个 <see cref="RealDft"/> 实例。
    /// </summary>
    internal sealed class RealDft
    {
        private readonly int _n;
        private readonly int _bins;
        // cos/sin 表按 [k * n + i] 行主序排布；cos/sin(2π k i / N)。
        private readonly float[] _cos;
        private readonly float[] _sin;
        private readonly Fft960Plan? _fft960;
        private readonly double[] _fftRe;
        private readonly double[] _fftIm;

        public int FftSize => _n;
        public int BinCount => _bins;

        public RealDft(int nFft)
        {
            if (nFft <= 0 || (nFft & 1) != 0)
                throw new ArgumentException($"nFft must be a positive even number, got {nFft}.", nameof(nFft));

            _n = nFft;
            _bins = nFft / 2 + 1;

            if (nFft == Fft960Plan.Size)
            {
                _fft960 = new Fft960Plan();
                _fftRe = new double[_n];
                _fftIm = new double[_n];
                _cos = [];
                _sin = [];
                return;
            }

            _cos = new float[_bins * _n];
            _sin = new float[_bins * _n];
            _fftRe = [];
            _fftIm = [];

            double twoPiOverN = 2.0 * Math.PI / _n;
            for (int k = 0; k < _bins; k++)
            {
                int baseIdx = k * _n;
                for (int i = 0; i < _n; i++)
                {
                    double angle = twoPiOverN * k * i;
                    _cos[baseIdx + i] = (float)Math.Cos(angle);
                    _sin[baseIdx + i] = (float)Math.Sin(angle);
                }
            }
        }

        /// <summary>
        /// 未归一化前向 rfft。<paramref name="frame"/> 长度须为 <see cref="FftSize"/>；
        /// 输出写入 <paramref name="re"/> / <paramref name="im"/> (各长度 <see cref="BinCount"/>)。
        /// </summary>
        public void Forward(ReadOnlySpan<float> frame, Span<float> re, Span<float> im)
        {
            if (frame.Length != _n)
                throw new ArgumentException($"frame length must be {_n}, got {frame.Length}.", nameof(frame));
            if (re.Length != _bins || im.Length != _bins)
                throw new ArgumentException($"re/im length must be {_bins}.");

            if (_fft960 is not null)
            {
                for (int i = 0; i < _n; i++)
                {
                    _fftRe[i] = frame[i];
                    _fftIm[i] = 0.0;
                }

                _fft960.Transform(_fftRe, _fftIm, inverse: false);
                for (int k = 0; k < _bins; k++)
                {
                    re[k] = (float)_fftRe[k];
                    im[k] = (float)_fftIm[k];
                }
                return;
            }

            for (int k = 0; k < _bins; k++)
            {
                int baseIdx = k * _n;
                double sumRe = 0.0;
                double sumIm = 0.0;
                for (int i = 0; i < _n; i++)
                {
                    float x = frame[i];
                    sumRe += x * _cos[baseIdx + i];
                    sumIm -= x * _sin[baseIdx + i];
                }
                re[k] = (float)sumRe;
                im[k] = (float)sumIm;
            }
        }

        /// <summary>
        /// 归一化逆 irfft (含 1/N)，从半谱还原 <see cref="FftSize"/> 个实数样本。
        /// 与 numpy 一致：直流与 Nyquist 分量的虚部被忽略。
        /// </summary>
        public void Inverse(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> frame)
        {
            if (re.Length != _bins || im.Length != _bins)
                throw new ArgumentException($"re/im length must be {_bins}.");
            if (frame.Length != _n)
                throw new ArgumentException($"frame length must be {_n}, got {frame.Length}.", nameof(frame));

            int nyquist = _n / 2;
            float invN = 1.0f / _n;

            if (_fft960 is not null)
            {
                _fftRe[0] = re[0];
                _fftIm[0] = 0.0;
                for (int k = 1; k < nyquist; k++)
                {
                    _fftRe[k] = re[k];
                    _fftIm[k] = im[k];
                    int mirror = _n - k;
                    _fftRe[mirror] = re[k];
                    _fftIm[mirror] = -im[k];
                }
                _fftRe[nyquist] = re[nyquist];
                _fftIm[nyquist] = 0.0;

                _fft960.Transform(_fftRe, _fftIm, inverse: true);
                double scale = invN;
                for (int i = 0; i < _n; i++)
                    frame[i] = (float)(_fftRe[i] * scale);
                return;
            }

            for (int idx = 0; idx < _n; idx++)
            {
                // k=0 (直流)：仅实部贡献。
                double acc = re[0];
                // k=1 .. N/2-1：每个频点出现两次 (k 与 N-k 共轭)，故系数 2。
                for (int k = 1; k < nyquist; k++)
                {
                    int t = k * _n + idx; // cos/sin(2π k idx / N)
                    acc += 2.0 * (re[k] * _cos[t] - im[k] * _sin[t]);
                }
                // k=N/2 (Nyquist)：cos(π·idx) = ±1，仅实部贡献，不乘 2。
                acc += re[nyquist] * _cos[nyquist * _n + idx];
                frame[idx] = (float)(acc * invN);
            }
        }

        /// <summary>
        /// Cooley-Tukey FFT for N=960 with N1=15, N2=64:
        /// first 15 strided 64-point FFTs, then 64 direct 15-point DFTs.
        /// The transform is unnormalized in both directions.
        /// </summary>
        private sealed class Fft960Plan
        {
            public const int Size = N1 * N2;
            private const int N1 = 15;
            private const int N2 = 64;

            private readonly Radix2Fft64 _fft64 = new();
            private readonly double[] _tmpRe = new double[N2];
            private readonly double[] _tmpIm = new double[N2];
            private readonly double[] _stageRe = new double[Size];
            private readonly double[] _stageIm = new double[Size];
            private readonly double[] _twRe = new double[Size];
            private readonly double[] _twIm = new double[Size];
            private readonly double[] _dft15Re = new double[N1 * N1];
            private readonly double[] _dft15Im = new double[N1 * N1];

            public Fft960Plan()
            {
                double twoPiOverN = 2.0 * Math.PI / Size;
                for (int k2 = 0; k2 < N2; k2++)
                {
                    for (int n1 = 0; n1 < N1; n1++)
                    {
                        double angle = twoPiOverN * n1 * k2;
                        int idx = k2 * N1 + n1;
                        _twRe[idx] = Math.Cos(angle);
                        _twIm[idx] = Math.Sin(angle);
                    }
                }

                double twoPiOverN1 = 2.0 * Math.PI / N1;
                for (int k1 = 0; k1 < N1; k1++)
                {
                    for (int n1 = 0; n1 < N1; n1++)
                    {
                        double angle = twoPiOverN1 * n1 * k1;
                        int idx = k1 * N1 + n1;
                        _dft15Re[idx] = Math.Cos(angle);
                        _dft15Im[idx] = Math.Sin(angle);
                    }
                }
            }

            public void Transform(double[] re, double[] im, bool inverse)
            {
                for (int n1 = 0; n1 < N1; n1++)
                {
                    for (int n2 = 0; n2 < N2; n2++)
                    {
                        int inputIdx = n1 + N1 * n2;
                        _tmpRe[n2] = re[inputIdx];
                        _tmpIm[n2] = im[inputIdx];
                    }

                    _fft64.Transform(_tmpRe, _tmpIm, inverse);

                    for (int k2 = 0; k2 < N2; k2++)
                    {
                        int idx = k2 * N1 + n1;
                        double wr = _twRe[idx];
                        double wi = inverse ? _twIm[idx] : -_twIm[idx];
                        double ar = _tmpRe[k2];
                        double ai = _tmpIm[k2];
                        _stageRe[idx] = ar * wr - ai * wi;
                        _stageIm[idx] = ar * wi + ai * wr;
                    }
                }

                for (int k2 = 0; k2 < N2; k2++)
                {
                    int stageBase = k2 * N1;
                    for (int k1 = 0; k1 < N1; k1++)
                    {
                        int dftBase = k1 * N1;
                        double sumRe = 0.0;
                        double sumIm = 0.0;
                        for (int n1 = 0; n1 < N1; n1++)
                        {
                            double wr = _dft15Re[dftBase + n1];
                            double wi = inverse ? _dft15Im[dftBase + n1] : -_dft15Im[dftBase + n1];
                            double ar = _stageRe[stageBase + n1];
                            double ai = _stageIm[stageBase + n1];
                            sumRe += ar * wr - ai * wi;
                            sumIm += ar * wi + ai * wr;
                        }

                        int outputIdx = k2 + N2 * k1;
                        re[outputIdx] = sumRe;
                        im[outputIdx] = sumIm;
                    }
                }
            }
        }

        private sealed class Radix2Fft64
        {
            private const int N = 64;
            private readonly int[] _bitReverse = new int[N];
            private readonly double[] _cos = new double[N / 2];
            private readonly double[] _sin = new double[N / 2];

            public Radix2Fft64()
            {
                for (int i = 0; i < N; i++)
                {
                    int x = i;
                    int r = 0;
                    for (int b = 0; b < 6; b++)
                    {
                        r = (r << 1) | (x & 1);
                        x >>= 1;
                    }
                    _bitReverse[i] = r;
                }

                for (int i = 0; i < N / 2; i++)
                {
                    double angle = 2.0 * Math.PI * i / N;
                    _cos[i] = Math.Cos(angle);
                    _sin[i] = Math.Sin(angle);
                }
            }

            public void Transform(double[] re, double[] im, bool inverse)
            {
                for (int i = 0; i < N; i++)
                {
                    int j = _bitReverse[i];
                    if (i >= j) continue;
                    (re[i], re[j]) = (re[j], re[i]);
                    (im[i], im[j]) = (im[j], im[i]);
                }

                for (int len = 2; len <= N; len <<= 1)
                {
                    int half = len >> 1;
                    int step = N / len;
                    for (int block = 0; block < N; block += len)
                    {
                        for (int j = 0; j < half; j++)
                        {
                            int tw = j * step;
                            double wr = _cos[tw];
                            double wi = inverse ? _sin[tw] : -_sin[tw];
                            int even = block + j;
                            int odd = even + half;

                            double vr = re[odd] * wr - im[odd] * wi;
                            double vi = re[odd] * wi + im[odd] * wr;
                            double ur = re[even];
                            double ui = im[even];

                            re[even] = ur + vr;
                            im[even] = ui + vi;
                            re[odd] = ur - vr;
                            im[odd] = ui - vi;
                        }
                    }
                }
            }
        }
    }
}
