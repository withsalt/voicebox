namespace VoiceBox.DNR.DPDFNet.Internal
{
    /// <summary>
    /// 实数信号的离散傅里叶变换 (rfft / irfft)，匹配 numpy <c>np.fft.rfft</c> / <c>np.fft.irfft</c> 的语义。
    ///
    /// DPDFNet 16 kHz 使用 n_fft=320 (= 2^6 × 5)，并非 2 的幂，无法走基-2 FFT；
    /// 这里改用预计算的余弦/正弦旋转因子表做直接 DFT。每帧约 1e5 次乘加，
    /// 相对 ONNX 推理可忽略 (100 帧/秒)，且与 numpy 双精度 FFT 数值对齐 (累加用 double)。
    ///
    /// 频点数 K = n_fft/2 + 1 (含直流与 Nyquist)。前向输出 K 个复数 (re/im)，
    /// 逆变换从 K 个复数还原 n_fft 个实数样本。线程不安全 (无内部状态，但调用方需避免并发复用 scratch)。
    /// </summary>
    internal sealed class RealDft
    {
        private readonly int _n;
        private readonly int _bins;
        // cos/sin 表按 [k * n + i] 行主序排布；cos/sin(2π k i / N)。
        private readonly float[] _cos;
        private readonly float[] _sin;

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
        /// 前向 rfft。<paramref name="frame"/> 长度须为 <see cref="FftSize"/>；
        /// 输出写入 <paramref name="re"/> / <paramref name="im"/> (各长度 <see cref="BinCount"/>)。
        /// </summary>
        public void Forward(ReadOnlySpan<float> frame, Span<float> re, Span<float> im)
        {
            if (frame.Length != _n)
                throw new ArgumentException($"frame length must be {_n}, got {frame.Length}.", nameof(frame));
            if (re.Length != _bins || im.Length != _bins)
                throw new ArgumentException($"re/im length must be {_bins}.");

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
        /// 逆 irfft，从半谱 (<paramref name="re"/> / <paramref name="im"/>, 各长度 <see cref="BinCount"/>)
        /// 还原 <see cref="FftSize"/> 个实数样本写入 <paramref name="frame"/>。
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
    }
}
