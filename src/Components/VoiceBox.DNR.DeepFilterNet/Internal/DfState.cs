namespace VoiceBox.DNR.DeepFilterNet.Internal
{
    /// <summary>
    /// DeepFilterNet 的 DSP 状态机，端口自 libDF 的 <c>DFState</c>（deep_filter crate）。
    /// 负责：STFT 分析 / ISTFT 合成（OLA）、ERB 能量特征、复数谱单位归一化特征、ERB 增益施加。
    /// 全部公式与 libDF 逐元素一致，保证与原始实现数值对齐。
    ///
    /// 频谱以 (re[], im[]) 两个并行数组表示（长度均为 <see cref="FreqSize"/>）。
    /// 归一化态（mean/unit）跨帧累积，由 <see cref="Reset"/> / <see cref="InitNormStates"/> 复位。
    /// 非线程安全：由上层服务持锁串行调用。
    /// </summary>
    internal sealed class DfState
    {
        private readonly int _frameSize;   // hop_size
        private readonly int _windowSize;  // fft_size
        private readonly int _freqSize;    // fft_size/2 + 1
        private readonly float _wnorm;     // 2*hop/fft²
        private readonly float[] _window;  // vorbis 窗 [fft]
        private readonly int[] _erb;       // ERB 带宽表 [nbErb]
        private readonly RealDft _dft;

        // OLA 记忆缓冲，长度 fft-hop。
        private readonly float[] _analysisMem;
        private readonly float[] _synthesisMem;

        // 运行式归一化态。
        private readonly float[] _meanNormState; // [nbErb]
        private readonly float[] _unitNormState; // [nbDf]
        private readonly int _nbErb;
        private readonly int _nbDf;

        // 复用工作缓冲。
        private readonly float[] _fftIn;   // [fft]
        private readonly float[] _ifftOut; // [fft]

        public int FreqSize => _freqSize;
        public int FrameSize => _frameSize;
        public int NbErb => _nbErb;
        public int NbDf => _nbDf;
        public IReadOnlyList<int> Erb => _erb;

        public DfState(int sr, int fftSize, int hopSize, int nbErb, int nbDf, int minNbErbFreqs)
        {
            if (sr <= 0)
                throw new ArgumentOutOfRangeException(nameof(sr), sr, "sr must be > 0.");
            if (fftSize <= 0 || (fftSize & 1) != 0)
                throw new ArgumentOutOfRangeException(nameof(fftSize), fftSize, "fftSize must be a positive even number.");
            if (hopSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(hopSize), hopSize, "hopSize must be > 0.");
            if (hopSize * 2 > fftSize)
                throw new ArgumentException($"hop_size*2 ({hopSize * 2}) must be <= fft_size ({fftSize}).");
            if (nbErb < 2)
                throw new ArgumentOutOfRangeException(nameof(nbErb), nbErb, "nbErb must be >= 2.");
            if (nbDf < 2 || nbDf > fftSize / 2 + 1)
                throw new ArgumentOutOfRangeException(nameof(nbDf), nbDf, $"nbDf must be in [2, {fftSize / 2 + 1}].");
            if (minNbErbFreqs <= 0)
                throw new ArgumentOutOfRangeException(nameof(minNbErbFreqs), minNbErbFreqs, "minNbErbFreqs must be > 0.");

            _frameSize = hopSize;
            _windowSize = fftSize;
            _freqSize = fftSize / 2 + 1;
            _nbErb = nbErb;
            _nbDf = nbDf;
            _dft = new RealDft(fftSize);
            _window = VorbisWindow.Create(fftSize);
            _wnorm = 1f / (fftSize * (float)fftSize / (2f * hopSize));
            _erb = ErbFilterBank.Compute(sr, fftSize, nbErb, minNbErbFreqs);

            // ERB 带宽表必须恰好覆盖全部频点 (compute_band_corr / apply_mask 依赖此不变量)。
            // 对受支持的配置必然成立；若未来配置不符则尽早抛出，避免静默产生错误特征/掩码。
            int erbSum = 0;
            for (int i = 0; i < _erb.Length; i++) erbSum += _erb[i];
            if (erbSum != _freqSize)
                throw new InvalidOperationException(
                    $"ERB band widths sum to {erbSum}, expected {_freqSize} " +
                    $"(sr={sr}, fft={fftSize}, nb_erb={nbErb}, min_nb_erb_freqs={minNbErbFreqs}).");

            int memLen = fftSize - hopSize;
            _analysisMem = new float[memLen];
            _synthesisMem = new float[memLen];

            _meanNormState = new float[nbErb];
            _unitNormState = new float[nbDf];
            InitNormStates();

            _fftIn = new float[fftSize];
            _ifftOut = new float[fftSize];
        }

        /// <summary>初始化两个归一化态到 libDF 的线性初值。</summary>
        public void InitNormStates()
        {
            // mean_norm_state: [-60, -90] 线性。
            float minM = -60f, maxM = -90f;
            float stepM = (maxM - minM) / (_nbErb - 1);
            for (int i = 0; i < _nbErb; i++) _meanNormState[i] = minM + i * stepM;

            // unit_norm_state: [0.001, 0.0001] 线性。
            float minU = 0.001f, maxU = 0.0001f;
            float stepU = (maxU - minU) / (_nbDf - 1);
            for (int i = 0; i < _nbDf; i++) _unitNormState[i] = minU + i * stepU;
        }

        /// <summary>复位 OLA 记忆与归一化态，开始新的独立流。</summary>
        public void Reset()
        {
            Array.Clear(_analysisMem);
            Array.Clear(_synthesisMem);
            InitNormStates();
        }

        /// <summary>
        /// STFT 分析：输入一个 hop 的时域样本，输出当前帧复数谱 (re/im, 各长 <see cref="FreqSize"/>)。
        /// 与 libDF <c>frame_analysis</c> 一致：前半窗作用于上一帧记忆、后半窗作用于新输入，
        /// 前向 FFT 后乘 wnorm。
        /// </summary>
        public void Analysis(ReadOnlySpan<float> input, Span<float> re, Span<float> im)
        {
            if (input.Length != _frameSize)
                throw new ArgumentException($"input length must be {_frameSize}.", nameof(input));

            int split = _windowSize - _frameSize; // 窗的前半长度，也是 _analysisMem 的长度
            // buf[0..split] = analysisMem · window[0..split]
            for (int i = 0; i < split; i++)
                _fftIn[i] = _analysisMem[i] * _window[i];
            // buf[split..fft] = input · window[split..fft]
            for (int i = 0; i < _frameSize; i++)
                _fftIn[split + i] = input[i] * _window[split + i];

            // 更新 analysisMem 供下一帧使用（analysis_split = memLen-hop；50% 重叠时为 0，即整段写入）。
            int analysisSplit = _analysisMem.Length - _frameSize;
            if (analysisSplit > 0)
            {
                // 左移 hop（rotate_left frame_size）。
                Array.Copy(_analysisMem, _frameSize, _analysisMem, 0, _analysisMem.Length - _frameSize);
            }
            for (int i = 0; i < _frameSize; i++)
                _analysisMem[analysisSplit + i] = input[i];

            _dft.Forward(_fftIn, re, im);
            for (int k = 0; k < _freqSize; k++)
            {
                re[k] *= _wnorm;
                im[k] *= _wnorm;
            }
        }

        /// <summary>
        /// ISTFT 合成：输入一帧复数谱，OLA 输出一个 hop 的时域样本。
        /// 与 libDF <c>frame_synthesis</c> 一致：未归一化逆 FFT（= numpy irfft × fft）、加合成窗、重叠相加。
        /// </summary>
        public void Synthesis(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> output)
        {
            if (output.Length != _frameSize)
                throw new ArgumentException($"output length must be {_frameSize}.", nameof(output));

            // 未归一化逆变换 = 归一化 irfft × windowSize。
            _dft.Inverse(re, im, _ifftOut);
            for (int i = 0; i < _windowSize; i++)
                _ifftOut[i] *= _windowSize * _window[i];

            // output = x[0..hop] + synthesisMem[0..hop]
            for (int i = 0; i < _frameSize; i++)
                output[i] = _ifftOut[i] + _synthesisMem[i];

            // 更新 synthesisMem。split = memLen-hop（50% 重叠时为 0）。
            int split = _synthesisMem.Length - _frameSize;
            if (split > 0)
                Array.Copy(_synthesisMem, _frameSize, _synthesisMem, 0, split);

            // s_first[0..split] += x_second[0..split]（重叠相加）
            for (int i = 0; i < split; i++)
                _synthesisMem[i] += _ifftOut[_frameSize + i];
            // s_second[split..] = x_second[split..]（覆盖左移留下的尾部）
            for (int i = split; i < _synthesisMem.Length; i++)
                _synthesisMem[i] = _ifftOut[_frameSize + i];
        }

        /// <summary>
        /// ERB 能量特征 (长 <see cref="NbErb"/>)：每带能量均值 → 10·log10(.+1e-10) → 指数均值归一化(÷40)。
        /// </summary>
        public void FeatErb(ReadOnlySpan<float> re, ReadOnlySpan<float> im, float alpha, Span<float> output)
        {
            // compute_band_corr：每带 (re²+im²) 的均值。
            int bcsum = 0;
            for (int b = 0; b < _nbErb; b++)
            {
                int bandSize = _erb[b];
                float k = 1f / bandSize;
                float acc = 0f;
                for (int j = 0; j < bandSize; j++)
                {
                    int idx = bcsum + j;
                    acc += (re[idx] * re[idx] + im[idx] * im[idx]) * k;
                }
                output[b] = acc;
                bcsum += bandSize;
            }

            // dB
            for (int b = 0; b < _nbErb; b++)
                output[b] = MathF.Log10(output[b] + 1e-10f) * 10f;

            // band_mean_norm_erb
            for (int b = 0; b < _nbErb; b++)
            {
                _meanNormState[b] = output[b] * (1f - alpha) + _meanNormState[b] * alpha;
                output[b] = (output[b] - _meanNormState[b]) / 40f;
            }
        }

        /// <summary>
        /// 复数谱单位归一化特征：对前 <see cref="NbDf"/> 个频点做运行式单位范数归一化，
        /// 输出实/虚部到 <paramref name="outRe"/> / <paramref name="outIm"/> (各长 NbDf)。
        /// </summary>
        public void FeatCplx(ReadOnlySpan<float> re, ReadOnlySpan<float> im, float alpha, Span<float> outRe, Span<float> outIm)
        {
            for (int f = 0; f < _nbDf; f++)
            {
                float norm = MathF.Sqrt(re[f] * re[f] + im[f] * im[f]);
                _unitNormState[f] = norm * (1f - alpha) + _unitNormState[f] * alpha;
                float denom = MathF.Sqrt(_unitNormState[f]);
                outRe[f] = re[f] / denom;
                outIm[f] = im[f] / denom;
            }
        }

        /// <summary>
        /// 施加 ERB 增益：把 <paramref name="gains"/> (长 NbErb) 升维到全谱并逐点乘到 (re/im)。
        /// 端口自 libDF <c>apply_interp_band_gain</c>。
        /// </summary>
        public void ApplyMask(Span<float> re, Span<float> im, ReadOnlySpan<float> gains)
        {
            int bcsum = 0;
            for (int b = 0; b < _nbErb; b++)
            {
                float g = gains[b];
                int bandSize = _erb[b];
                for (int j = 0; j < bandSize; j++)
                {
                    int idx = bcsum + j;
                    re[idx] *= g;
                    im[idx] *= g;
                }
                bcsum += bandSize;
            }
        }

        /// <summary>
        /// 计算与 python <c>get_norm_alpha</c> 一致的归一化衰减系数 α = exp(−hop/sr/τ)，
        /// 并按其取整规则（找到使结果 &lt; 1 的最小小数位数）输出。
        /// </summary>
        public static float ComputeNormAlpha(int sr, int hopSize, float tau)
        {
            if (sr <= 0) throw new ArgumentOutOfRangeException(nameof(sr), sr, "sr must be > 0.");
            if (hopSize <= 0) throw new ArgumentOutOfRangeException(nameof(hopSize), hopSize, "hopSize must be > 0.");
            if (!(tau > 0f)) throw new ArgumentOutOfRangeException(nameof(tau), tau, "norm_tau must be > 0.");

            double dt = (double)hopSize / sr;
            double a = Math.Exp(-dt / tau); // 0 < a < 1（因 dt>0, tau>0）
            int precision = 3;
            double rounded = 1.0;
            // a < 1，故必在有限步内 rounded < 1；上限 20 位纯为防御性兜底。
            while (rounded >= 1.0 && precision <= 20)
            {
                double scale = Math.Pow(10, precision);
                rounded = Math.Round(a * scale, MidpointRounding.AwayFromZero) / scale;
                precision += 1;
            }
            return (float)rounded;
        }
    }
}
