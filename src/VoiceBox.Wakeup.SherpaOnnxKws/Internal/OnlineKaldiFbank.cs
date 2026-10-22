namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 与 kaldi-native-fbank 行为对齐的流式 80 维 log-mel filterbank 特征提取。
    /// 用于 sherpa-onnx KWS Zipformer-Transducer 模型的输入特征。
    /// <para>
    /// 关键配置 (与 sherpa-onnx 默认 KWS 一致):
    /// 16kHz / 25ms 帧长 / 10ms 帧移 / 80 mel bins / Povey 窗口 / 0.97 预加重 /
    /// 去直流 / 功率谱 / log 自然对数 / snip_edges=false / dither=0 /
    /// low_freq=20Hz / high_freq=Nyquist - 400 = 7600Hz / 输入样本范围 [-1, 1] (normalize_samples=true)。
    /// </para>
    /// </summary>
    internal sealed class OnlineKaldiFbank
    {
        public const int SampleRate = 16000;
        public const int FrameLengthMs = 25;
        public const int FrameShiftMs = 10;
        public const int FrameLength = SampleRate * FrameLengthMs / 1000;   // 400
        public const int FrameShift = SampleRate * FrameShiftMs / 1000;     // 160
        public const int FftSize = 512;                                      // 大于等于 FrameLength 的最近 2 的幂
        public const int NumBins = 80;
        private const float PreemphCoef = 0.97f;
        private const float LowFreqHz = 20f;
        private const float HighFreqHz = SampleRate / 2f - 400f;             // 7600Hz, 与 sherpa-onnx 默认一致
        private const float LogFloor = 1.1920929e-7f;                        // float epsilon
        // 长流式运行时, 当样本缓冲区相对待用样本超过该阈值时压缩
        private const int CompactionThresholdSamples = SampleRate * 5; // 5秒

        private readonly float[] _window;
        private readonly float[][] _melFilter;
        private readonly float[] _twiddleRe;
        private readonly float[] _twiddleIm;

        // 样本缓冲。绝对样本索引 = _sampleStart + 在缓冲中的下标。
        private readonly List<float> _samples = new();
        private long _sampleStart; // 已经从 _samples 开头丢弃的样本数

        // 已发出的帧 (尾对齐到 _firstFrameInBuffer)
        // 绝对帧号用 long: 每帧 10ms, int 在约 248 天连续运行(无 Reset)后会溢出转负, 破坏
        // GetFrame/Pop 的帧号算术 (越界/不再丢帧导致内存增长)。帧在缓冲内的局部下标仍是 int。
        private readonly List<float[]> _frames = new();
        private long _firstFrameInBuffer;
        private long _numFramesReady;
        private bool _inputFinished;

        public long NumFramesReady => _numFramesReady;
        public int FeatureDim => NumBins;

        public OnlineKaldiFbank()
        {
            _window = ComputePoveyWindow(FrameLength);
            _melFilter = ComputeMelFilters(NumBins, FftSize, SampleRate, LowFreqHz, HighFreqHz);
            (_twiddleRe, _twiddleIm) = ComputeFftTwiddles(FftSize);
        }

        /// <summary>清空所有内部状态。</summary>
        public void Reset()
        {
            _samples.Clear();
            _sampleStart = 0;
            _frames.Clear();
            _firstFrameInBuffer = 0;
            _numFramesReady = 0;
            _inputFinished = false;
        }

        /// <summary>
        /// 接收新的音频样本 (要求归一化到 [-1, 1] 的 float32, 采样率必须等于 <see cref="SampleRate"/>)。
        /// </summary>
        public void AcceptWaveform(int samplingRate, ReadOnlySpan<float> samples)
        {
            if (samplingRate != SampleRate)
                throw new ArgumentException(
                    $"OnlineKaldiFbank only supports {SampleRate}Hz, but got {samplingRate}Hz.", nameof(samplingRate));
            if (_inputFinished)
                throw new InvalidOperationException("AcceptWaveform called after InputFinished.");
            if (samples.IsEmpty) return;

            _samples.Capacity = Math.Max(_samples.Capacity, _samples.Count + samples.Length);
            for (int i = 0; i < samples.Length; i++) _samples.Add(samples[i]);

            ProcessAvailableFrames();
            CompactSampleBuffer();
        }

        /// <summary>
        /// 输入终止后, 处理可能剩余的尾部样本 (snip_edges=false 时可以补出最后一帧)。
        /// </summary>
        public void InputFinished()
        {
            if (_inputFinished) return;
            _inputFinished = true;
            ProcessAvailableFrames();
        }

        /// <summary>
        /// 取第 frameIndex 帧 (绝对索引, 从 0 开始)。返回 NumBins 维 float 切片。
        /// </summary>
        public ReadOnlySpan<float> GetFrame(long frameIndex)
        {
            if (frameIndex < _firstFrameInBuffer)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} has been discarded (first available: {_firstFrameInBuffer}).");
            long localIndex = frameIndex - _firstFrameInBuffer;
            if (localIndex >= _frames.Count)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} is not ready yet (ready: {_numFramesReady}).");
            // localIndex ∈ [0, _frames.Count) (上面两个判断保证), 转回 int 安全。
            return _frames[(int)localIndex];
        }

        /// <summary>
        /// 复制连续 n 帧 (从 frameIndex 开始) 到目标缓冲, 形状为 [n, NumBins], 行优先。
        /// </summary>
        public void CopyFrames(long frameIndex, int n, Span<float> destination)
        {
            if (destination.Length < n * NumBins)
                throw new ArgumentException("destination too small.", nameof(destination));
            for (int i = 0; i < n; i++)
            {
                var src = GetFrame(frameIndex + i);
                src.CopyTo(destination.Slice(i * NumBins, NumBins));
            }
        }

        /// <summary>
        /// 丢弃所有 &lt; frameIndex 的帧缓存以释放内存。
        /// </summary>
        public void Pop(long frameIndex)
        {
            long discardCount = frameIndex - _firstFrameInBuffer;
            if (discardCount <= 0) return;
            if (discardCount > _frames.Count) discardCount = _frames.Count;
            // discardCount 已夹到 [1, _frames.Count], 转回 int 安全。
            _frames.RemoveRange(0, (int)discardCount);
            _firstFrameInBuffer += discardCount;
        }

        // ----------------- 帧提取核心 -----------------
        private void ProcessAvailableFrames()
        {
            // snip_edges=false 帧数公式 (来自 kaldi-native-fbank):
            //   InputFinished 后:  num_frames = (num_samples + frame_shift / 2) / frame_shift
            //                      最后若干帧可能用 mirror 填充右侧。
            //   在线 (未 InputFinished): 只发出 "右侧不需要 mirror" 的帧, 即第 i 帧右端样本
            //   索引 (i * FrameShift + FrameShift/2 + FrameLength/2 - 1) 必须在已有样本范围内。
            long totalSamples = _sampleStart + _samples.Count;

            long target;
            if (_inputFinished)
            {
                target = (totalSamples + FrameShift / 2) / FrameShift;
            }
            else
            {
                long needRight = FrameShift / 2 + FrameLength / 2; // 280
                target = totalSamples < needRight ? 0L : (totalSamples - needRight) / FrameShift + 1;
            }

            if (target <= _numFramesReady) return;

            for (long i = _numFramesReady; i < target; i++)
            {
                ProduceFrame(i, totalSamples);
            }
            _numFramesReady = target;
        }

        private void ProduceFrame(long frameIndex, long totalSamples)
        {
            // 第 i 帧中心 = i * FrameShift + FrameShift / 2; 起点 = 中心 - FrameLength / 2
            long startSample = (long)frameIndex * FrameShift + FrameShift / 2 - FrameLength / 2;
            Span<float> frame = stackalloc float[FftSize];
            // kaldi 反射式越界处理:
            //   idx < 0:           idx = -idx - 1
            //   idx >= total:      idx = 2 * total - idx - 1
            for (int j = 0; j < FrameLength; j++)
            {
                long idx = startSample + j;
                if (idx < 0) idx = -idx - 1;
                if (idx >= totalSamples) idx = 2 * totalSamples - idx - 1;
                if (idx < 0) idx = 0; // 极端短输入兜底
                frame[j] = _samples[(int)(idx - _sampleStart)];
            }

            // 去直流
            float mean = 0f;
            for (int j = 0; j < FrameLength; j++) mean += frame[j];
            mean /= FrameLength;
            for (int j = 0; j < FrameLength; j++) frame[j] -= mean;

            // 预加重 (kaldi: 从尾向头)
            for (int j = FrameLength - 1; j >= 1; j--)
                frame[j] -= PreemphCoef * frame[j - 1];
            frame[0] -= PreemphCoef * frame[0];

            // 加窗
            for (int j = 0; j < FrameLength; j++) frame[j] *= _window[j];
            // 零填充到 FftSize
            for (int j = FrameLength; j < FftSize; j++) frame[j] = 0f;

            // FFT (原地)
            Span<float> re = stackalloc float[FftSize];
            Span<float> im = stackalloc float[FftSize];
            frame.CopyTo(re);
            im.Clear();
            Fft(re, im, _twiddleRe, _twiddleIm);

            int numFftBins = FftSize / 2;
            var mel = new float[NumBins];
            for (int b = 0; b < NumBins; b++)
            {
                float[] filt = _melFilter[b];
                float sum = 0f;
                for (int j = 0; j < numFftBins; j++)
                {
                    float w = filt[j];
                    if (w == 0f) continue;
                    float r = re[j];
                    float ii = im[j];
                    sum += w * (r * r + ii * ii);
                }
                mel[b] = MathF.Log(sum < LogFloor ? LogFloor : sum);
            }
            _frames.Add(mel);
        }

        /// <summary>
        /// 长时间运行下压缩 _samples: 把绝对索引 &lt; minKeep 之前的样本丢弃,
        /// 同步更新 _sampleStart, 使后续 ProduceFrame 仍按绝对索引访问。
        /// </summary>
        private void CompactSampleBuffer()
        {
            // 下一帧 (frame = _numFramesReady) 需要的最早样本绝对索引:
            //   start = _numFramesReady * FrameShift + FrameShift/2 - FrameLength/2 (可能为负, 用 mirror)
            long needFrom = (long)_numFramesReady * FrameShift + FrameShift / 2 - FrameLength / 2;
            if (needFrom < 0) needFrom = 0;
            long unused = needFrom - _sampleStart;
            if (unused < CompactionThresholdSamples) return;

            // 留 1 帧的左 mirror 余量 (FrameLength 个样本), 避免后续 ProduceFrame 访问已丢弃的样本
            long keepFrom = needFrom - FrameLength;
            if (keepFrom < _sampleStart) keepFrom = _sampleStart;
            int discard = (int)(keepFrom - _sampleStart);
            if (discard <= 0) return;
            _samples.RemoveRange(0, discard);
            _sampleStart += discard;
        }

        // ----------------- helpers -----------------
        private static float[] ComputePoveyWindow(int n)
        {
            var w = new float[n];
            double a = 2.0 * Math.PI / (n - 1);
            for (int i = 0; i < n; i++)
            {
                w[i] = (float)Math.Pow(0.5 - 0.5 * Math.Cos(a * i), 0.85);
            }
            return w;
        }

        private static float MelScale(float freq) => 1127f * MathF.Log(1f + freq / 700f);

        private static float[][] ComputeMelFilters(int numBins, int fftSize, int sampleRate, float lowFreq, float highFreq)
        {
            int numFftBins = fftSize / 2;
            float fftBinWidth = sampleRate / (float)fftSize;
            float melLow = MelScale(lowFreq);
            float melHigh = MelScale(highFreq);
            float melDelta = (melHigh - melLow) / (numBins + 1);

            var filters = new float[numBins][];
            for (int b = 0; b < numBins; b++)
            {
                float left = melLow + b * melDelta;
                float center = melLow + (b + 1) * melDelta;
                float right = melLow + (b + 2) * melDelta;
                var filt = new float[numFftBins];
                for (int j = 0; j < numFftBins; j++)
                {
                    float freq = fftBinWidth * j;
                    float mel = MelScale(freq);
                    if (mel <= left || mel >= right) continue;
                    float w = mel <= center
                        ? (mel - left) / (center - left)
                        : (right - mel) / (right - center);
                    filt[j] = w;
                }
                filters[b] = filt;
            }
            return filters;
        }

        private static (float[] re, float[] im) ComputeFftTwiddles(int n)
        {
            int totalSize = 0;
            for (int len = 2; len <= n; len <<= 1) totalSize += len >> 1;
            var re = new float[totalSize];
            var im = new float[totalSize];
            int pos = 0;
            for (int len = 2; len <= n; len <<= 1)
            {
                int half = len >> 1;
                double baseAngle = -2.0 * Math.PI / len;
                for (int k = 0; k < half; k++)
                {
                    re[pos + k] = (float)Math.Cos(baseAngle * k);
                    im[pos + k] = (float)Math.Sin(baseAngle * k);
                }
                pos += half;
            }
            return (re, im);
        }

        private static void Fft(Span<float> re, Span<float> im, float[] twiddleRe, float[] twiddleIm)
        {
            int n = re.Length;

            int j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j)
                {
                    (re[i], re[j]) = (re[j], re[i]);
                    (im[i], im[j]) = (im[j], im[i]);
                }
            }

            int twPos = 0;
            for (int len = 2; len <= n; len <<= 1)
            {
                int half = len >> 1;
                for (int i = 0; i < n; i += len)
                {
                    for (int k = 0; k < half; k++)
                    {
                        float wRe = twiddleRe[twPos + k];
                        float wIm = twiddleIm[twPos + k];
                        float aRe = re[i + k];
                        float aIm = im[i + k];
                        float bRe = re[i + k + half];
                        float bIm = im[i + k + half];
                        float tRe = bRe * wRe - bIm * wIm;
                        float tIm = bRe * wIm + bIm * wRe;
                        re[i + k] = aRe + tRe;
                        im[i + k] = aIm + tIm;
                        re[i + k + half] = aRe - tRe;
                        im[i + k + half] = aIm - tIm;
                    }
                }
                twPos += half;
            }
        }
    }
}
