namespace VoiceBox.Wakeup.FsmnKws.Internal
{
    /// <summary>
    /// 与 FunASR WavFrontend (torchaudio.compliance.kaldi.fbank) 对齐的流式 80 维 log-mel fbank。
    /// <para>
    /// 关键配置 (与 xiaoyun KWS 模型训练一致, 区别于 sherpa-onnx 约定):
    /// 16kHz / 25ms 帧长 / 10ms 帧移 / 80 mel bins / Hamming 窗 / 0.97 预加重 / 去直流 /
    /// 功率谱 / log 自然对数 / snip_edges=true / dither=0 / low_freq=20Hz / high_freq=Nyquist(8000Hz) /
    /// 输入样本 [-1, 1], 内部按 kaldi 约定放大 32768 倍。
    /// </para>
    /// </summary>
    internal sealed class StreamingKaldiFbank
    {
        public const int SampleRate = 16000;
        public const int FrameLength = 400;   // 25ms
        public const int FrameShift = 160;    // 10ms
        public const int FftSize = 512;
        public const int NumBins = 80;
        private const float PreemphCoef = 0.97f;
        private const float LowFreqHz = 20f;
        private const float HighFreqHz = SampleRate / 2f; // 8000Hz (kaldi high_freq=0 → Nyquist)
        private const float SampleScale = 32768f;         // kaldi 以 int16 幅度计算
        private const float LogFloor = 1.1920929e-7f;     // float epsilon (torchaudio 同款)
        private const int CompactionThresholdSamples = SampleRate * 5;
        private const int MaxFreeFrameBuffers = 128;

        private readonly float[] _window;
        // mel filter 稀疏存储: 每个 bin 只保留非零段 [start, start+weights.Length)
        private readonly int[] _melStart;
        private readonly float[][] _melWeights;
        private readonly float[] _twiddleRe;
        private readonly float[] _twiddleIm;

        // 样本缓冲 (增长数组)。绝对样本索引 = _sampleStart + 缓冲内下标。
        private float[] _samples = new float[SampleRate];
        private int _sampleCount;
        private long _sampleStart;

        // 已发出的帧。绝对帧号用 long (10ms/帧, int 约 248 天连续运行会溢出)。
        private readonly List<float[]> _frames = new();
        private readonly Stack<float[]> _freeFrames = new();
        private long _firstFrameInBuffer;
        private long _numFramesReady;

        public long NumFramesReady => _numFramesReady;
        public int FeatureDim => NumBins;

        public StreamingKaldiFbank()
        {
            _window = ComputeHammingWindow(FrameLength);
            (_melStart, _melWeights) = ComputeMelFilters(NumBins, FftSize, SampleRate, LowFreqHz, HighFreqHz);
            (_twiddleRe, _twiddleIm) = ComputeFftTwiddles(FftSize);
        }

        public void Reset()
        {
            _sampleCount = 0;
            _sampleStart = 0;
            for (int i = 0; i < _frames.Count; i++)
                RecycleFrameBuffer(_frames[i]);
            _frames.Clear();
            _firstFrameInBuffer = 0;
            _numFramesReady = 0;
        }

        /// <summary>接收新的音频样本 ([-1, 1] float32, 16kHz)。</summary>
        public void AcceptWaveform(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty) return;
            EnsureSampleCapacity(_sampleCount + samples.Length);
            var dst = _samples.AsSpan(_sampleCount, samples.Length);
            for (int i = 0; i < samples.Length; i++) dst[i] = samples[i] * SampleScale;
            _sampleCount += samples.Length;

            ProcessAvailableFrames();
            CompactSampleBuffer();
        }

        /// <summary>取第 frameIndex 帧 (绝对索引)。</summary>
        public ReadOnlySpan<float> GetFrame(long frameIndex)
        {
            if (frameIndex < _firstFrameInBuffer)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} has been discarded (first available: {_firstFrameInBuffer}).");
            long localIndex = frameIndex - _firstFrameInBuffer;
            if (localIndex >= _frames.Count)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} is not ready yet (ready: {_numFramesReady}).");
            return _frames[(int)localIndex];
        }

        /// <summary>丢弃所有 &lt; frameIndex 的帧缓存。</summary>
        public void Pop(long frameIndex)
        {
            long discardCount = frameIndex - _firstFrameInBuffer;
            if (discardCount <= 0) return;
            if (discardCount > _frames.Count) discardCount = _frames.Count;
            int discard = (int)discardCount;
            for (int i = 0; i < discard; i++)
                RecycleFrameBuffer(_frames[i]);
            _frames.RemoveRange(0, discard);
            _firstFrameInBuffer += discardCount;
        }

        private void EnsureSampleCapacity(int required)
        {
            if (required <= _samples.Length) return;
            int newCap = _samples.Length * 2;
            while (newCap < required) newCap *= 2;
            Array.Resize(ref _samples, newCap);
        }

        private void ProcessAvailableFrames()
        {
            // snip_edges=true: 第 i 帧 = 样本 [i*160, i*160+400), 无镜像填充。
            long totalSamples = _sampleStart + _sampleCount;
            long target = totalSamples < FrameLength ? 0L : (totalSamples - FrameLength) / FrameShift + 1;
            if (target <= _numFramesReady) return;

            for (long i = _numFramesReady; i < target; i++) ProduceFrame(i);
            _numFramesReady = target;
        }

        private void ProduceFrame(long frameIndex)
        {
            long startSample = frameIndex * FrameShift;
            Span<float> frame = stackalloc float[FftSize];
            _samples.AsSpan((int)(startSample - _sampleStart), FrameLength).CopyTo(frame);

            // 去直流
            float mean = 0f;
            for (int j = 0; j < FrameLength; j++) mean += frame[j];
            mean /= FrameLength;
            for (int j = 0; j < FrameLength; j++) frame[j] -= mean;

            // 预加重 (kaldi: 从尾向头)
            for (int j = FrameLength - 1; j >= 1; j--)
                frame[j] -= PreemphCoef * frame[j - 1];
            frame[0] -= PreemphCoef * frame[0];

            // 加窗 + 零填充
            for (int j = 0; j < FrameLength; j++) frame[j] *= _window[j];
            for (int j = FrameLength; j < FftSize; j++) frame[j] = 0f;

            Span<float> re = stackalloc float[FftSize];
            Span<float> im = stackalloc float[FftSize];
            frame.CopyTo(re);
            im.Clear();
            Fft(re, im, _twiddleRe, _twiddleIm);

            // 功率谱只算一次 (mel 滤波器组相邻 bin 大量重叠)
            int numFftBins = FftSize / 2;
            Span<float> power = stackalloc float[numFftBins];
            for (int j = 0; j < numFftBins; j++)
            {
                float r = re[j];
                float ii = im[j];
                power[j] = r * r + ii * ii;
            }

            float[] mel = RentFrameBuffer();
            for (int b = 0; b < NumBins; b++)
            {
                float[] w = _melWeights[b];
                int start = _melStart[b];
                float sum = 0f;
                for (int j = 0; j < w.Length; j++)
                    sum += w[j] * power[start + j];
                mel[b] = MathF.Log(sum < LogFloor ? LogFloor : sum);
            }
            _frames.Add(mel);
        }

        private float[] RentFrameBuffer() =>
            _freeFrames.Count > 0 ? _freeFrames.Pop() : new float[NumBins];

        private void RecycleFrameBuffer(float[] frame)
        {
            if (_freeFrames.Count < MaxFreeFrameBuffers)
                _freeFrames.Push(frame);
        }

        private void CompactSampleBuffer()
        {
            long needFrom = _numFramesReady * FrameShift;
            long unused = needFrom - _sampleStart;
            if (unused < CompactionThresholdSamples) return;
            int keep = _sampleCount - (int)unused;
            Array.Copy(_samples, (int)unused, _samples, 0, keep);
            _sampleCount = keep;
            _sampleStart = needFrom;
        }

        // ----------------- helpers -----------------
        private static float[] ComputeHammingWindow(int n)
        {
            var w = new float[n];
            double a = 2.0 * Math.PI / (n - 1);
            for (int i = 0; i < n; i++)
                w[i] = (float)(0.54 - 0.46 * Math.Cos(a * i));
            return w;
        }

        private static float MelScale(float freq) => 1127f * MathF.Log(1f + freq / 700f);

        private static (int[] starts, float[][] weights) ComputeMelFilters(
            int numBins, int fftSize, int sampleRate, float lowFreq, float highFreq)
        {
            int numFftBins = fftSize / 2;
            float fftBinWidth = sampleRate / (float)fftSize;
            float melLow = MelScale(lowFreq);
            float melHigh = MelScale(highFreq);
            float melDelta = (melHigh - melLow) / (numBins + 1);

            var starts = new int[numBins];
            var weights = new float[numBins][];
            var dense = new float[numFftBins];
            for (int b = 0; b < numBins; b++)
            {
                float left = melLow + b * melDelta;
                float center = melLow + (b + 1) * melDelta;
                float right = melLow + (b + 2) * melDelta;
                int first = -1, last = -1;
                for (int j = 0; j < numFftBins; j++)
                {
                    float freq = fftBinWidth * j;
                    float mel = MelScale(freq);
                    if (mel <= left || mel >= right) { dense[j] = 0f; continue; }
                    dense[j] = mel <= center
                        ? (mel - left) / (center - left)
                        : (right - mel) / (right - center);
                    if (first < 0) first = j;
                    last = j;
                }
                if (first < 0) { starts[b] = 0; weights[b] = Array.Empty<float>(); continue; }
                starts[b] = first;
                weights[b] = dense.AsSpan(first, last - first + 1).ToArray();
            }
            return (starts, weights);
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
