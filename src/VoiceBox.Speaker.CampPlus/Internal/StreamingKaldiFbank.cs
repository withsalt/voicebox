using System.Buffers;

namespace VoiceBox.Speaker.CampPlus.Internal
{
    /// <summary>
    /// 与 torchaudio.compliance.kaldi.fbank (modelscope CAM++ SV pipeline 同款) 对齐的流式 80 维 log-mel fbank。
    /// <para>
    /// 关键配置 (区别于 FunASR/sherpa-onnx 约定的是窗函数为 Povey):
    /// 16kHz / 25ms 帧长 / 10ms 帧移 / 80 mel bins / Povey 窗 / 0.97 预加重 / 去直流 /
    /// 功率谱 / log 自然对数 / snip_edges=true / dither=0 / low_freq=20Hz / high_freq=Nyquist(8000Hz) /
    /// 输入样本 [-1, 1], 内部按 kaldi 约定放大 32768 倍。
    /// 每帧同时记录原始 RMS ([-1,1] 幅度) 供静音门使用。
    /// </para>
    /// </summary>
    internal sealed class StreamingKaldiFbank
    {
        public const int SampleRate = 16000;
        public const int FrameLength = 400;   // 25ms
        public const int FrameShift = 160;    // 10ms
        public const int FftSize = 512;
        private const float LowFreqHz = 20f;
        private const float HighFreqHz = SampleRate / 2f; // 8000Hz (kaldi high_freq=0 → Nyquist)
        private const float LogFloor = 1.1920929e-7f;     // float epsilon (torchaudio 同款)
        // 已消费样本只需保留不足一帧的尾部；每秒压缩一次只搬移约 240 个 float，
        // 同时避免实时链路为了 5 秒历史样本把缓冲扩到 128K 元素。
        private const int CompactionThresholdSamples = SampleRate;
        private const int InitialFrameCapacity = 64;

        private readonly int _numBins;
        private readonly float _preemphCoef;
        private readonly bool _removeDcOffset;
        private readonly float _sampleScale;
        private readonly float[] _window;
        // mel filter 稀疏存储: 每个 bin 只保留非零段 [start, start+weights.Length)
        private readonly int[] _melStart;
        private readonly float[][] _melWeights;
        private readonly float[] _twiddleRe;
        private readonly float[] _twiddleIm;

        // 样本缓冲 (增长数组)。绝对样本索引 = _sampleStart + 缓冲内下标。
        private float[] _samples = new float[SampleRate + FrameLength];
        private int _sampleCount;
        private long _sampleStart;

        // 已发出的帧以连续环形缓冲保存。避免每 10ms 分配一个 float[]，也避免 Pop 时 List.RemoveRange 搬移引用。
        // 绝对帧号用 long (10ms/帧, int 约 248 天连续运行会溢出)。
        private float[] _frameData;
        private float[] _frameRms;
        private int _frameHead;
        private int _frameCount;
        private long _firstFrameInBuffer;
        private long _numFramesReady;

        public long NumFramesReady => _numFramesReady;
        public int FeatureDim => _numBins;

        public StreamingKaldiFbank(int numBins = 80, string windowType = "povey",
            float preemphCoef = 0.97f, bool removeDcOffset = true, float sampleScale = 32768f,
            int initialFrameCapacity = InitialFrameCapacity)
        {
            if (initialFrameCapacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(initialFrameCapacity));
            _numBins = numBins;
            _preemphCoef = preemphCoef;
            _removeDcOffset = removeDcOffset;
            _sampleScale = sampleScale;
            _window = ComputeWindow(FrameLength, windowType);
            (_melStart, _melWeights) = ComputeMelFilters(numBins, FftSize, SampleRate, LowFreqHz, HighFreqHz);
            (_twiddleRe, _twiddleIm) = ComputeFftTwiddles(FftSize);
            _frameData = new float[initialFrameCapacity * numBins];
            _frameRms = new float[initialFrameCapacity];
        }

        public void Reset()
        {
            _sampleCount = 0;
            _sampleStart = 0;
            _frameHead = 0;
            _frameCount = 0;
            _firstFrameInBuffer = 0;
            _numFramesReady = 0;
        }

        /// <summary>接收新的音频样本 ([-1, 1] float32, 16kHz)。</summary>
        public void AcceptWaveform(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty) return;
            EnsureSampleCapacity(_sampleCount + samples.Length);
            var dst = _samples.AsSpan(_sampleCount, samples.Length);
            for (int i = 0; i < samples.Length; i++) dst[i] = samples[i] * _sampleScale;
            _sampleCount += samples.Length;

            ProcessAvailableFrames();
            CompactSampleBuffer();
        }

        /// <summary>取第 frameIndex 帧 (绝对索引)。</summary>
        public ReadOnlySpan<float> GetFrame(long frameIndex)
        {
            int offset = GetFrameOffset(frameIndex);
            int slot = (_frameHead + offset) % _frameRms.Length;
            return _frameData.AsSpan(slot * _numBins, _numBins);
        }

        /// <summary>取第 frameIndex 帧 (绝对索引) 的原始 RMS ([-1,1] 幅度)。</summary>
        public float GetFrameRms(long frameIndex)
        {
            int offset = GetFrameOffset(frameIndex);
            return _frameRms[(_frameHead + offset) % _frameRms.Length];
        }

        /// <summary>计算连续帧 RMS 的平方均值之和；累加顺序与逐帧 GetFrameRms 完全相同。</summary>
        public double SumSquaredRms(long firstFrame, int frameCount)
        {
            if (frameCount < 0)
                throw new ArgumentOutOfRangeException(nameof(frameCount));
            if (frameCount == 0) return 0d;

            int offset = GetFrameOffset(firstFrame);
            if (offset + frameCount > _frameCount)
                throw new ArgumentOutOfRangeException(nameof(frameCount), "The requested frame range is not ready.");

            int firstSlot = (_frameHead + offset) % _frameRms.Length;
            int firstCount = Math.Min(frameCount, _frameRms.Length - firstSlot);
            double sum = 0d;
            for (int i = 0; i < firstCount; i++)
            {
                float value = _frameRms[firstSlot + i];
                sum += (double)value * value;
            }
            for (int i = 0; i < frameCount - firstCount; i++)
            {
                float value = _frameRms[i];
                sum += (double)value * value;
            }
            return sum;
        }

        /// <summary>连续复制若干帧到行优先目标缓冲；环形缓冲至多需要两次块复制。</summary>
        public void CopyFramesTo(long firstFrame, int frameCount, Span<float> destination)
        {
            if (frameCount < 0)
                throw new ArgumentOutOfRangeException(nameof(frameCount));
            if (destination.Length < frameCount * _numBins)
                throw new ArgumentException("Destination is too small for the requested frames.", nameof(destination));
            if (frameCount == 0) return;

            int offset = GetFrameOffset(firstFrame);
            if (offset + frameCount > _frameCount)
                throw new ArgumentOutOfRangeException(nameof(frameCount), "The requested frame range is not ready.");

            int firstSlot = (_frameHead + offset) % _frameRms.Length;
            int firstCount = Math.Min(frameCount, _frameRms.Length - firstSlot);
            _frameData.AsSpan(firstSlot * _numBins, firstCount * _numBins).CopyTo(destination);
            int remaining = frameCount - firstCount;
            if (remaining > 0)
                _frameData.AsSpan(0, remaining * _numBins).CopyTo(destination[(firstCount * _numBins)..]);
        }

        /// <summary>
        /// 离线整段特征提取，直接写入调用方的最终缓冲，不建立逐帧缓存。
        /// 系数表只读，因此可与流式处理并发调用。
        /// </summary>
        public void ComputeFeatures(ReadOnlySpan<float> samples, Span<float> destination)
        {
            int frames = samples.Length < FrameLength
                ? 0
                : (samples.Length - FrameLength) / FrameShift + 1;
            if (destination.Length < frames * _numBins)
                throw new ArgumentException("Destination is too small for the input audio.", nameof(destination));
            if (frames == 0) return;

            float[] scaledBuffer = ArrayPool<float>.Shared.Rent(samples.Length);
            try
            {
                Span<float> scaled = scaledBuffer.AsSpan(0, samples.Length);
                for (int i = 0; i < samples.Length; i++)
                    scaled[i] = samples[i] * _sampleScale;

                for (int i = 0; i < frames; i++)
                {
                    ProduceFrameCore(
                        scaled.Slice(i * FrameShift, FrameLength),
                        destination.Slice(i * _numBins, _numBins),
                        out _);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scaledBuffer);
            }
        }

        private int GetFrameOffset(long frameIndex)
        {
            if (frameIndex < _firstFrameInBuffer)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} has been discarded (first available: {_firstFrameInBuffer}).");
            long offset = frameIndex - _firstFrameInBuffer;
            if (offset >= _frameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"frame {frameIndex} is not ready yet (ready: {_numFramesReady}).");
            return (int)offset;
        }

        /// <summary>丢弃所有 &lt; frameIndex 的帧缓存。</summary>
        public void Pop(long frameIndex)
        {
            long discardCount = frameIndex - _firstFrameInBuffer;
            if (discardCount <= 0) return;
            if (discardCount > _frameCount) discardCount = _frameCount;
            int discard = (int)discardCount;
            _frameHead = (_frameHead + discard) % _frameRms.Length;
            _frameCount -= discard;
            _firstFrameInBuffer += discardCount;
        }

        private void EnsureSampleCapacity(int required)
        {
            if (required <= _samples.Length) return;
            int newCap = Math.Max(_samples.Length * 2, required);
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
            EnsureFrameCapacity(_frameCount + 1);
            int slot = (_frameHead + _frameCount) % _frameRms.Length;
            Span<float> mel = _frameData.AsSpan(slot * _numBins, _numBins);
            ProduceFrameCore(
                _samples.AsSpan((int)(startSample - _sampleStart), FrameLength),
                mel,
                out _frameRms[slot]);
            _frameCount++;
        }

        private void ProduceFrameCore(ReadOnlySpan<float> samples, Span<float> mel, out float rms)
        {
            // re 同时充当时域帧和 FFT 实部，省去一个 512 点栈缓冲及一次完整复制。
            Span<float> re = stackalloc float[FftSize];
            samples.CopyTo(re);

            // 原始 RMS (静音门用, 换算回 [-1,1] 幅度)
            float sumSq = 0f;
            for (int j = 0; j < FrameLength; j++) sumSq += re[j] * re[j];
            rms = MathF.Sqrt(sumSq / FrameLength) / _sampleScale;

            // 去直流
            if (_removeDcOffset)
            {
                float mean = 0f;
                for (int j = 0; j < FrameLength; j++) mean += re[j];
                mean /= FrameLength;
                for (int j = 0; j < FrameLength; j++) re[j] -= mean;
            }

            // 预加重 (kaldi: 从尾向头)
            if (_preemphCoef != 0f)
            {
                for (int j = FrameLength - 1; j >= 1; j--)
                    re[j] -= _preemphCoef * re[j - 1];
                re[0] -= _preemphCoef * re[0];
            }

            // 加窗 + 零填充
            for (int j = 0; j < FrameLength; j++) re[j] *= _window[j];
            re[FrameLength..].Clear();

            Span<float> im = stackalloc float[FftSize];
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

            for (int b = 0; b < _numBins; b++)
            {
                float[] w = _melWeights[b];
                int start = _melStart[b];
                float sum = 0f;
                for (int j = 0; j < w.Length; j++)
                    sum += w[j] * power[start + j];
                mel[b] = MathF.Log(sum < LogFloor ? LogFloor : sum);
            }
        }

        private void EnsureFrameCapacity(int required)
        {
            int oldCapacity = _frameRms.Length;
            if (required <= oldCapacity) return;

            int newCapacity = oldCapacity * 2;
            while (newCapacity < required) newCapacity *= 2;
            var newData = new float[newCapacity * _numBins];
            var newRms = new float[newCapacity];

            int firstCount = Math.Min(_frameCount, oldCapacity - _frameHead);
            _frameData.AsSpan(_frameHead * _numBins, firstCount * _numBins).CopyTo(newData);
            _frameRms.AsSpan(_frameHead, firstCount).CopyTo(newRms);
            int remaining = _frameCount - firstCount;
            if (remaining > 0)
            {
                _frameData.AsSpan(0, remaining * _numBins).CopyTo(newData.AsSpan(firstCount * _numBins));
                _frameRms.AsSpan(0, remaining).CopyTo(newRms.AsSpan(firstCount));
            }

            _frameData = newData;
            _frameRms = newRms;
            _frameHead = 0;
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
        private static float[] ComputeWindow(int n, string windowType)
        {
            var w = new float[n];
            double a = 2.0 * Math.PI / (n - 1);
            for (int i = 0; i < n; i++)
            {
                w[i] = windowType switch
                {
                    // kaldi povey 窗: hanning^0.85
                    "povey" => (float)Math.Pow(0.5 - 0.5 * Math.Cos(a * i), 0.85),
                    "hamming" => (float)(0.54 - 0.46 * Math.Cos(a * i)),
                    _ => throw new ArgumentException($"Unsupported window type: {windowType}", nameof(windowType)),
                };
            }
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
