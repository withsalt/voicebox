namespace VoiceBox.VAD.FireRed
{
    /// <summary>
    /// 与 kaldi-native-fbank 行为一致的 80 维 log-mel filterbank 特征提取。
    /// 默认配置：16kHz / 25ms 帧长（400 样本）/ 10ms 帧移（160 样本）/ 80 mel bins / Povey 窗口 /
    /// 0.97 预加重 / 去直流 / 功率谱 / log 取自然对数 / snip_edges=true。
    /// 调用方负责管理跨调用的样本剩余（remainder）。
    /// </summary>
    internal sealed class KaldiFbankExtractor
    {
        public const int SampleRate = 16000;
        public const int FrameLengthMs = 25;
        public const int FrameShiftMs = 10;
        public const int FrameLength = SampleRate * FrameLengthMs / 1000;   // 400
        public const int FrameShift = SampleRate * FrameShiftMs / 1000;     // 160
        public const int FftSize = 512;                                      // 大于等于 400 的最近 2 的幂
        public const int NumBins = 80;
        private const float PreemphCoef = 0.97f;
        private const float LowFreqHz = 20f;
        // float 的 epsilon，对应 kaldi 中 std::numeric_limits<float>::epsilon()
        private const float LogEpsilon = 1.1920929e-7f;

        private readonly float[] _window;
        private readonly float[][] _melFilter; // [NumBins][FftSize/2]
        private readonly float[] _twiddleRe;
        private readonly float[] _twiddleIm;

        public KaldiFbankExtractor()
        {
            _window = ComputePoveyWindow(FrameLength);
            _melFilter = ComputeMelFilters(NumBins, FftSize, SampleRate, LowFreqHz, SampleRate / 2.0f);
            (_twiddleRe, _twiddleIm) = ComputeFftTwiddles(FftSize);
        }

        /// <summary>
        /// 从给定的连续样本（int16 量级的 float）中抽取尽可能多的特征帧。
        /// </summary>
        /// <param name="audioInt16Scale">int16 量级（-32768 ~ 32767）的浮点采样。</param>
        /// <param name="consumedSamples">输出实际被消耗的样本数（numFrames * FrameShift）。</param>
        /// <returns>抽取到的若干帧，每帧 80 维 log-mel 能量。</returns>
        public float[][] Extract(ReadOnlySpan<float> audioInt16Scale, out int consumedSamples)
        {
            if (audioInt16Scale.Length < FrameLength)
            {
                consumedSamples = 0;
                return Array.Empty<float[]>();
            }

            int numFrames = (audioInt16Scale.Length - FrameLength) / FrameShift + 1;
            consumedSamples = numFrames * FrameShift;
            var result = new float[numFrames][];

            var frameBuf = new float[FftSize];
            var fftRe = new float[FftSize];
            var fftIm = new float[FftSize];

            for (int i = 0; i < numFrames; i++)
            {
                int offset = i * FrameShift;

                // 1) 取 400 样本，去直流（减去帧均值）
                float mean = 0f;
                for (int j = 0; j < FrameLength; j++) mean += audioInt16Scale[offset + j];
                mean /= FrameLength;
                for (int j = 0; j < FrameLength; j++) frameBuf[j] = audioInt16Scale[offset + j] - mean;

                // 2) 预加重（与 kaldi 一致：从尾向头）
                for (int j = FrameLength - 1; j >= 1; j--)
                    frameBuf[j] -= PreemphCoef * frameBuf[j - 1];
                frameBuf[0] -= PreemphCoef * frameBuf[0];

                // 3) Povey 加窗
                for (int j = 0; j < FrameLength; j++) frameBuf[j] *= _window[j];

                // 4) 0 填充到 FFT 大小
                for (int j = FrameLength; j < FftSize; j++) frameBuf[j] = 0f;

                // 5) FFT
                Buffer.BlockCopy(frameBuf, 0, fftRe, 0, FftSize * sizeof(float));
                Array.Clear(fftIm, 0, FftSize);
                Fft(fftRe, fftIm, _twiddleRe, _twiddleIm);

                // 6) 功率谱（前 FftSize/2 个 bin，与 kaldi 一致，舍弃 Nyquist）
                int numFftBins = FftSize / 2;
                // 7) Mel filterbank + log
                var mel = new float[NumBins];
                for (int b = 0; b < NumBins; b++)
                {
                    float[] filt = _melFilter[b];
                    float sum = 0f;
                    for (int j = 0; j < numFftBins; j++)
                    {
                        if (filt[j] == 0f) continue;
                        float p = fftRe[j] * fftRe[j] + fftIm[j] * fftIm[j];
                        sum += filt[j] * p;
                    }
                    mel[b] = MathF.Log(sum < LogEpsilon ? LogEpsilon : sum);
                }

                result[i] = mel;
            }

            return result;
        }

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
            // 预计算每一层 butterfly 的旋转因子 e^{-j*2*pi*k/len}，按层连续存储。
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

        /// <summary>
        /// 原地基-2 Cooley-Tukey FFT（前向）。n 必须是 2 的幂。
        /// </summary>
        private static void Fft(float[] re, float[] im, float[] twiddleRe, float[] twiddleIm)
        {
            int n = re.Length;

            // bit-reversal 置换
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
                        float tRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
                        float tIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
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
