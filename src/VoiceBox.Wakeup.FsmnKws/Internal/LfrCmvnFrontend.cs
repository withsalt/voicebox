namespace VoiceBox.Wakeup.FsmnKws.Internal
{
    /// <summary>
    /// 流式 LFR(m=5, n=3) + CMVN, 与 FunASR WavFrontend 的 apply_lfr / apply_cmvn 对齐:
    /// 80 维 fbank 帧 → 左侧复制首帧 (m-1)/2=2 帧 → 每 3 帧取 1 个窗口堆叠 5 帧 → 400 维 →
    /// (x + AddShift) * Rescale。
    /// 流式约定: 不做尾部补帧 (连续音频流没有"结尾"), LFR 帧 i 在 fbank 帧 i*3+2 就绪后产出。
    /// </summary>
    internal sealed class LfrCmvnFrontend
    {
        public const int LfrM = 5;
        public const int LfrN = 3;
        public const int InputDim = StreamingKaldiFbank.NumBins; // 80
        public const int OutputDim = InputDim * LfrM;            // 400

        private readonly float[] _addShift;
        private readonly float[] _rescale;
        private long _nextLfrFrame;

        public LfrCmvnFrontend(IReadOnlyList<float> addShift, IReadOnlyList<float> rescale)
        {
            _addShift = addShift.ToArray();
            _rescale = rescale.ToArray();
            if (_addShift.Length != OutputDim || _rescale.Length != OutputDim)
                throw new InvalidDataException(
                    $"CMVN dim mismatch: expected {OutputDim}, got AddShift={_addShift.Length}, Rescale={_rescale.Length}.");
            if (_addShift.Any(value => !float.IsFinite(value))
                || _rescale.Any(value => !float.IsFinite(value) || value <= 0f))
                throw new InvalidDataException("CMVN values must be finite; Rescale values must be positive.");
        }

        /// <summary>下一个待产出的 LFR 帧号 (绝对索引, 从 0 起)。</summary>
        public long NextLfrFrame => _nextLfrFrame;

        public void Reset() => _nextLfrFrame = 0;

        /// <summary>当前 fbank 状态下可立即产出的 LFR 帧数。</summary>
        public int PendingFrameCount(StreamingKaldiFbank fbank)
        {
            // LFR 帧 i 需要 fbank 帧 i*3+2 就绪
            long ready = (fbank.NumFramesReady - (LfrM - 1) / 2 + LfrN - 1) / LfrN - _nextLfrFrame;
            return ready > 0 ? (int)ready : 0;
        }

        /// <summary>
        /// 从 fbank 取尽可能多的就绪 LFR 帧, 平铺写入 output (行优先 [n, 400]), 返回产出帧数。
        /// output 容量不足时只产出放得下的整数帧。
        /// </summary>
        public int ProduceFrames(StreamingKaldiFbank fbank, Span<float> output)
        {
            int capacity = output.Length / OutputDim;
            int produced = 0;
            while (produced < capacity
                   && _nextLfrFrame * LfrN + (LfrM - 1) / 2 < fbank.NumFramesReady)
            {
                Span<float> vec = output.Slice(produced * OutputDim, OutputDim);
                long baseIndex = _nextLfrFrame * LfrN - (LfrM - 1) / 2;
                for (int j = 0; j < LfrM; j++)
                {
                    long src = baseIndex + j;
                    if (src < 0) src = 0; // 左侧复制首帧
                    fbank.GetFrame(src).CopyTo(vec.Slice(j * InputDim, InputDim));
                }
                for (int d = 0; d < OutputDim; d++)
                    vec[d] = (vec[d] + _addShift[d]) * _rescale[d];

                _nextLfrFrame++;
                produced++;
            }

            // 后续 LFR 帧需要的最早 fbank 帧, 之前的可以释放
            long minNeeded = Math.Max(0, _nextLfrFrame * LfrN - (LfrM - 1) / 2);
            fbank.Pop(minNeeded);
            return produced;
        }
    }
}
