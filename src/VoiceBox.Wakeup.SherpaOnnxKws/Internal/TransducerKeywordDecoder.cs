namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 关键词解码的中间结果, 跨 chunk 累积。对应 sherpa-onnx 的 TransducerKeywordResult。
    /// </summary>
    internal sealed class KeywordDecoderResult
    {
        /// <summary>已处理 (子采样后) 的累计帧数, 用作 timestamps 的基数。</summary>
        public int FrameOffset;

        /// <summary>命中关键词时取出的 token id 序列。</summary>
        public List<long> Tokens { get; } = new();

        /// <summary>命中的关键词 (= ContextState.Phrase)。</summary>
        public string Keyword { get; set; } = string.Empty;

        /// <summary>命中关键词的平均声学概率 (用于上报 score)。</summary>
        public float KeywordProb { get; set; }

        /// <summary>当前最佳假设连续解码 blank 的帧数。</summary>
        public int NumTrailingBlanks;

        /// <summary>命中关键词时取出的 token 时间戳 (子采样后帧索引, 与 Tokens 长度一致)。</summary>
        public List<int> Timestamps { get; } = new();

        /// <summary>波束搜索的所有活跃假设。</summary>
        public Hypotheses Hyps { get; set; } = new();

        /// <summary>
        /// 诊断: 命中 context-graph 终止节点但被 (尾部 blank 不够 / 平均声学概率 &lt; 阈值) 否决的事件。
        /// 每次 Decode 调用结束后由调用方读取并清空。仅用于排查 "为什么唤醒词不响"。
        /// </summary>
        public List<KeywordNearMiss> NearMisses { get; } = new();
    }

    /// <summary>
    /// 一次被 context graph 标记 "已匹配关键词全部 token" 但触发条件不足的事件。
    /// </summary>
    internal readonly struct KeywordNearMiss
    {
        public readonly string Phrase;
        public readonly float YsProb;
        public readonly float AcThreshold;
        public readonly int NumTrailingBlanks;
        public readonly int RequiredTrailingBlanks;
        /// <summary>实际命中路径上最近 Level 个 token 的 id, 用于日志诊断 "听到了什么"。</summary>
        public readonly long[] HeardTokenIds;

        public KeywordNearMiss(string phrase, float ysProb, float acThreshold,
            int numTrailingBlanks, int requiredTrailingBlanks, long[] heardTokenIds)
        {
            Phrase = phrase;
            YsProb = ysProb;
            AcThreshold = acThreshold;
            NumTrailingBlanks = numTrailingBlanks;
            RequiredTrailingBlanks = requiredTrailingBlanks;
            HeardTokenIds = heardTokenIds;
        }
    }

    /// <summary>
    /// 流式 transducer 关键词识别 (modified beam search + Aho-Corasick context graph)。
    /// 算法与 sherpa-onnx C++ 的 TransducerKeywordDecoder::Decode 一致。
    /// </summary>
    internal sealed class TransducerKeywordDecoder
    {
        private readonly TransducerModel _model;
        private readonly int _maxActivePaths;
        private readonly int _numTrailingBlanks;
        private readonly int _unkId;
        private readonly bool _emitImmediately;
        private const int BlankId = 0; // sherpa-onnx 训练好的 transducer 模型 blank 总是 0

        public TransducerKeywordDecoder(
            TransducerModel model, int maxActivePaths, int numTrailingBlanks, int unkId,
            bool emitImmediately = false)
        {
            _model = model;
            _maxActivePaths = Math.Max(1, maxActivePaths);
            _numTrailingBlanks = Math.Max(0, numTrailingBlanks);
            _unkId = unkId;
            _emitImmediately = emitImmediately;
        }

        /// <summary>
        /// 构造初始 hyp 集 (只含一条 blank 序列, log_prob=0)。
        /// 该序列长度 = context_size, 内部为 [-1, ..., -1, 0]。
        /// </summary>
        public KeywordDecoderResult GetEmptyResult()
        {
            int cs = _model.ContextSize;
            var blanks = new List<long>(cs);
            for (int i = 0; i < cs; i++) blanks.Add(-1);
            blanks[cs - 1] = BlankId;

            var hyp = new Hypothesis
            {
                Ys = new List<long>(blanks),
                LogProb = 0,
            };
            var result = new KeywordDecoderResult();
            result.Hyps.Add(hyp);
            return result;
        }

        /// <summary>
        /// 在 encoder 输出上做一次完整解码 (T' 帧逐帧 modified beam search), 命中关键词时把 token/keyword
        /// 写入 result 并重置 hyps 为初始集; 否则把 hyps 作为新状态返回。
        /// </summary>
        /// <param name="encoderOut">[N=1, T', joiner_dim]; 此处 N 固定为 1。</param>
        /// <param name="encoderOutShape">长度 3 的 shape。</param>
        /// <param name="contextGraph">该流绑定的上下文图。</param>
        /// <param name="result">解码状态 (输入: 上一次的 hyps; 输出: 新 hyps + 可能的命中信息)。</param>
        public void Decode(
            float[] encoderOut, long[] encoderOutShape,
            ContextGraph contextGraph, KeywordDecoderResult result)
        {
            if (encoderOutShape.Length != 3)
                throw new ArgumentException("encoder_out must be 3-D.", nameof(encoderOutShape));
            int batch = (int)encoderOutShape[0];
            if (batch != 1)
                throw new NotSupportedException("Only batch=1 is supported.");
            int numFrames = (int)encoderOutShape[1];
            int joinerDim = (int)encoderOutShape[2];
            int vocab = _model.VocabSize;
            int cs = _model.ContextSize;

            // 把 result.Hyps 迁出作为 prev (上一帧的所有活跃假设); cur 用于当前帧扩展后的去重集合。
            var blankSeq = new List<long>(cs);
            for (int i = 0; i < cs; i++) blankSeq.Add(-1);
            blankSeq[cs - 1] = BlankId;

            Hypotheses cur = result.Hyps;

            // 用于复用的缓冲
            var encFrame = new float[joinerDim]; // 单帧 encoder out
            long[] encFrameShape = new long[] { 1, joinerDim };

            for (int t = 0; t < numFrames; t++)
            {
                // 1) 收集 prev (所有活跃假设)
                var prev = new List<Hypothesis>(cur.Size);
                foreach (var h in cur.Values) prev.Add(h);
                int numHyps = prev.Count;
                if (numHyps == 0)
                {
                    // 极端情况: 无任何假设, 终止
                    break;
                }
                cur.Clear();

                // 2) 复制 encoder 在第 t 帧上的输出 (joinerDim)
                int srcOffset = t * joinerDim; // 因为 batch=1
                Array.Copy(encoderOut, srcOffset, encFrame, 0, joinerDim);

                // 3) Decoder forward: 把 numHyps 个假设的 last context_size tokens 一批送进 decoder
                var (decInData, decInShape) = _model.BuildDecoderInput(prev);
                float[] decoderOut = _model.RunDecoder(decInData, decInShape, out long[] decOutShape);
                // decOutShape = [numHyps, joinerDim]
                int decoderDim = (int)decOutShape[1];
                if (decoderDim != joinerDim)
                    throw new InvalidOperationException(
                        $"Joiner dim mismatch: encoder {joinerDim} vs decoder {decoderDim}.");

                // 4) Repeat encoder frame to match hyps count (复制 numHyps 份)
                var encRepeated = new float[numHyps * joinerDim];
                for (int i = 0; i < numHyps; i++)
                {
                    Array.Copy(encFrame, 0, encRepeated, i * joinerDim, joinerDim);
                }
                long[] encRepeatedShape = new long[] { numHyps, joinerDim };

                // 5) Joiner: [numHyps, joinerDim] + [numHyps, joinerDim] -> [numHyps, vocab]
                float[] logit = _model.RunJoiner(encRepeated, encRepeatedShape, decoderOut, decOutShape);

                // 6) log_softmax (per row)
                LogSoftmax(logit, numHyps, vocab);

                // 备份纯声学 logprobs (用于命中时计算平均 token 概率)
                var acousticLogprobs = (float[])logit.Clone();

                // 7) 把每条 hyp 的累计 log_prob 加上当前帧的 token logprob
                for (int i = 0; i < numHyps; i++)
                {
                    double lp = prev[i].LogProb;
                    int rowOffset = i * vocab;
                    for (int k = 0; k < vocab; k++)
                    {
                        logit[rowOffset + k] = (float)(logit[rowOffset + k] + lp);
                    }
                }

                // 8) 在 numHyps * vocab 个候选中取 topK
                int[] topkIdx = TopkIndex(logit, _maxActivePaths);

                // 9) 扩展 topK 候选并写入 cur
                for (int kk = 0; kk < topkIdx.Length; kk++)
                {
                    int globalIdx = topkIdx[kk];
                    int hypIdx = globalIdx / vocab;
                    int newToken = globalIdx % vocab;

                    Hypothesis newHyp = prev[hypIdx].Clone();
                    float contextScore = 0f;
                    ContextState? ctxState = newHyp.ContextState ?? contextGraph.Root;

                    bool isBlank = (newToken == BlankId) || (newToken == _unkId);
                    if (!isBlank)
                    {
                        newHyp.Ys.Add(newToken);
                        newHyp.Timestamps.Add(t + result.FrameOffset);
                        newHyp.YsProbs.Add(MathF.Exp(acousticLogprobs[hypIdx * vocab + newToken]));
                        newHyp.NumTrailingBlanks = 0;

                        var (s, newState, _) = contextGraph.ForwardOneStep(ctxState, newToken);
                        contextScore = s;
                        newHyp.ContextState = newState;

                        // 如果 context graph 把状态拉回了 root (token=-1), 说明 fail 链已经
                        // 让我们脱离了任何前缀, 应该清空 token 历史以便后续匹配从头开始。
                        if (newHyp.ContextState.Token == -1)
                        {
                            newHyp.Ys = new List<long>(blankSeq);
                            newHyp.Timestamps.Clear();
                            newHyp.YsProbs.Clear();
                        }
                    }
                    else
                    {
                        newHyp.NumTrailingBlanks++;
                    }
                    newHyp.LogProb = logit[globalIdx] + contextScore;
                    cur.Add(newHyp);
                }

                // 10) 检测当前帧是否已命中关键词
                // 原算法只看最佳假设, 但快速连续说话时关键词路径会被新词路径瞬间挤掉,
                // 导致 trailing blanks 永远凑不齐。EmitImmediately=true 时扫描所有 hyps,
                // 找到声学概率最高的命中并立即触发, 不等 trailing blanks。
                Hypothesis? bestMatched = null;
                ContextState? bestMatchedNode = null;
                float bestMatchedYsProb = 0f;
                if (_emitImmediately)
                {
                    foreach (var h in cur.Values)
                    {
                        var (m, node) = contextGraph.IsMatched(h.ContextState);
                        if (!m || node == null) continue;
                        float p = ComputeAvgYsProb(h, node.Level);
                        if (bestMatched == null || p > bestMatchedYsProb)
                        {
                            bestMatched = h;
                            bestMatchedNode = node;
                            bestMatchedYsProb = p;
                        }
                    }
                }
                else
                {
                    Hypothesis bestProb = cur.GetMostProbable();
                    var (m, node) = contextGraph.IsMatched(bestProb.ContextState);
                    if (m && node != null)
                    {
                        bestMatched = bestProb;
                        bestMatchedNode = node;
                        bestMatchedYsProb = ComputeAvgYsProb(bestProb, node.Level);
                    }
                }
                if (bestMatched != null && bestMatchedNode != null)
                {
                    Hypothesis best = bestMatched;
                    ContextState matchedNode = bestMatchedNode;
                    float ysProb = bestMatchedYsProb;
                    bool blanksOk = _emitImmediately || best.NumTrailingBlanks > _numTrailingBlanks;
                    bool probOk = ysProb >= matchedNode.AcThreshold;
                    if (!blanksOk || !probOk)
                    {
                        // 仅记录每个 phrase 一次, 避免同一 chunk 内重复刷屏
                        bool already = false;
                        for (int n = 0; n < result.NearMisses.Count; n++)
                            if (result.NearMisses[n].Phrase == matchedNode.Phrase) { already = true; break; }
                        if (!already)
                        {
                            // 把命中路径上最近 level 个 token id 抓出来给日志诊断用
                            int level = matchedNode.Level;
                            int ysFrom = best.Ys.Count - level;
                            if (ysFrom < 0) ysFrom = 0;
                            int heardLen = best.Ys.Count - ysFrom;
                            var heard = new long[heardLen];
                            for (int i = 0; i < heardLen; i++) heard[i] = best.Ys[ysFrom + i];

                            result.NearMisses.Add(new KeywordNearMiss(
                                matchedNode.Phrase, ysProb, matchedNode.AcThreshold,
                                best.NumTrailingBlanks, _numTrailingBlanks, heard));
                        }
                    }
                    if (blanksOk && probOk)
                    {
                        // 触发! 把最近 level 个 token 写出到 result, 然后重置 hyps
                        int level = matchedNode.Level;
                        result.Tokens.Clear();
                        result.Timestamps.Clear();
                        int ysFrom = best.Ys.Count - level;
                        if (ysFrom < 0) ysFrom = 0;
                        for (int i = ysFrom; i < best.Ys.Count; i++) result.Tokens.Add(best.Ys[i]);
                        int tsFrom = best.Timestamps.Count - level;
                        if (tsFrom < 0) tsFrom = 0;
                        for (int i = tsFrom; i < best.Timestamps.Count; i++) result.Timestamps.Add(best.Timestamps[i]);
                        result.Keyword = matchedNode.Phrase;
                        result.KeywordProb = ysProb;

                        // 重置 cur 为只含一条初始 hyp (回到 root)
                        var resetHyp = new Hypothesis
                        {
                            Ys = new List<long>(blankSeq),
                            LogProb = 0,
                            ContextState = contextGraph.Root,
                        };
                        cur = new Hypotheses();
                        cur.Add(resetHyp);
                    }
                }
            } // for t

            // 把 cur 写回 result.Hyps; 更新 num_trailing_blanks 和 frame_offset
            result.Hyps = cur;
            Hypothesis finalBest = cur.GetMostProbable();
            result.NumTrailingBlanks = finalBest.NumTrailingBlanks;
            result.FrameOffset += numFrames;
        }

        // ----------------- helpers -----------------
        /// <summary>计算 hyp 最近 level 个 token 的平均声学概率 (与原算法一致)。</summary>
        private static float ComputeAvgYsProb(Hypothesis h, int level)
        {
            int from = h.YsProbs.Count - level;
            if (from < 0) from = 0;
            int count = h.YsProbs.Count - from;
            if (count <= 0) return 0f;
            float sum = 0f;
            for (int i = from; i < h.YsProbs.Count; i++) sum += h.YsProbs[i];
            return sum / count;
        }

        private static void LogSoftmax(float[] data, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
            {
                int offset = r * cols;
                float max = float.NegativeInfinity;
                for (int c = 0; c < cols; c++) if (data[offset + c] > max) max = data[offset + c];
                double sum = 0.0;
                for (int c = 0; c < cols; c++) sum += Math.Exp(data[offset + c] - max);
                float lse = max + (float)Math.Log(sum);
                for (int c = 0; c < cols; c++) data[offset + c] -= lse;
            }
        }

        /// <summary>
        /// 返回 data 中 top-K 元素的 索引 (按 value 降序)。K 通常远小于 N (beam ≤ 8 vs vocab*beam ~万级),
        /// 因此用 O(N + K log K) 的部分排序而不是完整排序, 避免每帧扫描的 N log N 开销。
        /// </summary>
        private static int[] TopkIndex(float[] data, int k)
        {
            int n = data.Length;
            if (k > n) k = n;
            if (k <= 0) return Array.Empty<int>();

            // 维护一个大小为 K 的最小堆 (按 value 升序), 堆顶是当前 top-K 中最差的元素。
            // 任意时刻只有 K 个候选, O(N log K) 整体。
            int[] heap = new int[k];          // 存索引
            for (int i = 0; i < k; i++) heap[i] = i;
            // 初始化堆 (建堆 O(K))
            for (int i = k / 2 - 1; i >= 0; i--) SiftDown(heap, k, i, data);
            for (int i = k; i < n; i++)
            {
                if (data[i] > data[heap[0]])
                {
                    heap[0] = i;
                    SiftDown(heap, k, 0, data);
                }
            }
            // 堆是 minheap, 但我们要返回降序 → 把 heap 排序为降序
            Array.Sort(heap, (a, b) => data[b].CompareTo(data[a]));
            return heap;
        }

        private static void SiftDown(int[] heap, int size, int i, float[] data)
        {
            while (true)
            {
                int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
                if (l < size && data[heap[l]] < data[heap[smallest]]) smallest = l;
                if (r < size && data[heap[r]] < data[heap[smallest]]) smallest = r;
                if (smallest == i) break;
                (heap[i], heap[smallest]) = (heap[smallest], heap[i]);
                i = smallest;
            }
        }
    }
}
