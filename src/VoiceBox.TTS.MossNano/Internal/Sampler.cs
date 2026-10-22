namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 与 ort_cpu_runtime.py 中的采样工具 1:1 对应。
    /// 高频路径接受 <see cref="SamplerWorkspace"/> 复用 buffer, 避免每帧每通道的堆分配。
    /// </summary>
    internal static class Sampler
    {
        public static int ArgMax(ReadOnlySpan<float> values)
        {
            int best = 0;
            float bestValue = float.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > bestValue)
                {
                    bestValue = values[i];
                    best = i;
                }
            }
            return best;
        }

        /// <summary>
        /// 对 logits 在指定 token id 集合上施加重复惩罚 (in-place)。
        /// 与 Python: `result[token] = result[token] * penalty if result[token] < 0 else result[token] / penalty`
        /// </summary>
        public static void ApplyRepetitionPenalty(Span<float> values, IReadOnlyCollection<int> previousTokenSet, float repetitionPenalty)
        {
            if (previousTokenSet.Count == 0 || repetitionPenalty == 1f) return;
            foreach (int tokenId in previousTokenSet)
            {
                if ((uint)tokenId >= (uint)values.Length) continue;
                float v = values[tokenId];
                values[tokenId] = v < 0 ? v * repetitionPenalty : v / repetitionPenalty;
            }
        }

        /// <summary>
        /// argmax 同时施加重复惩罚 (不修改原数组)。
        /// </summary>
        public static int ArgMaxWithRepetitionPenalty(ReadOnlySpan<float> values, IReadOnlyCollection<int> previousTokenSet, float repetitionPenalty)
        {
            int best = 0;
            float bestValue = float.NegativeInfinity;
            bool applyPenalty = previousTokenSet.Count > 0 && repetitionPenalty != 1f;

            HashSet<int>? penaltySet = applyPenalty
                ? (previousTokenSet as HashSet<int>) ?? new HashSet<int>(previousTokenSet)
                : null;

            for (int i = 0; i < values.Length; i++)
            {
                float score = values[i];
                if (penaltySet is not null && penaltySet.Contains(i))
                    score = score < 0 ? score * repetitionPenalty : score / repetitionPenalty;
                if (score > bestValue)
                {
                    bestValue = score;
                    best = i;
                }
            }
            return best;
        }

        /// <summary>
        /// 数值稳定版 softmax。
        /// </summary>
        public static void Softmax(Span<float> values)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
                if (values[i] > max) max = values[i];

            double sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double e = Math.Exp(values[i] - max);
                values[i] = (float)e;
                sum += e;
            }
            if (sum <= 0) sum = 1;
            float inv = (float)(1.0 / sum);
            for (int i = 0; i < values.Length; i++)
                values[i] *= inv;
        }

        /// <summary>
        /// 与 Python `_sample_from_scores` 1:1 对齐: top-k → top-p → 抽样。
        /// 输入 values 会被原地修改 (除以 temperature, 截断为 -Inf)。
        /// 工作 buffer 由 <paramref name="ws"/> 提供, 避免每次堆分配。
        /// </summary>
        public static int SampleFromScores(
            Span<float> values,
            bool doSample,
            float temperature,
            int topK,
            float topP,
            Random rng,
            SamplerWorkspace ws)
        {
            if (!doSample) return ArgMax(values);
            if (temperature <= 0f)
                throw new ArgumentOutOfRangeException(nameof(temperature), "temperature 必须为正数。");

            int n = values.Length;
            ws.Ensure(n);

            // / temperature
            float invT = 1f / temperature;
            for (int i = 0; i < n; i++) values[i] *= invT;

            // top-k 截断: 用 quickselect 找第 topK 大的阈值, O(n) 平均, 不做完整排序。
            if (topK > 0 && topK < n)
            {
                Span<float> copy = ws.TopKBuffer.AsSpan(0, n);
                values.CopyTo(copy);
                // 第 k 大 = 升序索引 (n - topK), 即截断阈值。
                float threshold = QuickSelectKthSmallest(copy, n - topK);
                for (int i = 0; i < n; i++)
                    if (values[i] < threshold) values[i] = float.NegativeInfinity;
            }

            // top-p 截断: 按分数降序排序索引, 累加 softmax 概率。
            if (topP > 0f && topP < 1f)
            {
                var indices = ws.Indices.AsSpan(0, n);
                var sortKeys = ws.SortKeys.AsSpan(0, n);
                for (int i = 0; i < n; i++)
                {
                    indices[i] = i;
                    // 取负数实现降序排序 (Span.Sort 仅升序), -Inf 取负为 +Inf 自然排到末尾。
                    sortKeys[i] = float.IsNegativeInfinity(values[i]) ? float.PositiveInfinity : -values[i];
                }
                sortKeys.Sort(indices);

                float maxScore = values[indices[0]];
                var probs = ws.Probs.AsSpan(0, n);
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    float v = values[indices[i]];
                    double e = float.IsNegativeInfinity(v) ? 0.0 : Math.Exp(v - maxScore);
                    probs[i] = e;
                    sum += e;
                }
                if (sum <= 0) sum = 1;
                double invSum = 1.0 / sum;
                for (int i = 0; i < n; i++) probs[i] *= invSum;

                // Python 行为: cumulative > topP 标记移除, 然后整体右移 1 位 (永远保留概率最高的那个)。
                // 与其用 bool 数组并右移, 这里直接计算保留前缀长度: 第一个累计 > topP 的 i, 保留 [0..i] (含 i)。
                double cumulative = 0;
                int keepCount = n; // 默认全保留
                for (int i = 0; i < n; i++)
                {
                    cumulative += probs[i];
                    if (cumulative > topP) { keepCount = i + 1; break; }
                }
                if (keepCount < n)
                {
                    // 把 keepCount..n-1 对应的原 index 置 -Inf。
                    for (int i = keepCount; i < n; i++)
                        values[indices[i]] = float.NegativeInfinity;
                }
            }

            // 最终 softmax + 抽样
            float vMax = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
                if (values[i] > vMax) vMax = values[i];

            var expVals = ws.ExpVals.AsSpan(0, n);
            double total = 0;
            for (int i = 0; i < n; i++)
            {
                double e = float.IsNegativeInfinity(values[i]) ? 0.0 : Math.Exp(values[i] - vMax);
                expVals[i] = e;
                total += e;
            }
            if (total <= 0) return ArgMax(values);

            double u = rng.NextDouble() * total;
            for (int i = 0; i < n; i++)
            {
                u -= expVals[i];
                if (u <= 0) return i;
            }
            return ArgMax(values);
        }

        /// <summary>
        /// 在 [audio_assistant_slot_token_id, audio_end_token_id] 两候选上做受温度/top-k/top-p 控制的采样。
        /// 候选数 =2, 不需要 workspace。
        /// </summary>
        public static int SampleAssistantTextToken(
            ReadOnlySpan<float> textLogits,
            int audioAssistantSlotTokenId,
            int audioEndTokenId,
            bool doSample,
            float textTemperature,
            int textTopK,
            float textTopP,
            Random rng,
            SamplerWorkspace ws)
        {
            Span<int> candidateIds = stackalloc int[2] { audioAssistantSlotTokenId, audioEndTokenId };
            Span<float> candidateScores = stackalloc float[2]
            {
                textLogits[audioAssistantSlotTokenId],
                textLogits[audioEndTokenId],
            };
            int sampledIndex = SampleFromScores(
                candidateScores,
                doSample,
                textTemperature,
                Math.Min(textTopK, candidateScores.Length),
                textTopP,
                rng,
                ws);
            return candidateIds[sampledIndex];
        }

        /// <summary>
        /// 与 Python `_sample_audio_token` 1:1 对齐。
        /// </summary>
        public static int SampleAudioToken(
            Span<float> audioLogits,
            IReadOnlyCollection<int> previousTokenSet,
            bool doSample,
            float audioTemperature,
            int audioTopK,
            float audioTopP,
            float audioRepetitionPenalty,
            Random rng,
            SamplerWorkspace ws)
        {
            if (!doSample)
                return ArgMaxWithRepetitionPenalty(audioLogits, previousTokenSet, audioRepetitionPenalty);

            ApplyRepetitionPenalty(audioLogits, previousTokenSet, audioRepetitionPenalty);
            return SampleFromScores(audioLogits, true, audioTemperature, audioTopK, audioTopP, rng, ws);
        }

        // ---------------- Quickselect ----------------

        /// <summary>
        /// 在 <paramref name="arr"/> 上原地 quickselect, 返回升序第 k 小的值。
        /// Hoare partition 三路逻辑, 平均 O(n)。
        /// </summary>
        private static float QuickSelectKthSmallest(Span<float> arr, int k)
        {
            int lo = 0, hi = arr.Length - 1;
            while (lo < hi)
            {
                int p = Partition(arr, lo, hi);
                if (p == k) return arr[p];
                if (p < k) lo = p + 1;
                else hi = p - 1;
            }
            return arr[lo];
        }

        private static int Partition(Span<float> arr, int lo, int hi)
        {
            // 取中点作 pivot, 减少有序输入 O(n^2) 概率。
            int mid = lo + ((hi - lo) >> 1);
            float pivot = arr[mid];
            (arr[mid], arr[hi]) = (arr[hi], arr[mid]);
            int store = lo;
            for (int i = lo; i < hi; i++)
            {
                if (arr[i] < pivot)
                {
                    (arr[store], arr[i]) = (arr[i], arr[store]);
                    store++;
                }
            }
            (arr[store], arr[hi]) = (arr[hi], arr[store]);
            return store;
        }
    }

    /// <summary>
    /// <see cref="Sampler"/> 主路径的复用 buffer。每个推理实例持有一份, 避免每帧每通道分配。
    /// 不是线程安全的; 上层已通过 SemaphoreSlim 串行化。
    /// Probs/ExpVals 用 double 与 Python `_sample_from_scores` 数值一致 (sampling 对累计概率精度敏感)。
    /// </summary>
    internal sealed class SamplerWorkspace
    {
        public float[] TopKBuffer = Array.Empty<float>();
        public int[] Indices = Array.Empty<int>();
        public float[] SortKeys = Array.Empty<float>();
        public double[] Probs = Array.Empty<double>();
        public double[] ExpVals = Array.Empty<double>();

        public void Ensure(int n)
        {
            if (TopKBuffer.Length < n) TopKBuffer = new float[n];
            if (Indices.Length < n) Indices = new int[n];
            if (SortKeys.Length < n) SortKeys = new float[n];
            if (Probs.Length < n) Probs = new double[n];
            if (ExpVals.Length < n) ExpVals = new double[n];
        }
    }
}
