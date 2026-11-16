namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 单条波束搜索假设, 对应 sherpa-onnx 的 Hypothesis。
    /// </summary>
    internal sealed class Hypothesis
    {
        /// <summary>已预测的 token id 序列。</summary>
        public List<long> Ys { get; set; } = new();

        /// <summary>每个 token 解码时所在的子采样后帧索引 (与 Ys 长度一致)。</summary>
        public List<int> Timestamps { get; set; } = new();

        /// <summary>每个 token 的声学概率 (与 Ys 长度一致, 由 softmax 后取 exp 得到)。</summary>
        public List<float> YsProbs { get; set; } = new();

        /// <summary>累计 log 概率 (仅声学, 不含 LM)。</summary>
        public double LogProb { get; set; }

        /// <summary>本假设处于上下文图 (ContextGraph) 中的当前节点。</summary>
        public ContextState? ContextState { get; set; }

        /// <summary>连续解码出 blank/unk 的帧数, 用于触发条件判定。</summary>
        public int NumTrailingBlanks { get; set; }

        /// <summary>用作 Hypotheses 集合中的去重 key, 仅基于 token 序列。</summary>
        public string Key
        {
            get
            {
                if (Ys.Count == 0) return string.Empty;
                var sb = new System.Text.StringBuilder(Ys.Count * 4);
                for (int i = 0; i < Ys.Count; i++)
                {
                    if (i > 0) sb.Append('-');
                    sb.Append(Ys[i]);
                }
                return sb.ToString();
            }
        }

        public Hypothesis Clone()
        {
            return new Hypothesis
            {
                Ys = new List<long>(Ys),
                Timestamps = new List<int>(Timestamps),
                YsProbs = new List<float>(YsProbs),
                LogProb = LogProb,
                ContextState = ContextState,
                NumTrailingBlanks = NumTrailingBlanks,
            };
        }
    }

    /// <summary>
    /// 多条假设的集合 (key -> hypothesis), 提供合并 (log-add) / topK 等操作。
    /// </summary>
    internal sealed class Hypotheses
    {
        private readonly Dictionary<string, Hypothesis> _dict = new(StringComparer.Ordinal);

        public int Size => _dict.Count;

        public Hypotheses() { }

        public Hypotheses(IEnumerable<Hypothesis> hyps)
        {
            foreach (var h in hyps) _dict[h.Key] = h;
        }

        public IEnumerable<Hypothesis> Values => _dict.Values;

        /// <summary>
        /// 加入一条假设; 若已存在相同 token 序列, 则对 log_prob 做 log-sum-exp 合并;
        /// 保留先插入的那条假设 (与 sherpa-onnx C++ Hypotheses::Add 一致)。
        /// </summary>
        public void Add(Hypothesis hyp)
        {
            if (_dict.TryGetValue(hyp.Key, out var existed))
            {
                existed.LogProb = LogAdd(existed.LogProb, hyp.LogProb);
            }
            else
            {
                _dict[hyp.Key] = hyp;
            }
        }

        /// <summary>取 log_prob 最大的一条假设。</summary>
        public Hypothesis GetMostProbable()
        {
            Hypothesis? best = null;
            foreach (var h in _dict.Values)
            {
                if (best == null || h.LogProb > best.LogProb) best = h;
            }
            return best ?? throw new InvalidOperationException("Hypotheses is empty.");
        }

        public void Clear() => _dict.Clear();

        // ln(double-epsilon)
        private const double MinLogDiff = -36.04365338911715;

        private static double LogAdd(double x, double y)
        {
            double diff;
            if (x < y) { diff = x - y; x = y; }
            else { diff = y - x; }
            if (diff >= MinLogDiff)
            {
                // diff ≤ 0, exp(diff) ≤ 1, 因此 log(1 + exp(diff)) 是数值稳定的。
                return x + Math.Log(1.0 + Math.Exp(diff));
            }
            return x;
        }
    }
}
