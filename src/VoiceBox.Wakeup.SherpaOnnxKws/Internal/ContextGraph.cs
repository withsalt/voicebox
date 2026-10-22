namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 上下文图 (Aho-Corasick) 中的一个节点。每个节点代表"已匹配到关键词的某个 token 前缀"。
    /// 对应 sherpa-onnx 的 ContextState 结构。
    /// </summary>
    internal sealed class ContextState
    {
        public int Token { get; set; } = -1;
        /// <summary>到达本节点单 token 增量 boost 分数。</summary>
        public float TokenScore { get; set; }
        /// <summary>从根到本节点的累计 boost 分数。</summary>
        public float NodeScore { get; set; }
        /// <summary>命中关键词时的输出分数 (= NodeScore 当 is_end 为 true)。</summary>
        public float OutputScore { get; set; }
        /// <summary>层级 (= 已匹配的 token 数, 根为 0)。</summary>
        public int Level { get; set; }
        /// <summary>命中后的声学触发阈值, 0 表示沿用全局阈值。</summary>
        public float AcThreshold { get; set; }
        /// <summary>是否为关键词结尾节点。</summary>
        public bool IsEnd { get; set; }
        /// <summary>命中关键词的原始字符串 (仅 IsEnd 节点上有意义)。</summary>
        public string Phrase { get; set; } = string.Empty;

        public Dictionary<int, ContextState> Next { get; } = new();
        public ContextState? Fail { get; set; }
        public ContextState? Output { get; set; }
    }

    /// <summary>
    /// 上下文图: 把所有关键词的 token id 序列构建成 Aho-Corasick 自动机,
    /// 在解码过程中提供 boost score 与命中判定。
    /// </summary>
    internal sealed class ContextGraph
    {
        private readonly float _globalScore;
        private float _globalThreshold;
        private readonly ContextState _root;

        public ContextState Root => _root;

        public ContextGraph(
            IReadOnlyList<int[]> tokenIds,
            float contextScore,
            float acThreshold,
            IReadOnlyList<float>? perKeywordScores = null,
            IReadOnlyList<string>? phrases = null,
            IReadOnlyList<float>? perKeywordThresholds = null)
        {
            _globalScore = contextScore;
            _globalThreshold = acThreshold;
            _root = new ContextState { Token = -1, Level = 0 };
            _root.Fail = _root;
            Build(tokenIds, perKeywordScores, phrases, perKeywordThresholds);
            FillFailOutput();
        }

        /// <summary>
        /// 更新图中全部关键词结尾节点的声学触发阈值。
        /// </summary>
        public void SetThreshold(float threshold)
        {
            _globalThreshold = threshold;

            var pending = new Stack<ContextState>();
            pending.Push(_root);
            while (pending.Count > 0)
            {
                var node = pending.Pop();
                foreach (var child in node.Next.Values)
                {
                    if (child.IsEnd) child.AcThreshold = threshold;
                    pending.Push(child);
                }
            }
        }

        private void Build(
            IReadOnlyList<int[]> tokenIds,
            IReadOnlyList<float>? scores,
            IReadOnlyList<string>? phrases,
            IReadOnlyList<float>? thresholds)
        {
            for (int i = 0; i < tokenIds.Count; i++)
            {
                int[] ids = tokenIds[i];
                float score = scores != null && scores.Count > i ? scores[i] : 0f;
                if (score == 0f) score = _globalScore;
                float threshold = thresholds != null && thresholds.Count > i ? thresholds[i] : 0f;
                if (threshold == 0f) threshold = _globalThreshold;
                string phrase = phrases != null && phrases.Count > i ? phrases[i] : string.Empty;

                ContextState node = _root;
                for (int j = 0; j < ids.Length; j++)
                {
                    int token = ids[j];
                    bool isEnd = (j == ids.Length - 1);

                    if (!node.Next.TryGetValue(token, out var child))
                    {
                        child = new ContextState
                        {
                            Token = token,
                            TokenScore = score,
                            NodeScore = node.NodeScore + score,
                            OutputScore = isEnd ? node.NodeScore + score : 0f,
                            Level = j + 1,
                            AcThreshold = isEnd ? threshold : 0f,
                            IsEnd = isEnd,
                            Phrase = isEnd ? phrase : string.Empty,
                        };
                        node.Next[token] = child;
                    }
                    else
                    {
                        // 同一 token 同时是多个关键词的某一位 — 取较大 token_score, 更新 node_score
                        float ts = Math.Max(score, child.TokenScore);
                        child.TokenScore = ts;
                        child.NodeScore = node.NodeScore + ts;
                        bool newEnd = isEnd || child.IsEnd;
                        child.OutputScore = newEnd ? child.NodeScore : 0f;
                        child.IsEnd = newEnd;
                        if (isEnd)
                        {
                            child.Phrase = phrase;
                            child.AcThreshold = threshold;
                        }
                    }
                    node = child;
                }
            }
        }

        private void FillFailOutput()
        {
            // BFS: 第一层 fail 指针都指向 root
            var queue = new Queue<ContextState>();
            foreach (var kv in _root.Next)
            {
                kv.Value.Fail = _root;
                queue.Enqueue(kv.Value);
            }
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                foreach (var kv in current.Next)
                {
                    int token = kv.Key;
                    var child = kv.Value;
                    // 计算 child.Fail
                    var fail = current.Fail!;
                    if (fail.Next.TryGetValue(token, out var failNext))
                    {
                        fail = failNext;
                    }
                    else
                    {
                        while (!fail.Next.ContainsKey(token))
                        {
                            fail = fail.Fail!;
                            if (fail.Token == -1) break; // root
                        }
                        if (fail.Next.TryGetValue(token, out var f2)) fail = f2;
                    }
                    child.Fail = fail;

                    // 沿 fail 链找最近的 IsEnd 作为 output
                    ContextState? output = fail;
                    while (output != null && !output.IsEnd)
                    {
                        output = output.Fail;
                        if (output != null && output.Token == -1) { output = null; break; }
                    }
                    child.Output = output;
                    child.OutputScore += output?.OutputScore ?? 0f;

                    queue.Enqueue(child);
                }
            }
        }

        /// <summary>
        /// 输入当前状态与 token, 返回 (score 增量, 新状态, 命中节点 (若有))。
        /// </summary>
        public (float Score, ContextState NewState, ContextState? MatchedNode) ForwardOneStep(
            ContextState state, int token)
        {
            ContextState node;
            float score;
            if (state.Next.TryGetValue(token, out var direct))
            {
                node = direct;
                score = node.TokenScore;
            }
            else
            {
                node = state.Fail!;
                while (!node.Next.ContainsKey(token))
                {
                    node = node.Fail!;
                    if (node.Token == -1) break; // root
                }
                if (node.Next.TryGetValue(token, out var next)) node = next;
                score = node.NodeScore - state.NodeScore;
            }

            ContextState? matched = node.IsEnd ? node : node.Output;
            return (score + node.OutputScore, node, matched);
        }

        /// <summary>
        /// 判断当前状态是否命中关键词。命中时返回 (true, 命中节点); 否则 (false, null)。
        /// </summary>
        public (bool Matched, ContextState? Node) IsMatched(ContextState? state)
        {
            if (state == null) return (false, null);
            if (state.IsEnd) return (true, state);
            if (state.Output != null) return (true, state.Output);
            return (false, null);
        }
    }
}
