namespace VoiceBox.Wakeup.FsmnKws.Internal
{
    /// <summary>
    /// 帧同步 CTC prefix beam search 关键词检测器
    /// (算法与 funasr.utils.kws_utils.KwsCtcPrefixDecoder 对齐, 为无限流场景增加:
    /// 过期假设剪枝 / 概率重归一化 / 关键词最大时长约束)。
    /// <para>
    /// 每帧只保留 ScoreBeam 个且后验 &gt; TokenProbFloor 且属于"关键词 token ∪ blank"的候选,
    /// 维护至多 PathBeam 条前缀假设; 当任一假设的前缀包含某关键词 token 序列、时间跨度合法且
    /// 分数 (= sqrt(各 token 峰值后验之积)) 达到阈值时上报命中。
    /// </para>
    /// </summary>
    internal sealed class CtcKeywordDecoder
    {
        internal readonly record struct Detection(
            string Keyword, float Score, float Threshold, long StartFrame, long EndFrame);

        internal readonly record struct KeywordEntry(string Keyword, int[] TokenIds, float Threshold);

        private readonly record struct TokenNode(int Token, long Frame, double Prob);

        private sealed class Hypothesis
        {
            public int[] Prefix = Array.Empty<int>();
            public double Pb;   // 以 blank 结尾的概率
            public double Pnb;  // 以非 blank 结尾的概率
            public List<TokenNode> Nodes = new();
        }

        private const int BlankId = 0;

        private readonly KeywordEntry[] _keywords;
        private readonly bool[] _tokenMask;
        private readonly int _scoreBeam;
        private readonly int _pathBeam;
        private readonly double _probFloor;
        private readonly long _maxSpanFrames;

        private List<Hypothesis> _hyps = new();
        private long _frame;

        public CtcKeywordDecoder(
            IReadOnlyList<KeywordEntry> keywords,
            int scoreBeam, int pathBeam, float tokenProbFloor, long maxSpanFrames)
        {
            if (keywords.Count == 0) throw new ArgumentException("at least one keyword required.", nameof(keywords));
            _keywords = keywords.ToArray();
            int maxTokenId = BlankId;
            foreach (var kw in _keywords)
                foreach (int id in kw.TokenIds)
                    if (id >= 0) maxTokenId = Math.Max(maxTokenId, id);
            _tokenMask = new bool[maxTokenId + 1];
            _tokenMask[BlankId] = true;
            foreach (var kw in _keywords)
                foreach (int id in kw.TokenIds)
                    if (id >= 0) _tokenMask[id] = true;
            _scoreBeam = scoreBeam;
            _pathBeam = pathBeam;
            _probFloor = tokenProbFloor;
            _maxSpanFrames = maxSpanFrames;
            Reset();
        }

        public void Reset()
        {
            _hyps = new List<Hypothesis> { new() { Pb = 1.0, Pnb = 0.0 } };
            // 注意: _frame 不清零, 保持单调, 供调用方做时间换算
        }

        /// <summary>完全重置 (含帧计数), 用于 Start()。</summary>
        public void ResetAll()
        {
            _frame = 0;
            Reset();
        }

        /// <summary>
        /// 更新全部关键词的触发阈值。关键词 token 图和搜索状态无需重建。
        /// </summary>
        public void SetThreshold(float threshold)
        {
            for (int i = 0; i < _keywords.Length; i++)
                _keywords[i] = _keywords[i] with { Threshold = threshold };
        }

        /// <summary>处理一帧后验概率 (长度 = vocab), 返回命中或 null。命中后调用方应 <see cref="Reset"/>。</summary>
        public Detection? Step(ReadOnlySpan<float> probs)
        {
            long t = _frame++;

            Span<int> topIdx = stackalloc int[_scoreBeam];
            Span<float> topProb = stackalloc float[_scoreBeam];
            TopK(probs, topIdx, topProb);

            // 过滤: 后验 > floor 且属于关键词 token 集 (含 blank)
            Span<int> candIdx = stackalloc int[_scoreBeam];
            Span<float> candProb = stackalloc float[_scoreBeam];
            int candCount = 0;
            for (int i = 0; i < _scoreBeam; i++)
            {
                int token = topIdx[i];
                if (topProb[i] > _probFloor
                    && (uint)token < (uint)_tokenMask.Length
                    && _tokenMask[token])
                {
                    candIdx[candCount] = token;
                    candProb[candCount] = topProb[i];
                    candCount++;
                }
            }

            if (candCount == 1 && candIdx[0] == BlankId)
            {
                // 快路径: 候选只有 blank (静音/非关键词语音, 绝大多数帧)。
                // 每条假设映射到自身前缀, 无合并/无新前缀, 等价于完整扩展:
                // Pb' = (Pb + Pnb) * p(blank), Pnb' = 0, Prefix/Nodes 不变。
                double pBlank = candProb[0];
                foreach (var hyp in _hyps)
                {
                    hyp.Pb = (hyp.Pb + hyp.Pnb) * pBlank;
                    hyp.Pnb = 0.0;
                }
                if (_hyps.Count > _pathBeam)
                {
                    _hyps.Sort(static (a, b) => (b.Pb + b.Pnb).CompareTo(a.Pb + a.Pnb));
                    _hyps.RemoveRange(_pathBeam, _hyps.Count - _pathBeam);
                }
            }
            else if (candCount > 0)
            {
                ExpandHypotheses(candIdx[..candCount], candProb[..candCount], t);
            }

            PruneAndRenormalize(t);
            return MatchKeywords(t);
        }

        private void ExpandHypotheses(ReadOnlySpan<int> tokens, ReadOnlySpan<float> probs, long t)
        {
            var next = new Dictionary<PrefixKey, Hypothesis>(_hyps.Count * (tokens.Length + 1));

            for (int ci = 0; ci < tokens.Length; ci++)
            {
                int s = tokens[ci];
                double ps = probs[ci];

                foreach (var hyp in _hyps)
                {
                    int last = hyp.Prefix.Length > 0 ? hyp.Prefix[^1] : -1;
                    if (s == BlankId)
                    {
                        var n = GetOrAdd(next, hyp.Prefix);
                        n.Pb += (hyp.Pb + hyp.Pnb) * ps;
                        n.Nodes = new List<TokenNode>(hyp.Nodes);
                    }
                    else if (s == last)
                    {
                        // *ss -> *s (重复帧, 留在原前缀)
                        if (hyp.Pnb > 1e-6)
                        {
                            var n = GetOrAdd(next, hyp.Prefix);
                            var nodes = new List<TokenNode>(hyp.Nodes);
                            if (ps > nodes[^1].Prob) nodes[^1] = new TokenNode(s, t, ps);
                            n.Pnb += hyp.Pnb * ps;
                            n.Nodes = nodes;
                        }
                        // *s-s -> *ss (经 blank 后再次出现, 扩展前缀)
                        if (hyp.Pb > 1e-6)
                        {
                            var n = GetOrAdd(next, Append(hyp.Prefix, s));
                            var nodes = new List<TokenNode>(hyp.Nodes) { new(s, t, ps) };
                            n.Pnb += hyp.Pb * ps;
                            n.Nodes = nodes;
                        }
                    }
                    else
                    {
                        var n = GetOrAdd(next, Append(hyp.Prefix, s));
                        if (n.Nodes.Count > 0)
                        {
                            if (ps > n.Nodes[^1].Prob) n.Nodes[^1] = new TokenNode(s, t, ps);
                        }
                        else
                        {
                            n.Nodes = new List<TokenNode>(hyp.Nodes) { new(s, t, ps) };
                        }
                        n.Pnb += (hyp.Pb + hyp.Pnb) * ps;
                    }
                }
            }

            _hyps.Clear();
            _hyps.AddRange(next.Values);
            _hyps.Sort(static (a, b) => (b.Pb + b.Pnb).CompareTo(a.Pb + a.Pnb));
            if (_hyps.Count > _pathBeam)
                _hyps.RemoveRange(_pathBeam, _hyps.Count - _pathBeam);
        }

        private void PruneAndRenormalize(long t)
        {
            long minFrame = t - _maxSpanFrames;

            // 1) 丢弃"最后一个 token 已过期"的假设 (流式: 防止陈年前缀与新语音拼接),
            //    并裁掉仍活跃假设中过老的前缀 token。否则连续语音/噪声里只要关键词 token
            //    持续零星出现, Prefix/Nodes 会随运行时间无限增长。
            _hyps.RemoveAll(h => h.Nodes.Count > 0 && h.Nodes[^1].Frame < minFrame);
            foreach (var hyp in _hyps)
                TrimExpiredPrefix(hyp, minFrame);

            // 2) 确保空前缀假设存在 (作为兜底路径)
            double best = 0.0;
            bool hasEmpty = false;
            foreach (var h in _hyps)
            {
                best = Math.Max(best, h.Pb + h.Pnb);
                if (h.Prefix.Length == 0) hasEmpty = true;
            }
            if (!hasEmpty)
                _hyps.Add(new Hypothesis { Pb = best > 0 ? best : 1.0, Pnb = 0.0 });

            // 3) 防上溢/下溢: 长时间运行概率持续衰减, 周期性按最大值归一
            if (best > 0 && (best < 1e-10 || best > 1e10))
            {
                foreach (var h in _hyps)
                {
                    h.Pb /= best;
                    h.Pnb /= best;
                }
            }
        }

        private Detection? MatchKeywords(long t)
        {
            Detection? bestHit = null;
            foreach (var hyp in _hyps)
            {
                foreach (var kw in _keywords)
                {
                    int offset = IndexOfSubsequence(hyp.Prefix, kw.TokenIds, 0);
                    while (offset >= 0)
                    {
                        var first = hyp.Nodes[offset];
                        var last = hyp.Nodes[offset + kw.TokenIds.Length - 1];
                        if (last.Frame - first.Frame <= _maxSpanFrames)
                        {
                            double score = 1.0;
                            for (int i = offset; i < offset + kw.TokenIds.Length; i++)
                                score *= hyp.Nodes[i].Prob;
                            score = Math.Sqrt(score);

                            if (score >= kw.Threshold && (bestHit is null || score > bestHit.Value.Score))
                                bestHit = new Detection(kw.Keyword, (float)score, kw.Threshold, first.Frame, last.Frame);
                        }

                        offset = IndexOfSubsequence(hyp.Prefix, kw.TokenIds, offset + 1);
                    }
                }
            }
            return bestHit;
        }

        private static void TrimExpiredPrefix(Hypothesis hyp, long minFrame)
        {
            int removeCount = 0;
            while (removeCount < hyp.Nodes.Count && hyp.Nodes[removeCount].Frame < minFrame)
                removeCount++;

            if (removeCount == 0) return;

            hyp.Nodes.RemoveRange(0, removeCount);
            hyp.Prefix = hyp.Prefix.AsSpan(removeCount).ToArray();
        }

        private static int IndexOfSubsequence(int[] main, int[] sub, int start)
        {
            if (main.Length < sub.Length) return -1;
            for (int i = start; i <= main.Length - sub.Length; i++)
            {
                int j = 0;
                while (j < sub.Length && main[i + j] == sub[j]) j++;
                if (j == sub.Length) return i;
            }
            return -1;
        }

        private void TopK(ReadOnlySpan<float> probs, Span<int> idx, Span<float> val)
        {
            idx.Fill(-1);
            val.Fill(float.MinValue);
            for (int i = 0; i < probs.Length; i++)
            {
                float p = probs[i];
                if (p <= val[^1]) continue;
                int pos = val.Length - 1;
                while (pos > 0 && p > val[pos - 1]) pos--;
                for (int m = val.Length - 1; m > pos; m--)
                {
                    val[m] = val[m - 1];
                    idx[m] = idx[m - 1];
                }
                val[pos] = p;
                idx[pos] = i;
            }
        }

        private static int[] Append(int[] prefix, int token)
        {
            var result = new int[prefix.Length + 1];
            prefix.CopyTo(result, 0);
            result[^1] = token;
            return result;
        }

        private static Hypothesis GetOrAdd(Dictionary<PrefixKey, Hypothesis> dict, int[] prefix)
        {
            var key = new PrefixKey(prefix);
            if (!dict.TryGetValue(key, out var hyp))
            {
                hyp = new Hypothesis { Prefix = prefix };
                dict[key] = hyp;
            }
            return hyp;
        }

        private readonly struct PrefixKey : IEquatable<PrefixKey>
        {
            private readonly int[] _ids;
            private readonly int _hash;

            public PrefixKey(int[] ids)
            {
                _ids = ids;
                var hash = new HashCode();
                foreach (int id in ids) hash.Add(id);
                _hash = hash.ToHashCode();
            }

            public bool Equals(PrefixKey other) =>
                _hash == other._hash && _ids.AsSpan().SequenceEqual(other._ids);

            public override bool Equals(object? obj) => obj is PrefixKey other && Equals(other);

            public override int GetHashCode() => _hash;
        }
    }
}
