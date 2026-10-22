namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 长文本按 token 预算切分, 与 Python OnnxTtsRuntime.split_voice_clone_text 1:1 对齐。
    /// 需外部注入 countTokens 回调 (SentencePiece tokenizer 的 EncodeToIds)。
    /// </summary>
    internal static class TextChunker
    {
        private static readonly HashSet<char> SentenceEndPunctuation = new("。！？；;.!?");
        private static readonly HashSet<char> ClauseSplitPunctuation = new(",，、；;：:");
        // closing punctuation: 各种右括号 / 右引号 (含全角)。用 codepoint 拼装避免源文件转义出错。
        private static readonly HashSet<char> ClosingPunctuation = new(new[]
        {
            '"', '\'',
            '”', '’', // ”’
            ')', ']', '}',
            '）', '】', '》', '」', '』', // ）】》」』
        });
        private static readonly HashSet<char> PreferredBoundaryChars;

        static TextChunker()
        {
            PreferredBoundaryChars = new HashSet<char>(ClauseSplitPunctuation);
            PreferredBoundaryChars.UnionWith(SentenceEndPunctuation);
            PreferredBoundaryChars.Add(' ');
        }

        public static bool ContainsCjk(string text)
        {
            foreach (char c in text)
            {
                if (c is >= '一' and <= '鿿'
                    or >= '㐀' and <= '䶿'
                    or >= '぀' and <= 'ヿ'
                    or >= '가' and <= '힯') return true;
            }
            return false;
        }

        /// <summary>
        /// 与 Python `_prepare_text_for_sentence_chunking` 1:1 对齐。
        /// 规整空白; CJK 文本补尾句号; 西文首字母大写并补尾句号; 短西文文本前补空格 (模型行为保护)。
        /// </summary>
        public static string PrepareTextForSentenceChunking(string text)
        {
            var normalized = (text ?? string.Empty).Trim().Replace('\r', ' ').Replace('\n', ' ');
            while (normalized.Contains("  ")) normalized = normalized.Replace("  ", " ");
            if (string.IsNullOrEmpty(normalized)) throw new ArgumentException("文本不能为空。", nameof(text));

            if (ContainsCjk(normalized))
            {
                if (!SentenceEndPunctuation.Contains(normalized[^1]))
                    normalized += "。";
                return normalized;
            }

            // 西文路径
            if (char.IsLower(normalized[0]))
                normalized = char.ToUpperInvariant(normalized[0]) + normalized[1..];
            if (char.IsLetterOrDigit(normalized[^1]))
                normalized += ".";
            int wordCount = 0;
            foreach (var c in normalized) { if (c == ' ') wordCount++; }
            wordCount++; // token ≈ word+1
            if (wordCount < 5)
                normalized = "        " + normalized;
            return normalized;
        }

        /// <summary>
        /// 按标点集合切分文本。
        /// 与 Python `_split_text_by_punctuation` 1:1 对齐 (含 closing punctuation lookahead)。
        /// </summary>
        public static List<string> SplitTextByPunctuation(string text, HashSet<char> punctuation)
        {
            var sentences = new List<string>();
            var current = new List<char>();
            int index = 0;
            while (index < text.Length)
            {
                char ch = text[index];
                current.Add(ch);
                if (punctuation.Contains(ch))
                {
                    int lookahead = index + 1;
                    while (lookahead < text.Length && ClosingPunctuation.Contains(text[lookahead]))
                    {
                        current.Add(text[lookahead]);
                        lookahead++;
                    }
                    string sentence = new string(current.ToArray()).Trim();
                    if (sentence.Length > 0) sentences.Add(sentence);
                    current.Clear();
                    while (lookahead < text.Length && char.IsWhiteSpace(text[lookahead]))
                        lookahead++;
                    index = lookahead;
                    continue;
                }
                index++;
            }
            string tail = new string(current.ToArray()).Trim();
            if (tail.Length > 0) sentences.Add(tail);
            return sentences;
        }

        public static string JoinSentenceParts(string left, string right)
        {
            if (string.IsNullOrEmpty(left)) return right;
            if (string.IsNullOrEmpty(right)) return left;
            if (ContainsCjk(left) || ContainsCjk(right)) return left + right;
            return left + " " + right;
        }

        /// <summary>
        /// 使用二分查找切分过长文本, 使每段 ≤ maxTokens。
        /// 与 Python `split_text_by_token_budget` 1:1 对齐。
        /// </summary>
        public static List<string> SplitTextByTokenBudget(string text, int maxTokens, Func<string, int> countTokens)
        {
            string remaining = (text ?? string.Empty).Trim();
            if (remaining.Length == 0) return new();
            var pieces = new List<string>();
            while (remaining.Length > 0)
            {
                if (countTokens(remaining) <= maxTokens)
                {
                    pieces.Add(remaining);
                    break;
                }
                int low = 1, high = remaining.Length, bestPrefixLength = 1;
                while (low <= high)
                {
                    int mid = (low + high) / 2;
                    string candidate = remaining[..mid].Trim();
                    if (candidate.Length == 0) { low = mid + 1; continue; }
                    if (countTokens(candidate) <= maxTokens) { bestPrefixLength = mid; low = mid + 1; }
                    else { high = mid - 1; }
                }
                string prefix = remaining[..bestPrefixLength];
                int cutIndex = bestPrefixLength;
                int scanMin = Math.Max(-1, prefix.Length - 25);
                for (int scanIdx = prefix.Length - 1; scanIdx > scanMin; scanIdx--)
                {
                    if (PreferredBoundaryChars.Contains(prefix[scanIdx]))
                    {
                        cutIndex = scanIdx + 1;
                        break;
                    }
                }
                string piece = remaining[..cutIndex].Trim();
                if (piece.Length == 0) { piece = remaining[..bestPrefixLength].Trim(); cutIndex = bestPrefixLength; }
                pieces.Add(piece);
                remaining = remaining[cutIndex..].Trim();
            }
            return pieces;
        }

        /// <summary>
        /// 完整的 voice clone 文本切分流程。
        /// 与 Python `OnnxTtsRuntime.split_voice_clone_text` 1:1 对齐。
        /// </summary>
        public static List<string> SplitVoiceCloneText(string text, int maxTokens, Func<string, int> countTokens)
        {
            string normalized = (text ?? string.Empty).Trim();
            if (normalized.Length == 0) return new();
            int safeMaxTokens = Math.Max(1, maxTokens);
            string prepared = PrepareTextForSentenceChunking(normalized);

            var sentenceCandidates = SplitTextByPunctuation(prepared, SentenceEndPunctuation);
            if (sentenceCandidates.Count == 0) sentenceCandidates.Add(prepared.Trim());

            var slices = new List<(int tokenCount, string text)>();
            foreach (string sentenceText in sentenceCandidates)
            {
                string ns = sentenceText.Trim();
                if (ns.Length == 0) continue;
                int stc = countTokens(ns);
                if (stc <= safeMaxTokens) { slices.Add((stc, ns)); continue; }
                var clauses = SplitTextByPunctuation(ns, ClauseSplitPunctuation);
                if (clauses.Count <= 1) clauses = new List<string> { ns };
                foreach (string clauseText in clauses)
                {
                    string nc = clauseText.Trim();
                    if (nc.Length == 0) continue;
                    int ctc = countTokens(nc);
                    if (ctc <= safeMaxTokens) { slices.Add((ctc, nc)); continue; }
                    foreach (string piece in SplitTextByTokenBudget(nc, safeMaxTokens, countTokens))
                    {
                        string np = piece.Trim();
                        if (np.Length > 0) slices.Add((countTokens(np), np));
                    }
                }
            }

            var chunks = new List<string>();
            string currentChunk = string.Empty;
            int currentChunkTokenCount = 0;
            foreach (var (stc, st) in slices)
            {
                if (currentChunk.Length == 0) { currentChunk = st; currentChunkTokenCount = stc; continue; }
                if (currentChunkTokenCount + stc > safeMaxTokens)
                {
                    chunks.Add(currentChunk.Trim());
                    currentChunk = st;
                    currentChunkTokenCount = stc;
                }
                else
                {
                    currentChunk = JoinSentenceParts(currentChunk, st);
                    currentChunkTokenCount = countTokens(currentChunk);
                }
            }
            if (currentChunk.Length > 0) chunks.Add(currentChunk.Trim());
            // 与 Python OnnxTtsRuntime.split_voice_clone_text 的差异:
            // Python 在单 chunk 情况下回退到 normalized_text (原始输入, 无补全标点)。
            // 实测这会让模型对 "你好" 这种无标点短输入触发明显的复读 — 模型无法判断 "句子结束".
            // 始终回退到 prepared（已补全句末标点）版本。
            return chunks.Count > 1 ? chunks : new List<string> { prepared.Trim() };
        }

        /// <summary>
        /// 估算 chunk 间停顿秒数。
        /// </summary>
        public static float EstimateInterChunkPauseSeconds(string textChunk)
        {
            int wordCount = 0;
            bool inWord = false;
            foreach (char c in textChunk)
            {
                if (char.IsWhiteSpace(c)) { inWord = false; }
                else { if (!inWord) wordCount++; inWord = true; }
            }
            return wordCount <= 4 ? 0.40f : 0.24f;
        }
    }
}
