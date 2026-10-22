namespace VoiceBox.TTS.Kokoro.Internal
{
    internal static class PhonemeSplitter
    {
        public const int MaxPhonemeLength = 510;

        // 尾巴比这个短就并入上一批，避免产生只含 1-2 个 phoneme 的寄生批次
        // (例如句末标点单独成批，会浪费一次 ONNX 推理)。
        private const int MinTailPhonemes = 6;

        private static readonly char[] PunctuationChars = ['.', ',', '!', '?', ';'];

        public static List<string> Split(string phonemes)
            => Split(phonemes, firstBatchBudget: 0);

        /// <summary>
        /// 把整段 phoneme 拆成多批，单批长度上限 <see cref="MaxPhonemeLength"/>。
        /// 当 <paramref name="firstBatchBudget"/> 在 (0, MaxPhonemeLength) 之间时，
        /// 第一批会被进一步限制在该预算内（在不超过预算的最近一个标点处切断），以降低首块延迟。
        /// </summary>
        public static List<string> Split(string phonemes, int firstBatchBudget)
        {
            if (string.IsNullOrEmpty(phonemes)) return [];

            bool useShortFirst = firstBatchBudget > 0
                && firstBatchBudget < MaxPhonemeLength
                && phonemes.Length > firstBatchBudget;

            if (!useShortFirst && phonemes.Length <= MaxPhonemeLength)
                return [phonemes];

            var parts = SplitByPunctuation(phonemes);
            var batches = new List<string>();
            var current = new System.Text.StringBuilder(MaxPhonemeLength);

            foreach (string part in parts)
            {
                string trimmed = part.Trim();
                if (trimmed.Length == 0) continue;

                int budget = (useShortFirst && batches.Count == 0) ? firstBatchBudget : MaxPhonemeLength;

                if (current.Length + trimmed.Length + 1 >= budget)
                {
                    if (current.Length > 0)
                    {
                        batches.Add(current.ToString().Trim());
                        current.Clear();
                    }
                    if (trimmed.Length > budget)
                    {
                        // 无标点的超长片段（典型: 中文长句, SplitSentences 已按逗号切过, 单句段内部
                        // 不再有 ASCII 标点）比模型上下文上限 (510) 还长。原先整体放入再被
                        // KokoroTokenizer.Tokenize 截断, 会导致末尾音素对应的音频永远不被合成。
                        // 这里硬切成不超过 budget 的多批, 仅牺牲切分位置质量, 保证音频完整。
                        AppendHardSplit(trimmed, batches, budget, MaxPhonemeLength);
                    }
                    else
                    {
                        // 极端情况下单段已经超过预算（首批没有合适标点切点），仍然把它放进去
                        current.Append(trimmed);
                    }
                }
                else
                {
                    if (trimmed.Length == 1 && PunctuationChars.Contains(trimmed[0]))
                    {
                        current.Append(trimmed);
                    }
                    else
                    {
                        if (current.Length > 0)
                            current.Append(' ');
                        current.Append(trimmed);
                    }
                }
            }

            if (current.Length > 0)
            {
                // 残余太短就并入上一批，避免单标点 / 寄生小批次浪费推理。
                if (batches.Count > 0 && current.Length <= MinTailPhonemes)
                {
                    string tail = current.ToString().Trim();
                    if (tail.Length > 0)
                    {
                        string prev = batches[^1];
                        string separator = tail.Length == 1 && PunctuationChars.Contains(tail[0])
                            ? string.Empty
                            : " ";
                        int previousBudget = useShortFirst && batches.Count == 1
                            ? firstBatchBudget
                            : MaxPhonemeLength;
                        if (prev.Length + separator.Length + tail.Length <= previousBudget)
                            batches[^1] = prev + separator + tail;
                        else
                            batches.Add(tail);
                    }
                }
                else
                {
                    batches.Add(current.ToString().Trim());
                }
            }

            return batches;
        }

        private static List<string> SplitByPunctuation(string text)
        {
            var result = new List<string>();
            int start = 0;

            for (int i = 0; i < text.Length; i++)
            {
                if (PunctuationChars.Contains(text[i]))
                {
                    if (i > start)
                        result.Add(text[start..i]);
                    result.Add(text[i].ToString());
                    start = i + 1;
                }
            }

            if (start < text.Length)
                result.Add(text[start..]);

            return result;
        }

        /// <summary>
        /// 把超过预算的无标点片段硬切成多批, 首块用 <paramref name="firstChunkBudget"/> (降低首块延迟),
        /// 后续块用 <paramref name="restChunkBudget"/> (= MaxPhonemeLength)。优先在空格处切断以保持
        /// 音素 token 完整; 找不到空格时直接按预算硬切。每个块 trim 后非空才入列。
        /// </summary>
        private static void AppendHardSplit(string text, List<string> batches, int firstChunkBudget, int restChunkBudget)
        {
            int start = 0;
            bool first = true;
            while (start < text.Length)
            {
                int budget = first ? firstChunkBudget : restChunkBudget;
                first = false;
                if (budget < 1) budget = restChunkBudget;

                int remaining = text.Length - start;
                if (remaining <= budget)
                {
                    string tail = text[start..].Trim();
                    if (tail.Length > 0) batches.Add(tail);
                    return;
                }

                // 在 [start, start+budget) 内找最后一个空格, 保持音素 token 完整。
                int windowEnd = start + budget;
                int cut = -1;
                for (int k = windowEnd - 1; k > start; k--)
                {
                    if (text[k] == ' ') { cut = k; break; }
                }
                int end = cut > start ? cut : windowEnd;
                string piece = text[start..end].Trim();
                if (piece.Length > 0) batches.Add(piece);
                start = cut > start ? cut + 1 : windowEnd;
            }
        }
    }
}
