using System.Collections.Frozen;
using System.Text;
using VoiceBox.ASR.FireRedASR2.Internal.Model;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Default FireRed token assembler. Combines BPE / WordPiece sub-words for non-Chinese
    /// pieces and emits Chinese characters as-is. Behaviour is identical to the original
    /// monolithic Decode() — only the structure has been pulled out.
    /// </summary>
    internal sealed class FireRedTokenAssembler : ITokenAssembler
    {
        private const int DecodeStopTokenId = 2;
        private const char WordBoundary = '▁'; // ▁

        private static readonly char[] TokenSplitChars = { '\t', ' ' };

        private static readonly FrozenSet<string> SpecialTokens = new HashSet<string>(StringComparer.Ordinal)
        {
            "</s>", "<s>", "<sos/eos>", "<blank>", "<unk>", "<sos>", "<eos>", "<pad>",
            "<sil>", "<silence>", "<noise>", "<spnoise>", "<vocalized-noise>",
            "<music>", "<laughter>", "<breath>", "<cough>", "<non-speech>",
        }.ToFrozenSet(StringComparer.Ordinal);

        private readonly string[] _vocabulary;

        public FireRedTokenAssembler(string[] vocabulary)
        {
            ArgumentNullException.ThrowIfNull(vocabulary);
            _vocabulary = vocabulary;
        }

        public OfflineRecognizerResultEntity Assemble(IReadOnlyList<long> tokens, IReadOnlyList<int[]> timestamps)
        {
            var result = new OfflineRecognizerResultEntity();
            var textBuilder = new StringBuilder();
            string lastToken = string.Empty;
            int[]? lastTimestamp = null;

            int count = Math.Min(tokens.Count, timestamps.Count);
            for (int i = 0; i < count; i++)
            {
                long tokenId = tokens[i];
                int[] timestamp = timestamps[i];
                if (tokenId == DecodeStopTokenId) break;
                if (tokenId < 0 || tokenId >= _vocabulary.Length) continue;

                string current = _vocabulary[tokenId].Split(TokenSplitChars, 2)[0].ToLowerInvariant();
                if (SpecialTokens.Contains(current)) continue;

                if (IsAllChinese(current))
                {
                    textBuilder.Append(current);
                    result.Tokens.Add(current);
                    result.Timestamps.Add(timestamp);
                    continue;
                }

                // Non-Chinese: surround with word-boundary markers and merge with prior piece.
                textBuilder.Append(WordBoundary).Append(current).Append(WordBoundary);

                string combined = lastToken + WordBoundary + current + WordBoundary;
                bool hasBpeMarker = combined.IndexOf("@@▁▁", StringComparison.Ordinal) > 0;
                int underscoreCount = CountChar(combined, WordBoundary);
                bool isWordPiece = (underscoreCount == 3 || underscoreCount == 5)
                    && combined.IndexOf("▁▁▁", StringComparison.Ordinal) < 0;

                if (hasBpeMarker)
                {
                    string mergedToken = combined.Replace("@@▁▁", string.Empty);
                    int[] mergedTimestamp = MergeTimestamps(lastTimestamp, timestamp);
                    ReplaceLastOrAdd(result.Tokens, mergedToken.Replace(WordBoundary.ToString(), string.Empty), allowEmpty: false);
                    ReplaceLastOrAdd(result.Timestamps, mergedTimestamp, allowEmpty: false);
                    lastToken = mergedToken;
                    lastTimestamp = mergedTimestamp;
                }
                else if (isWordPiece)
                {
                    string mergedToken = combined.Replace("▁▁", string.Empty);
                    int[] mergedTimestamp = MergeTimestamps(lastTimestamp, timestamp);
                    ReplaceLastOrAdd(result.Tokens, mergedToken.Replace(WordBoundary.ToString(), string.Empty), allowEmpty: true);
                    ReplaceLastOrAdd(result.Timestamps, mergedTimestamp, allowEmpty: true);
                    lastToken = mergedToken;
                    lastTimestamp = mergedTimestamp;
                }
                else
                {
                    result.Tokens.Add(current.Replace(WordBoundary.ToString(), string.Empty));
                    result.Timestamps.Add(timestamp);
                    lastToken = WordBoundary + current + WordBoundary;
                    lastTimestamp = timestamp;
                }
            }

            string text = textBuilder.ToString();
            if (text.IndexOf("@@▁▁", StringComparison.Ordinal) > 0
                || text.IndexOf("▁▁▁", StringComparison.Ordinal) < 0)
            {
                text = text
                    .Replace("@@▁▁", string.Empty)
                    .Replace("▁▁", " ")
                    .Replace("@@", " ")
                    .Replace(WordBoundary.ToString(), " ");
            }
            else
            {
                text = text
                    .Replace("▁▁▁", " ")
                    .Replace("▁▁", string.Empty)
                    .Replace(WordBoundary.ToString(), string.Empty);
            }

            result.Text = text;
            result.TextLen = text.Length;
            return result;
        }

        private static bool IsAllChinese(string s)
        {
            if (s.Length == 0) return false;
            foreach (char c in s)
                if (c < 0x4e00 || c > 0x9fa5) return false;
            return true;
        }

        private static int CountChar(string s, char c)
        {
            int n = 0;
            for (int i = 0; i < s.Length; i++) if (s[i] == c) n++;
            return n;
        }

        private static int[] MergeTimestamps(int[]? prev, int[] curr)
        {
            if (prev is null) return curr;
            int[] merged = new int[prev.Length + curr.Length];
            Array.Copy(prev, merged, prev.Length);
            Array.Copy(curr, 0, merged, prev.Length, curr.Length);
            return merged;
        }

        private static void ReplaceLastOrAdd<T>(IList<T> list, T value, bool allowEmpty)
        {
            if (list.Count > 0) list[^1] = value;
            else if (allowEmpty) list.Add(value);
        }
    }
}
