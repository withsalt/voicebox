using System.Text;

namespace VoiceBox.Assistant.Modules.Recognition.Formatting
{
    internal static class AsrTextNormalizer
    {
        private const string PlaybackBargeInFillerChars = "嗯恩啊呃额哦喔噢唔哼";
        private static readonly HashSet<string> PlaybackBargeInFillerWords = new(StringComparer.OrdinalIgnoreCase)
        {
            "uh",
            "um",
            "mm",
            "hm",
            "hmm",
            "mhm",
        };

        /// <summary>
        /// 去掉 ASR 输出里残留的特殊标记 (如 &lt;sil&gt; / &lt;noise&gt; / &lt;laughter&gt; 等)。
        /// </summary>
        public static string Sanitize(string? text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;

            var sb = new StringBuilder(text.Length);
            int i = 0;
            while (i < text.Length)
            {
                if (text[i] == '<')
                {
                    int end = text.IndexOf('>', i + 1);
                    if (end > i && end - i <= 32)
                    {
                        i = end + 1;
                        continue;
                    }
                }

                sb.Append(text[i]);
                i++;
            }

            return sb.ToString().Trim();
        }

        /// <summary>
        /// 检测 ASR 文本是否含有至少一个有意义字符。纯标点、纯空白等明显噪音会被过滤。
        /// </summary>
        public static bool HasMeaningfulCharacters(string text)
        {
            foreach (char c in text)
            {
                if (char.IsLetterOrDigit(c)) return true;
            }

            return false;
        }

        /// <summary>
        /// 判断识别文本是否包含任一配置短语。匹配时只保留字母和数字,
        /// 因此 ASR 插入的空白、中文或英文标点不会影响结果。
        /// </summary>
        public static bool ContainsAnyPhrase(string text, IEnumerable<string>? phrases)
        {
            if (string.IsNullOrWhiteSpace(text) || phrases is null)
                return false;

            string normalizedText = CompactLettersAndDigits(text);
            if (normalizedText.Length == 0)
                return false;

            foreach (string? phrase in phrases)
            {
                string normalizedPhrase = CompactLettersAndDigits(phrase ?? string.Empty);
                if (normalizedPhrase.Length > 0 && normalizedText.Contains(normalizedPhrase, StringComparison.Ordinal))
                    return true;
            }

            return false;
        }

        /// <summary>
        /// 播放期 barge-in 的二次确认: 普通对话里的"嗯"仍可作为有效发言,
        /// 但播放中短促的"嗯/啊/哦/呃"更常见于误触发或 ASR 占位, 不应打断 TTS。
        /// </summary>
        public static bool IsMeaningfulPlaybackBargeInText(string text)
        {
            if (string.IsNullOrWhiteSpace(text) || !HasMeaningfulCharacters(text))
                return false;

            string compact = CompactLettersAndDigits(text);
            if (compact.Length == 0)
                return false;

            if (PlaybackBargeInFillerWords.Contains(compact))
                return false;

            if (compact.Length <= 2)
            {
                bool allFillers = true;
                foreach (char c in compact)
                {
                    if (!PlaybackBargeInFillerChars.Contains(c))
                    {
                        allFillers = false;
                        break;
                    }
                }

                if (allFillers)
                    return false;
            }

            return true;
        }

        private static string CompactLettersAndDigits(string text)
        {
            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if (char.IsLetterOrDigit(c))
                    sb.Append(char.ToLowerInvariant(c));
            }

            return sb.ToString();
        }
    }
}
