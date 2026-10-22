using System.Text;

namespace VoiceBox.Assistant.Modules.Response.Formatting
{
    /// <summary>
    /// 把 LLM 输出里对 TTS 朗读没意义、且容易导致音素生成错乱的字符过滤掉。
    /// 主要包括 emoji / 各类符号 / 零宽与变体选择符 / 常见 markdown 标记。
    /// 不动中英文、数字、常规标点和空白。
    /// </summary>
    internal static class TtsTextSanitizer
    {
        public static string Clean(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            // 多数 chunk 都不含被屏蔽的字符, 快速路径直接返回原串。
            bool needsClean = false;
            foreach (var rune in text.EnumerateRunes())
            {
                if (IsBlocked(rune.Value)) { needsClean = true; break; }
            }
            if (!needsClean) return text;

            var sb = new StringBuilder(text.Length);
            Span<char> encodedRune = stackalloc char[2];
            foreach (var rune in text.EnumerateRunes())
            {
                if (IsBlocked(rune.Value)) continue;
                int length = rune.EncodeToUtf16(encodedRune);
                sb.Append(encodedRune[..length]);
            }
            return sb.ToString();
        }

        private static bool IsBlocked(int cp)
        {
            // Emoji & symbol planes
            if (cp >= 0x1F000 && cp <= 0x1FFFF) return true;   // Emoticons / Misc Symbols and Pictographs / Transport / Supplemental Symbols
            if (cp >= 0x2600 && cp <= 0x27BF) return true;     // Misc Symbols + Dingbats
            if (cp >= 0x2300 && cp <= 0x23FF) return true;     // Misc Technical (含 ⌚⌛ 等)
            if (cp >= 0x2B00 && cp <= 0x2BFF) return true;     // Misc Symbols and Arrows
            if (cp >= 0xE000 && cp <= 0xF8FF) return true;     // Private Use Area

            // 零宽 / 变体选择符: 即使 emoji 本体被删, ZWJ/VS 残留也会让后续字符渲染异常
            if (cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0xFE0E || cp == 0xFE0F) return true;

            // Markdown 常见格式字符: 朗读无意义且常成对出现破坏分句
            if (cp == '*' || cp == '`' || cp == '~') return true;

            return false;
        }
    }
}
