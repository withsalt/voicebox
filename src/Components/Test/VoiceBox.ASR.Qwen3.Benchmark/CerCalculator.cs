using System.Globalization;
using System.Text;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 中文 ASR 的字符错误率 (CER) 计算工具。
    /// 在对比前会做轻量归一化: 全角空格/常见标点去除、英文大写统一、Unicode NFC, 防止
    /// 因标点/大小写差异导致虚高 CER。归一化后逐"字符"(实际是逐 Rune/code point) 计算
    /// Levenshtein 编辑距离。中文一个字 = 一个 code point, 与 AIShell 评测惯例一致。
    /// </summary>
    internal static class CerCalculator
    {
        public readonly struct Score
        {
            public int Substitutions { get; init; }
            public int Insertions { get; init; }
            public int Deletions { get; init; }
            public int RefLength { get; init; }
            public int HypLength { get; init; }
            public double Cer => RefLength == 0
                ? (HypLength == 0 ? 0.0 : 1.0)
                : (double)(Substitutions + Insertions + Deletions) / RefLength;
            public bool ExactMatch => Substitutions == 0 && Insertions == 0 && Deletions == 0;
        }

        public static Score Compute(string reference, string hypothesis)
        {
            int[] refChars = ToCodePoints(Normalize(reference));
            int[] hypChars = ToCodePoints(Normalize(hypothesis));
            return Levenshtein(refChars, hypChars);
        }

        public static string Normalize(string? s)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            string nfc = s.Normalize(NormalizationForm.FormC);
            var sb = new StringBuilder(nfc.Length);
            foreach (var rune in nfc.EnumerateRunes())
            {
                int cp = rune.Value;
                if (cp <= 0x20) continue;                       // ASCII 控制符与空格
                if (cp == 0x3000) continue;                     // 全角空格
                if (IsPunctuation(cp)) continue;
                if (cp >= 'A' && cp <= 'Z')
                    sb.Append((char)(cp - 'A' + 'a'));           // 英文转小写
                else
                    sb.Append(rune.ToString());
            }
            return sb.ToString();
        }

        private static bool IsPunctuation(int cp)
        {
            // 常见 ASCII 标点
            if (cp == '.' || cp == ',' || cp == '!' || cp == '?' || cp == ';' || cp == ':' ||
                cp == '"' || cp == '\'' || cp == '`' || cp == '-' || cp == '_' || cp == '/' ||
                cp == '\\' || cp == '(' || cp == ')' || cp == '[' || cp == ']' ||
                cp == '{' || cp == '}' || cp == '<' || cp == '>' || cp == '~' || cp == '|' ||
                cp == '@' || cp == '#' || cp == '$' || cp == '%' || cp == '^' || cp == '&' || cp == '*')
                return true;
            // 中文/全角标点 (CJK Symbols and Punctuation + Halfwidth/Fullwidth Forms 子集)
            if (cp >= 0x3001 && cp <= 0x303F) return true;       // 、。〃〈〉《》「」『』
            if (cp == 0xFF0C || cp == 0xFF0E || cp == 0xFF1F || cp == 0xFF01 || // ，．？！
                cp == 0xFF1A || cp == 0xFF1B ||                                 // ：；
                cp == 0xFF08 || cp == 0xFF09 ||                                 // （）
                cp == 0xFF02 || cp == 0xFF07 ||                                 // ＂＇
                cp == 0x201C || cp == 0x201D || cp == 0x2018 || cp == 0x2019 ||  // 智能引号
                cp == 0x2026 || cp == 0x2014 || cp == 0x2013)                    // …——–
                return true;
            return false;
        }

        private static int[] ToCodePoints(string s)
        {
            if (s.Length == 0) return Array.Empty<int>();
            var list = new List<int>(s.Length);
            foreach (var rune in s.EnumerateRunes())
                list.Add(rune.Value);
            return list.ToArray();
        }

        private static Score Levenshtein(int[] reference, int[] hypothesis)
        {
            int n = reference.Length;
            int m = hypothesis.Length;
            if (n == 0)
            {
                return new Score
                {
                    Insertions = m,
                    RefLength = 0,
                    HypLength = m,
                };
            }
            if (m == 0)
            {
                return new Score
                {
                    Deletions = n,
                    RefLength = n,
                    HypLength = 0,
                };
            }

            // dp[i, j] = edit distance between ref[..i] and hyp[..j]
            // 同时跟踪 (S, I, D) 三个分量以便后续输出 wer-like 细分。
            var dp = new int[n + 1, m + 1];
            var ops = new (int s, int i, int d)[n + 1, m + 1];
            for (int i = 0; i <= n; i++) { dp[i, 0] = i; ops[i, 0] = (0, 0, i); }
            for (int j = 0; j <= m; j++) { dp[0, j] = j; ops[0, j] = (0, j, 0); }

            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    int cost = reference[i - 1] == hypothesis[j - 1] ? 0 : 1;
                    int sub = dp[i - 1, j - 1] + cost;
                    int ins = dp[i, j - 1] + 1;
                    int del = dp[i - 1, j] + 1;
                    int best = Math.Min(sub, Math.Min(ins, del));
                    dp[i, j] = best;
                    if (best == sub)
                    {
                        var prev = ops[i - 1, j - 1];
                        ops[i, j] = cost == 0 ? prev : (prev.s + 1, prev.i, prev.d);
                    }
                    else if (best == ins)
                    {
                        var prev = ops[i, j - 1];
                        ops[i, j] = (prev.s, prev.i + 1, prev.d);
                    }
                    else
                    {
                        var prev = ops[i - 1, j];
                        ops[i, j] = (prev.s, prev.i, prev.d + 1);
                    }
                }
            }

            var (sCount, iCount, dCount) = ops[n, m];
            return new Score
            {
                Substitutions = sCount,
                Insertions = iCount,
                Deletions = dCount,
                RefLength = n,
                HypLength = m,
            };
        }

        public static string FormatPercent(double v) => (v * 100).ToString("F2", CultureInfo.InvariantCulture) + "%";
    }
}
