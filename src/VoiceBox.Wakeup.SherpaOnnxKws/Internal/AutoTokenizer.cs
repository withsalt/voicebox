using System.Text;
using Misaki.NET.Lang.Chinese;
using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 从关键词原文 (中文 / 英文 / 中英混排) 自动生成 sherpa-onnx KWS 模型可识别的 token 序列。
    /// <para>
    /// 中文路径: 调用 <see cref="PinyinLookup"/> 得到 tone3 拼音, 再按《汉语拼音方案》把
    /// 「声母 + 带调韵母」逐字拆成两个 token (例如 "周望军" → "zh ōu w àng j ūn"),
    /// y/w 作为声母保留, ü 用 v 暂存并最终替换。结果与 sherpa-onnx 自带 keywords.txt 对齐。
    /// </para>
    /// <para>
    /// 英文路径: 查模型同目录的 CMU 风格发音词典 (例如 en.phone): "HELLO" → "HH AH0 L OW1"。
    /// </para>
    /// <para>
    /// 自动生成的变体:
    /// </para>
    /// <list type="bullet">
    ///   <item>3+3 上声变调: 任意相邻两个 tone-3 中, 前者口语中常念成 tone-2 (例 "你好" → "ní hǎo")。</item>
    ///   <item>多音字穷举 (可选): 当某汉字回落到单字字典且字典中给了多个读音时, 把每个读音都
    ///         作为一条候选, 在多个汉字间做笛卡尔积。短语字典命中的段不展开 (短语字典已限定上下文读音)。</item>
    /// </list>
    /// </summary>
    internal sealed class AutoTokenizer
    {
        private static readonly HashSet<string> TwoCharInitials = new(StringComparer.Ordinal) { "zh", "ch", "sh" };
        private static readonly HashSet<char> SingleCharInitials = new()
        {
            'b','p','m','f','d','t','n','l',
            'g','k','h','j','q','x',
            'r','z','c','s','y','w'
        };

        private static readonly Dictionary<char, string> ToneMarks = new()
        {
            ['a'] = "āáǎà",
            ['o'] = "ōóǒò",
            ['e'] = "ēéěè",
            ['i'] = "īíǐì",
            ['u'] = "ūúǔù",
            ['v'] = "ǖǘǚǜ",
        };

        private readonly SymbolTable _symbols;
        private readonly PinyinLookup _pinyin;
        private readonly CmuPhoneDictionary? _cmuDict;
        private readonly bool _expandPolyphone;
        private readonly int _maxVariants;
        private readonly KwsConfusionVariants _confusion;

        public AutoTokenizer(SymbolTable symbols, CmuPhoneDictionary? cmuDict)
            : this(symbols, cmuDict, expandPolyphone: true, maxVariants: 16, confusion: KwsConfusionVariants.None)
        {
        }

        /// <param name="expandPolyphone">单字回落时是否把多个候选读音展开成额外变体。</param>
        /// <param name="maxVariants">单个关键词输出的变体数量上限 (含规范读音), ≤0 视为不限。</param>
        /// <param name="confusion">叠加的口音 / 方言混淆规则; <see cref="KwsConfusionVariants.None"/> 表示不生成混淆变体。</param>
        public AutoTokenizer(SymbolTable symbols, CmuPhoneDictionary? cmuDict,
            bool expandPolyphone, int maxVariants,
            KwsConfusionVariants confusion = KwsConfusionVariants.None)
        {
            _symbols = symbols ?? throw new ArgumentNullException(nameof(symbols));
            _pinyin = PinyinLookup.Instance;
            _cmuDict = cmuDict;
            _expandPolyphone = expandPolyphone;
            _maxVariants = maxVariants;
            _confusion = confusion;
        }

        /// <summary>原文 → 单一规范 token 序列 (无变体)。</summary>
        public string Tokenize(string keyword) => TokenizeAll(keyword)[0];

        /// <summary>
        /// 原文 → 1~N 条 token 序列。第 0 条始终为规范读音 (短语字典 + 单字第一读音 + 不变调),
        /// 之后按以下顺序追加: 规范读音的 3+3 上声变调; 多音字 (单字回落) 笛卡尔积及其变调。
        /// 全部 token 都能在 tokens.txt 中找到。调用方可把每条变体作为独立 ContextGraph 入口注册,
        /// 共享同一 Phrase。
        /// </summary>
        public IReadOnlyList<string> TokenizeAll(string keyword)
        {
            if (string.IsNullOrWhiteSpace(keyword))
                throw new ArgumentException("Keyword must not be empty.", nameof(keyword));

            var plan = BuildPlan(keyword);

            // 规范读音 = 每段都取 readings[0]
            int[] choice = new int[plan.Count];
            string canonical = RenderPlan(plan, choice, applySandhi: false);

            var seen = new HashSet<string>(StringComparer.Ordinal) { canonical };
            var variants = new List<string> { canonical };

            void TryAdd(string s)
            {
                if (seen.Add(s)) variants.Add(s);
            }

            bool ReachedLimit() => _maxVariants > 0 && variants.Count >= _maxVariants;

            // 规范读音的 3+3 上声变调
            if (HasAdjacentTone3(plan, choice))
            {
                TryAdd(RenderPlan(plan, choice, applySandhi: true));
                if (ReachedLimit()) return variants;
            }

            // 多音字穷举 (笛卡尔积) + 每条候选的变调
            if (_expandPolyphone && HasPolyphone(plan))
            {
                foreach (var combo in EnumerateChoices(plan))
                {
                    if (ReachedLimit()) break;
                    if (IsAllZero(combo)) continue; // 规范读音已在变体[0]
                    TryAdd(RenderPlan(plan, combo, applySandhi: false));
                    if (ReachedLimit()) break;
                    if (HasAdjacentTone3(plan, combo))
                        TryAdd(RenderPlan(plan, combo, applySandhi: true));
                }
            }

            // 混淆变体: 对当前所有 base 变体做单点声/韵替换 (zh↔z 等)。
            // 单点替换避免组合爆炸; 已存在的 base 数量在此被冻结, 防止与下方循环互相产卵。
            if (_confusion != KwsConfusionVariants.None && !ReachedLimit())
            {
                int baseCount = variants.Count;
                for (int i = 0; i < baseCount && !ReachedLimit(); i++)
                    EmitConfusionVariants(variants[i], TryAdd, ReachedLimit);
            }

            return variants;
        }

        // ---- 混淆变体规则 ---------------------------------------------------

        // 声母互换对 (单点替换). 仅在该规则被启用时使用.
        private static readonly (string A, string B)[] InitialSwapPairs =
        {
            ("zh", "z"), ("ch", "c"), ("sh", "s"), // RetroflexFlat
            ("n",  "l"),                            // NasalLateral
            ("f",  "h"),                            // FrontVelar
            ("r",  "l"),                            // Rhotic
        };

        // 给定声母 token, 返回它在某条规则下的对换值 (空表示不命中规则).
        private string? TrySwapInitial(string token)
        {
            foreach (var (a, b) in InitialSwapPairs)
            {
                if (token == a)
                {
                    var rule = ClassifyInitialRule(a);
                    if ((_confusion & rule) != 0) return b;
                }
                else if (token == b)
                {
                    var rule = ClassifyInitialRule(a); // pair owner = a
                    if ((_confusion & rule) != 0) return a;
                }
            }
            return null;
        }

        private static KwsConfusionVariants ClassifyInitialRule(string a) => a switch
        {
            "zh" or "ch" or "sh" => KwsConfusionVariants.RetroflexFlat,
            "n" => KwsConfusionVariants.NasalLateral,
            "f" => KwsConfusionVariants.FrontVelar,
            "r" => KwsConfusionVariants.Rhotic,
            _ => KwsConfusionVariants.None,
        };

        // 末尾鼻音互换: token 以 'n' 结尾 → 加 'g' 变 ng; 以 'ng' 结尾 → 去掉 'g'.
        // 声调标记在韵腹元音上 (例 "ān" / "āng"), 末字符不会是声调字符, 所以只看末尾辅音.
        private string? TrySwapNasalFinal(string token)
        {
            if ((_confusion & KwsConfusionVariants.NasalFinal) == 0) return null;
            if (token.Length < 2) return null;

            if (token.EndsWith("ng", StringComparison.Ordinal))
                return token[..^1]; // 去 g

            if (token[^1] == 'n')
                return token + "g"; // 加 g

            return null;
        }

        /// <summary>
        /// 对一条 base 变体 (空格分隔的 token 串) 做"单点"声韵母替换:
        /// 每次只替换一个 token, 替换后必须能在 tokens.txt 中找到, 否则丢弃。
        /// </summary>
        private void EmitConfusionVariants(string baseVariant, Action<string> tryAdd, Func<bool> reachedLimit)
        {
            var tokens = baseVariant.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int idx = 0; idx < tokens.Length; idx++)
            {
                if (reachedLimit()) return;
                string original = tokens[idx];

                string? swappedInitial = TrySwapInitial(original);
                if (swappedInitial != null && _symbols.Contains(swappedInitial))
                {
                    tokens[idx] = swappedInitial;
                    tryAdd(string.Join(' ', tokens));
                    tokens[idx] = original;
                    if (reachedLimit()) return;
                }

                string? swappedFinal = TrySwapNasalFinal(original);
                if (swappedFinal != null && _symbols.Contains(swappedFinal))
                {
                    tokens[idx] = swappedFinal;
                    tryAdd(string.Join(' ', tokens));
                    tokens[idx] = original;
                    if (reachedLimit()) return;
                }
            }
        }

        // ---- 计划阶段: 把原文切成 [English phones] / [中文音节] 序列 ---------

        // 多字短语长度上限。phrase_tone3.json 几乎全部 entry 在 4 字以内,
        // 但留 8 字余量以兼容偶尔的成语 / 长词。
        private const int MaxPhraseLength = 8;

        private List<Segment> BuildPlan(string keyword)
        {
            var plan = new List<Segment>();
            var enBuffer = new StringBuilder();
            var cjkRun = new StringBuilder();

            foreach (char ch in keyword)
            {
                if (IsCjk(ch))
                {
                    FlushEnglish(enBuffer, plan, keyword);
                    cjkRun.Append(ch);
                }
                else if (IsAsciiLetter(ch) || ch == '\'')
                {
                    FlushChinese(cjkRun, plan, keyword);
                    enBuffer.Append(ch);
                }
                else
                {
                    FlushChinese(cjkRun, plan, keyword);
                    FlushEnglish(enBuffer, plan, keyword);
                }
            }
            FlushChinese(cjkRun, plan, keyword);
            FlushEnglish(enBuffer, plan, keyword);

            if (plan.Count == 0)
                throw new InvalidDataException(
                    $"Cannot auto-tokenize keyword '{keyword}': no recognizable Chinese/English content.");
            return plan;
        }

        /// <summary>
        /// 把累积的 CJK 字符串按 phrase_tone3.json 做贪心最长匹配。匹配到的整段调用一次
        /// <see cref="PinyinLookup.GetTone3(string)"/> 得到 N 个带调拼音 (覆盖多音字),
        /// 没匹配到的字回落到单字读音 (此时支持多音字穷举)。
        /// </summary>
        private void FlushChinese(StringBuilder buf, List<Segment> plan, string keyword)
        {
            if (buf.Length == 0) return;
            string run = buf.ToString();
            buf.Clear();

            int i = 0;
            while (i < run.Length)
            {
                int matched = 0;
                int maxLen = Math.Min(MaxPhraseLength, run.Length - i);
                for (int len = maxLen; len >= 2; len--)
                {
                    if (_pinyin.IsPhrase(run.Substring(i, len))) { matched = len; break; }
                }

                if (matched >= 2)
                {
                    string phrase = run.Substring(i, matched);
                    var pinyins = _pinyin.GetTone3(phrase);
                    if (pinyins.Count != matched)
                        throw new InvalidDataException(
                            $"Phrase '{phrase}' (in keyword '{keyword}') returned {pinyins.Count} pinyin syllables for {matched} chars.");
                    for (int k = 0; k < matched; k++)
                    {
                        var reading = SplitInitialFinalAndTone(pinyins[k], keyword);
                        // 短语字典命中的字: 读音由短语上下文唯一确定, 不展开多音字。
                        plan.Add(Segment.Chinese(new[] { reading }, phrase[k]));
                    }
                    i += matched;
                }
                else
                {
                    plan.Add(BuildChineseSegment(run[i], keyword));
                    i++;
                }
            }
        }

        private Segment BuildChineseSegment(char ch, string keyword)
        {
            var allReadings = _pinyin.GetAllTone3(ch);
            if (allReadings.Count == 0 || string.IsNullOrEmpty(allReadings[0]))
                throw new InvalidDataException(
                    $"Pinyin lookup failed for character '{ch}' in keyword '{keyword}'. " +
                    "请确认该字在 Misaki.NET 拼音字典中存在, 或通过 AddKeyword 显式指定 Tokens。");

            var readings = new Reading[allReadings.Count];
            for (int i = 0; i < allReadings.Count; i++)
                readings[i] = SplitInitialFinalAndTone(allReadings[i], keyword);
            return Segment.Chinese(readings, ch);
        }

        private void FlushEnglish(StringBuilder buf, List<Segment> plan, string keyword)
        {
            if (buf.Length == 0) return;
            string word = buf.ToString();
            buf.Clear();
            if (_cmuDict == null)
                throw new InvalidDataException(
                    $"Keyword '{keyword}' contains English word '{word}' but no CMU phone dictionary " +
                    "was configured (set " + nameof(SherpaOnnxKwsOptions.CmuPhoneDictionaryPath) + ").");
            if (!_cmuDict.TryGet(word, out var phones))
                throw new InvalidDataException(
                    $"English word '{word}' (in keyword '{keyword}') is not in the CMU phone dictionary. " +
                    "请补充发音字典, 或通过 AddKeyword 显式指定 Tokens。");

            foreach (var p in phones)
            {
                if (!_symbols.Contains(p))
                    throw new InvalidDataException(
                        $"CMU phone '{p}' (for word '{word}') not in tokens.txt.");
            }
            plan.Add(Segment.English(phones));
        }

        // ---- 渲染阶段 -------------------------------------------------------

        /// <param name="choice">每个段选用第几个 reading; 英文段忽略。</param>
        private string RenderPlan(List<Segment> plan, int[] choice, bool applySandhi)
        {
            // 计算每个中文段的实际声调 (考虑 3+3 → 2+3 上声变调)
            int[]? effectiveTones = null;
            if (applySandhi)
            {
                effectiveTones = new int[plan.Count];
                for (int i = 0; i < plan.Count; i++)
                {
                    if (!plan[i].IsChinese) { effectiveTones[i] = 0; continue; }
                    int tone = plan[i].Readings![choice[i]].Tone;
                    if (tone == 3 && i + 1 < plan.Count && plan[i + 1].IsChinese
                        && plan[i + 1].Readings![choice[i + 1]].Tone == 3)
                        tone = 2;
                    effectiveTones[i] = tone;
                }
            }

            var output = new List<string>();
            for (int i = 0; i < plan.Count; i++)
            {
                var seg = plan[i];
                if (!seg.IsChinese)
                {
                    output.AddRange(seg.EnglishPhones!);
                    continue;
                }
                var reading = seg.Readings![choice[i]];
                int tone = effectiveTones != null ? effectiveTones[i] : reading.Tone;
                string finalToken = ApplyTone(reading.FinalBody, tone);
                if (finalToken.IndexOf('v') >= 0) finalToken = finalToken.Replace("v", "ü");

                if (!string.IsNullOrEmpty(reading.Initial))
                {
                    if (!_symbols.Contains(reading.Initial))
                        throw new InvalidDataException(
                            $"Initial '{reading.Initial}' (from '{seg.SourceChar}') not found in tokens.txt.");
                    output.Add(reading.Initial);
                }
                if (!_symbols.Contains(finalToken))
                    throw new InvalidDataException(
                        $"Final '{finalToken}' (from '{seg.SourceChar}', tone={tone}) not found in tokens.txt.");
                output.Add(finalToken);
            }
            return string.Join(' ', output);
        }

        private static bool HasAdjacentTone3(List<Segment> plan, int[] choice)
        {
            for (int i = 0; i + 1 < plan.Count; i++)
            {
                if (plan[i].IsChinese && plan[i].Readings![choice[i]].Tone == 3 &&
                    plan[i + 1].IsChinese && plan[i + 1].Readings![choice[i + 1]].Tone == 3)
                    return true;
            }
            return false;
        }

        private static bool HasPolyphone(List<Segment> plan)
        {
            for (int i = 0; i < plan.Count; i++)
                if (plan[i].IsChinese && plan[i].Readings!.Length > 1) return true;
            return false;
        }

        private static bool IsAllZero(int[] choice)
        {
            for (int i = 0; i < choice.Length; i++) if (choice[i] != 0) return false;
            return true;
        }

        /// <summary>枚举每段 readings 索引的笛卡尔积。规范组合 (全 0) 也会被产生。</summary>
        private static IEnumerable<int[]> EnumerateChoices(List<Segment> plan)
        {
            int n = plan.Count;
            int[] sizes = new int[n];
            for (int i = 0; i < n; i++)
                sizes[i] = plan[i].IsChinese ? plan[i].Readings!.Length : 1;
            int[] cur = new int[n];
            while (true)
            {
                yield return (int[])cur.Clone();
                int k = n - 1;
                while (k >= 0)
                {
                    cur[k]++;
                    if (cur[k] < sizes[k]) break;
                    cur[k] = 0;
                    k--;
                }
                if (k < 0) yield break;
            }
        }

        // ---- 拼音拆分 -------------------------------------------------------

        private static Reading SplitInitialFinalAndTone(string tone3, string keyword)
        {
            if (string.IsNullOrEmpty(tone3))
                throw new InvalidDataException($"Empty pinyin for keyword '{keyword}'.");

            int tone;
            string body;
            char lastChar = tone3[^1];
            if (char.IsDigit(lastChar))
            {
                tone = lastChar - '0';
                body = tone3[..^1];
            }
            else
            {
                tone = 5;
                body = tone3;
            }
            if (body.Length == 0)
                throw new InvalidDataException($"Malformed pinyin '{tone3}' for keyword '{keyword}'.");

            string initial;
            string final;
            if (body.Length >= 2 && TwoCharInitials.Contains(body[..2]))
            {
                initial = body[..2];
                final = body[2..];
            }
            else if (SingleCharInitials.Contains(body[0]))
            {
                initial = body[..1];
                final = body[1..];
            }
            else
            {
                initial = string.Empty;
                final = body;
            }

            if (final.Length == 0)
                throw new InvalidDataException($"Pinyin '{tone3}' (keyword '{keyword}') has no final.");

            return new Reading(initial, final, tone);
        }

        private static string ApplyTone(string final, int tone)
        {
            if (tone < 1 || tone > 4) return final;

            int idx;
            char vowel;
            if (final == "iu") { idx = 1; vowel = 'u'; }
            else if (final == "ui") { idx = 1; vowel = 'i'; }
            else
            {
                idx = -1; vowel = '\0';
                foreach (char c in "aoeiuv")
                {
                    int p = final.IndexOf(c);
                    if (p >= 0) { idx = p; vowel = c; break; }
                }
                if (idx < 0) return final;
            }

            string marks = ToneMarks[vowel];
            char marked = marks[tone - 1];
            return string.Concat(final.AsSpan(0, idx), marked.ToString(), final.AsSpan(idx + 1));
        }

        private static bool IsCjk(char ch) =>
            (ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF);

        private static bool IsAsciiLetter(char ch) =>
            (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');

        // ---- 内部计划单元 ---------------------------------------------------

        private readonly struct Reading
        {
            public readonly string Initial;     // 不含声调 (例如 "zh" / "" / "y")
            public readonly string FinalBody;   // 不含声调 (例如 "ou" / "ang" / "v" / "iu")
            public readonly int Tone;           // 1..5
            public Reading(string initial, string finalBody, int tone)
            {
                Initial = initial; FinalBody = finalBody; Tone = tone;
            }
        }

        private readonly struct Segment
        {
            public readonly bool IsChinese;
            public readonly Reading[]? Readings;     // 中文: 1~N 个候选读音, 第 0 个为规范
            public readonly char SourceChar;
            public readonly string[]? EnglishPhones;

            private Segment(bool isCh, Reading[]? readings, char src, string[]? phones)
            {
                IsChinese = isCh; Readings = readings; SourceChar = src; EnglishPhones = phones;
            }

            public static Segment Chinese(Reading[] readings, char src) =>
                new(true, readings, src, null);
            public static Segment English(string[] phones) =>
                new(false, null, '\0', phones);
        }
    }
}
