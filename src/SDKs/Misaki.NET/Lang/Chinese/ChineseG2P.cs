using System.Text;
using System.Text.RegularExpressions;
using Misaki.NET.Models;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Chinese Grapheme-to-Phoneme converter.
/// Converts Chinese text to Zhuyin (Bopomofo) or IPA phoneme representation.
/// </summary>
public sealed class ChineseG2P : IG2P
{
    private readonly string? _version;
    private readonly string _unk;
    private readonly Func<string, string>? _enCallable;
    private readonly PinyinLookup _pinyinLookup;
    private readonly WordSegmenter _segmenter;
    private readonly Dictionary<string, string> _userPhrases = new(StringComparer.Ordinal);

    #region ZH_MAP (Pinyin to Zhuyin mapping)

    private static readonly Dictionary<string, string> ZhMap = new()
    {
        ["b"] = "ㄅ", ["p"] = "ㄆ", ["m"] = "ㄇ", ["f"] = "ㄈ",
        ["d"] = "ㄉ", ["t"] = "ㄊ", ["n"] = "ㄋ", ["l"] = "ㄌ",
        ["g"] = "ㄍ", ["k"] = "ㄎ", ["h"] = "ㄏ", ["j"] = "ㄐ",
        ["q"] = "ㄑ", ["x"] = "ㄒ", ["zh"] = "ㄓ", ["ch"] = "ㄔ",
        ["sh"] = "ㄕ", ["r"] = "ㄖ", ["z"] = "ㄗ", ["c"] = "ㄘ",
        ["s"] = "ㄙ", ["a"] = "ㄚ", ["o"] = "ㄛ", ["e"] = "ㄜ",
        ["ie"] = "ㄝ", ["ai"] = "ㄞ", ["ei"] = "ㄟ", ["ao"] = "ㄠ",
        ["ou"] = "ㄡ", ["an"] = "ㄢ", ["en"] = "ㄣ", ["ang"] = "ㄤ",
        ["eng"] = "ㄥ", ["er"] = "ㄦ", ["i"] = "ㄧ", ["u"] = "ㄨ",
        ["v"] = "ㄩ", ["ii"] = "ㄭ", ["iii"] = "十",
        ["ve"] = "月", ["ia"] = "压", ["ian"] = "言", ["iang"] = "阳",
        ["iao"] = "要", ["in"] = "阴", ["ing"] = "应", ["iong"] = "用",
        ["iou"] = "又", ["ong"] = "中", ["ua"] = "穵", ["uai"] = "外",
        ["uan"] = "万", ["uang"] = "王", ["uei"] = "为", ["uen"] = "文",
        ["ueng"] = "瓮", ["uo"] = "我", ["van"] = "元", ["vn"] = "云",
        // Punctuation and special chars map to themselves
        [";"] = ";", [":"] = ":", [","] = ",", ["."] = ".",
        ["!"] = "!", ["?"] = "?", ["/"] = "/", ["—"] = "—",
        ["…"] = "…", ["\""] = "\"", ["("] = "(", [")"] = ")",
        ["\u201C"] = "\u201C", ["\u201D"] = "\u201D", [" "] = " ",
        ["1"] = "1", ["2"] = "2", ["3"] = "3", ["4"] = "4", ["5"] = "5",
        ["R"] = "R",
    };

    #endregion

    #region Erhua word lists

    private static readonly HashSet<string> MustErhua = new()
    {
        "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
    };

    private static readonly HashSet<string> NotErhua = new()
    {
        "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
        "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
        "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
        "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
        "狗儿", "少儿"
    };

    #endregion

    private static readonly HashSet<char> Punctuation = new(";:,.!?—…\"()\u201C\u201D");

    #region Built-in polyphone overrides

    /// <summary>
    /// Built-in corrections for phrases the embedded dictionaries get wrong.
    /// Applied at construction time before user-supplied <c>userPhrases</c>, so
    /// callers can still override any entry here. Each value is space-separated
    /// tone3 pinyin with one syllable per character.
    ///
    /// Selection criteria:
    ///   - The embedded phrase dict is missing the phrase, AND
    ///   - The default single-char reading is wrong in modern usage, AND
    ///   - The phrase is unambiguous (no plausible alternative reading).
    /// </summary>
    private static readonly Dictionary<string, string> BuiltInPhraseOverrides = new(StringComparer.Ordinal)
    {
        // 行 — default is xing2; the 量词/名词 reading hang2 is needed for
        // "一行 + 具体形象 (鸟/字/泪/诗/代码/命令)" contexts.
        ["一行白鹭"] = "yi1 hang2 bai2 lu4",
        ["一行白鹭上青天"] = "yi1 hang2 bai2 lu4 shang4 qing1 tian1",
        ["一行字"] = "yi1 hang2 zi4",
        ["一行诗"] = "yi1 hang2 shi1",
        ["一行话"] = "yi1 hang2 hua4",
        ["一行代码"] = "yi1 hang2 dai4 ma3",
        ["几行代码"] = "ji3 hang2 dai4 ma3",
        ["代码行"] = "dai4 ma3 hang2",
        ["命令行"] = "ming4 ling4 hang2",
        ["标题行"] = "biao1 ti2 hang2",
        ["两行泪"] = "liang3 hang2 lei4",
        ["两行字"] = "liang3 hang2 zi4",
        ["第一行"] = "di4 yi1 hang2",
        ["最后一行"] = "zui4 hou4 yi1 hang2",
        ["下一行"] = "xia4 yi1 hang2",
        ["上一行"] = "shang4 yi1 hang2",
        ["新一行"] = "xin1 yi1 hang2",
        ["另起一行"] = "ling4 qi3 yi1 hang2",

        // 重 — default is zhong4; modern "re-" prefix usage needs chong2.
        ["重做"] = "chong2 zuo4",
        ["重写"] = "chong2 xie3",
        ["重排"] = "chong2 pai2",
        ["重塑"] = "chong2 su4",
        ["重置"] = "chong2 zhi4",
        ["重连"] = "chong2 lian2",
        ["重传"] = "chong2 chuan2",
        ["重发"] = "chong2 fa1",
        ["重算"] = "chong2 suan4",
        ["重跑"] = "chong2 pao3",
        ["重玩"] = "chong2 wan2",
        ["重学"] = "chong2 xue2",
        ["重命名"] = "chong2 ming4 ming2",
        ["重定向"] = "chong2 ding4 xiang4",
        ["重定义"] = "chong2 ding4 yi4",
        ["重启动"] = "chong2 qi3 dong4",
        ["重装系统"] = "chong2 zhuang1 xi4 tong3",
        ["重建索引"] = "chong2 jian4 suo3 yin3",

        // 圈 — default is quan1; pen/enclosure for animals needs juan4.
        ["猪圈"] = "zhu1 juan4",
    };

    #endregion

    private static readonly Regex ChineseCharRegex = new(
        @"[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+",
        RegexOptions.Compiled);

    private static readonly Regex EnglishSegmentRegex = new(
        @"([A-Za-z \'-]*[A-Za-z][A-Za-z \'-]*)|([^A-Za-z]+)",
        RegexOptions.Compiled);

    private static readonly Regex SingleIFinalRegex = new(
        @"^i\d$",
        RegexOptions.Compiled);

    private static readonly Regex BeforeDigitRegex = new(
        @"(?=\d)",
        RegexOptions.Compiled);

    /// <summary>
    /// Creates a new Chinese G2P instance.
    /// </summary>
    /// <param name="version">Version string. Use "1.1" for the frontend pipeline with tone sandhi.</param>
    /// <param name="unk">Unknown phoneme placeholder.</param>
    /// <param name="enCallable">Optional callback for English text segments.</param>
    /// <param name="userPhrases">
    /// Optional map of user-defined phrase pronunciations, keyed by the phrase
    /// (e.g. <c>"一行白鹭"</c>) with space-separated tone3 pinyin as the value
    /// (e.g. <c>"yi1 hang2 bai2 lu4"</c>). Each phrase is registered with the segmenter
    /// so it survives jieba's cut, and its pinyin overrides both the embedded phrase
    /// dictionary and the per-character dictionary. Useful for poetry, names, and
    /// domain terms where the built-in heteronym choice is wrong.
    /// </param>
    public ChineseG2P(
        string? version = "1.1",
        string unk = "❓",
        Func<string, string>? enCallable = null,
        IReadOnlyDictionary<string, string>? userPhrases = null)
    {
        _version = version;
        _unk = unk;
        _enCallable = enCallable;
        _pinyinLookup = PinyinLookup.Instance;
        _segmenter = new WordSegmenter(_pinyinLookup);
        // Initialize the bridges for tone sandhi merge and sub-word splitting
        PinyinLookupBridge.Initialize(_pinyinLookup);
        JiebaBridge.CutForSearchFunc = _segmenter.CutForSearch;

        // Built-in polyphone corrections first; user-supplied entries override them.
        foreach (var (phrase, pinyin) in BuiltInPhraseOverrides)
            AddUserPhrase(phrase, pinyin);

        if (userPhrases != null)
        {
            foreach (var (phrase, pinyin) in userPhrases)
                AddUserPhrase(phrase, pinyin);
        }
    }

    /// <summary>
    /// Registers (or updates) a user-defined phrase pronunciation at runtime.
    /// Same semantics as the constructor's <c>userPhrases</c> parameter.
    /// </summary>
    public void AddUserPhrase(string phrase, string tone3Pinyin)
    {
        if (string.IsNullOrEmpty(phrase) || string.IsNullOrWhiteSpace(tone3Pinyin))
            return;
        var parts = tone3Pinyin.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != phrase.Length)
            throw new ArgumentException(
                $"Pinyin token count ({parts.Length}) must match phrase char count ({phrase.Length}): " +
                $"\"{phrase}\" / \"{tone3Pinyin}\"",
                nameof(tone3Pinyin));

        _userPhrases[phrase] = string.Join(' ', parts);
        // Force jieba to treat the phrase as a single token so the override key matches.
        _segmenter.AddWord(phrase);
    }

    /// <summary>Removes a previously registered user phrase. The jieba-side word entry is left in place.</summary>
    public bool RemoveUserPhrase(string phrase) => _userPhrases.Remove(phrase);

    /// <summary>
    /// Processes Chinese text and returns phoneme transcription.
    /// Supports inline pinyin annotations of the form <c>[word|tone3 pinyin]</c>
    /// (e.g. <c>"一[行|hang2]白鹭"</c>) for one-off polyphone control.
    /// </summary>
    public G2PResult Process(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new G2PResult(string.Empty);

        // Inline annotation pass: strips [词|...] markers and accumulates per-call overrides.
        text = InlinePhonemeParser.Parse(text, out var inlineOverrides);

        // Build the effective override view: user-phrase defaults overridden by inline annotations.
        // Skip allocation when both are empty so the hot path stays cheap.
        IReadOnlyDictionary<string, string>? overrides = null;
        if (_userPhrases.Count > 0 || inlineOverrides.Count > 0)
        {
            if (inlineOverrides.Count == 0)
            {
                overrides = _userPhrases;
            }
            else if (_userPhrases.Count == 0)
            {
                overrides = inlineOverrides;
            }
            else
            {
                var merged = new Dictionary<string, string>(_userPhrases, StringComparer.Ordinal);
                foreach (var (k, v) in inlineOverrides)
                    merged[k] = v;
                overrides = merged;
            }
        }

        // Register inline-annotated phrases with the segmenter so they survive jieba's cut.
        foreach (var key in inlineOverrides.Keys)
            _segmenter.AddWord(key);

        // Number normalization (cn2an transform equivalent)
        text = NumberConverter.Transform(text);

        // Map Chinese punctuation to ASCII
        text = MapPunctuation(text);

        if (_version != "1.1")
        {
            return new G2PResult(LegacyCall(text));
        }

        // Version 1.1: Frontend pipeline with tone sandhi
        var segments = new List<string>();
        var matches = EnglishSegmentRegex.Matches(text);

        foreach (Match m in matches)
        {
            var en = m.Groups[1].Value.Trim();
            var zh = m.Groups[2].Value.Trim();

            if (!string.IsNullOrEmpty(zh))
            {
                var (phonemes, _) = ProcessFrontend(zh, overrides);
                segments.Add(phonemes);
            }
            else if (!string.IsNullOrEmpty(en))
            {
                if (_enCallable != null)
                    segments.Add(_enCallable(en));
                else
                    segments.Add(_unk);
            }
        }

        var result = string.Join(" ", segments.Where(s => !string.IsNullOrEmpty(s)));
        return new G2PResult(result);
    }

    /// <summary>
    /// Frontend pipeline: segmentation → pinyin → tone sandhi → erhua → Zhuyin encoding.
    /// </summary>
    private (string phonemes, List<MToken> tokens) ProcessFrontend(
        string text, IReadOnlyDictionary<string, string>? overrides)
    {
        var tokens = new List<MToken>();

        // Word segmentation with POS tagging
        var segCut = _segmenter.Segment(text);

        // Pre-merge for tone sandhi
        segCut = ToneSandhi.PreMergeForModify(segCut);

        foreach (var (word, pos) in segCut)
        {
            // Adjust POS for Chinese chars tagged as 'x'
            var actualPos = pos;
            if (pos == "x" && word.Length > 0 && word.All(c => c >= '\u4E00' && c <= '\u9FFF'))
                actualPos = "X";
            else if (pos != "x" && word.Length == 1 && Punctuation.Contains(word[0]))
                actualPos = "x";

            var tk = new MToken { Text = word, Tag = actualPos, Whitespace = "" };

            if (actualPos is "x" or "eng")
            {
                if (!string.IsNullOrWhiteSpace(word))
                {
                    if (actualPos == "x" && word.Length == 1 && Punctuation.Contains(word[0]))
                        tk.Phonemes = word;
                    tokens.Add(tk);
                }
                else if (tokens.Count > 0)
                {
                    tokens[^1].Whitespace += word;
                }
                continue;
            }

            if (tokens.Count > 0 && tokens[^1].Tag is not "x" and not "eng"
                && string.IsNullOrEmpty(tokens[^1].Whitespace))
            {
                tokens[^1].Whitespace = "/";
            }

            // Get initials and finals in a single pass (shares the underlying tone3 lookup).
            var (initials, finals) = _pinyinLookup.GetInitialsAndFinals(word, overrides);

            // Handle special finals (ii, iii)
            for (int i = 0; i < initials.Count && i < finals.Count; i++)
            {
                var f = finals[i];
                if (SingleIFinalRegex.IsMatch(f))
                {
                    if (initials[i] is "z" or "c" or "s")
                        finals[i] = f.Replace("i", "ii");
                    else if (initials[i] is "zh" or "ch" or "sh" or "r")
                        finals[i] = f.Replace("i", "iii");
                }
            }

            // Tone sandhi
            finals = ToneSandhi.ModifiedTone(word, actualPos, finals);

            // Erhua merging
            (initials, finals) = MergeErhua(initials, finals, word, actualPos);

            // Convert to Zhuyin
            var phones = new List<string>();
            for (int i = 0; i < initials.Count && i < finals.Count; i++)
            {
                var c = initials[i];
                var v = finals[i];
                if (!string.IsNullOrEmpty(c))
                    phones.Add(c);
                if (!string.IsNullOrEmpty(v) && (v != c || !Punctuation.Contains(v[0])))
                    phones.Add(v);
            }

            // Join with underscore, then split on tone digits
            var phoneStr = string.Join("_", phones);
            // Erhua'd "e" final → "er" (ㄦ absorbs the R marker)
            if (phoneStr.StartsWith("eR"))
                phoneStr = "er" + phoneStr[2..];
            phoneStr = phoneStr.Replace("_eR", "_er");
            // Separate remaining R markers into their own segment
            phoneStr = phoneStr.Replace("R", "_R");
            phoneStr = BeforeDigitRegex.Replace(phoneStr, "_");
            var phoneParts = phoneStr.Split('_', StringSplitOptions.RemoveEmptyEntries);

            tk.Phonemes = string.Concat(phoneParts.Select(p =>
                ZhMap.TryGetValue(p, out var mapped) ? mapped : _unk));
            tokens.Add(tk);
        }

        var result = string.Concat(tokens.Select(tk =>
            (tk.Phonemes ?? _unk) + tk.Whitespace));

        return (result, tokens);
    }

    /// <summary>
    /// Legacy call: simple pinyin → IPA conversion without tone sandhi.
    /// </summary>
    private string LegacyCall(string text)
    {
        var sb = new StringBuilder();
        var matches = ChineseCharRegex.Matches(text);
        bool isZh = text.Length > 0 && text[0] >= '\u4E00' && text[0] <= '\u9FFF';

        foreach (Match m in matches)
        {
            var segment = m.Value;
            if (isZh)
            {
                var words = _segmenter.SegmentWords(segment);
                var ipaParts = new List<string>();
                foreach (var word in words)
                {
                    var pinyins = _pinyinLookup.GetTone3(word);
                    var ipa = string.Concat(pinyins.Select(py => PinyinToIpa.Retone(PinyinToIpa.Convert(py))));
                    ipaParts.Add(ipa);
                }
                sb.Append(string.Join(" ", ipaParts));
            }
            else
            {
                sb.Append(segment);
            }
            isZh = !isZh;
        }

        return sb.ToString();
    }

    private (List<string> initials, List<string> finals) MergeErhua(
        List<string> initials, List<string> finals, string word, string pos)
    {
        // Fix er1 -> er2 for final 儿
        for (int i = 0; i < finals.Count; i++)
        {
            if (i == finals.Count - 1 && i < word.Length && word[i] == '儿' && finals[i] == "er1")
                finals[i] = "er2";
        }

        // Check if erhua should be applied
        if (!MustErhua.Contains(word) && (NotErhua.Contains(word) || pos is "a" or "j" or "nr"))
            return (initials, finals);

        if (finals.Count != word.Length)
            return (initials, finals);

        var newInitials = new List<string>();
        var newFinals = new List<string>();

        for (int i = 0; i < finals.Count; i++)
        {
            if (i == finals.Count - 1 && word[i] == '儿'
                && finals[i] is "er2" or "er5"
                && (word.Length < 2 || !NotErhua.Contains(word[^2..]))
                && newFinals.Count > 0)
            {
                // Merge erhua: add R before tone number
                var lastFinal = newFinals[^1];
                newFinals[^1] = lastFinal[..^1] + "R" + lastFinal[^1];
            }
            else
            {
                newInitials.Add(initials[i]);
                newFinals.Add(finals[i]);
            }
        }

        return (newInitials, newFinals);
    }

    private static string MapPunctuation(string text)
    {
        var sb = new StringBuilder(text.Length + 16);
        for (int i = 0; i < text.Length; i++)
        {
            var c = text[i];
            string? replacement = c switch
            {
                '、' or '，' => ", ",
                '。' or '．' => ". ",
                '！' => "! ",
                '：' => ": ",
                '；' => "; ",
                '？' => "? ",
                '«' or '《' or '「' or '【' => " \"",
                '»' or '》' or '」' or '】' => "\" ",
                '（' => " (",
                '）' => ") ",
                _ => null
            };
            if (replacement != null)
                sb.Append(replacement);
            else
                sb.Append(c);
        }
        return sb.ToString().Trim();
    }
}
