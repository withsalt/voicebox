using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Misaki.NET.Lang.English;

/// <summary>
/// English pronunciation lexicon with gold/silver dictionaries and morphological stemming.
/// </summary>
public sealed class EnglishLexicon
{
    private readonly bool _british;
    private readonly Dictionary<string, object> _golds; // string or Dictionary<string,string?>
    private readonly Dictionary<string, object> _silvers;

    private static readonly HashSet<int> LexiconOrds = new(
        new[] { 39, 45 }.Concat(Enumerable.Range(65, 26)).Concat(Enumerable.Range(97, 26)));

    private static readonly HashSet<char> Consonants = new("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ");
    private static readonly HashSet<char> Vowels = new("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");
    private static readonly HashSet<char> UsTaus = new("AIOWYiuæɑəɛɪɹʊʌ");
    private static readonly HashSet<string> Ordinals = new() { "st", "nd", "rd", "th" };

    private static readonly Regex SuffixRegex = new(@"[a-z']+$", RegexOptions.Compiled);
    private static readonly Regex NonAlphaRegex = new(@"[^a-z]+", RegexOptions.Compiled);
    private static readonly Regex DoubleConsonantIngRegex = new(
        @"([bcdgklmnprstvxz])\1ing$|cking$", RegexOptions.Compiled);
    private static readonly Regex VsRegex = new(@"(?i)vs\.?$", RegexOptions.Compiled);

    private static readonly Dictionary<string, (string major, string minor)> Currencies = new()
    {
        ["$"] = ("dollar", "cent"),
        ["£"] = ("pound", "pence"),
        ["€"] = ("euro", "cent"),
    };

    private static readonly Dictionary<string, string> AddSymbols = new()
    {
        ["."] = "dot", ["/"] = "slash"
    };

    private static readonly Dictionary<string, string> Symbols = new()
    {
        ["%"] = "percent", ["&"] = "and", ["+"] = "plus", ["@"] = "at"
    };

    private const char PrimaryStress = 'ˈ';
    private const char SecondaryStress = 'ˌ';
    private static readonly string Stresses = "ˌˈ";

    public EnglishLexicon(bool british)
    {
        _british = british;
        var prefix = british ? "gb" : "us";
        _golds = LoadAndGrowDictionary($"Misaki.NET.Resources.English.{prefix}_gold.json");
        _silvers = LoadAndGrowDictionary($"Misaki.NET.Resources.English.{prefix}_silver.json");
    }

    private static Dictionary<string, object> LoadAndGrowDictionary(string resourceName)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException($"Resource not found: {resourceName}");
        using var reader = new StreamReader(stream);
        var json = reader.ReadToEnd();

        var raw = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json)
            ?? new Dictionary<string, JsonElement>();

        var dict = new Dictionary<string, object>(raw.Count * 2, StringComparer.Ordinal);
        foreach (var (key, value) in raw)
        {
            if (value.ValueKind == JsonValueKind.String)
            {
                dict[key] = value.GetString()!;
            }
            else if (value.ValueKind == JsonValueKind.Object)
            {
                var inner = new Dictionary<string, string?>(StringComparer.Ordinal);
                foreach (var prop in value.EnumerateObject())
                {
                    inner[prop.Name] = prop.Value.ValueKind == JsonValueKind.Null
                        ? null : prop.Value.GetString();
                }
                dict[key] = inner;
            }
        }

        return GrowDictionary(dict);
    }

    private static Dictionary<string, object> GrowDictionary(Dictionary<string, object> d)
    {
        var extras = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (k, v) in d)
        {
            if (k.Length < 2) continue;
            if (k == k.ToLowerInvariant())
            {
                var cap = char.ToUpperInvariant(k[0]) + k[1..];
                if (cap != k)
                    extras.TryAdd(cap, v);
            }
            else if (k == char.ToUpperInvariant(k[0]) + k[1..].ToLowerInvariant())
            {
                extras.TryAdd(k.ToLowerInvariant(), v);
            }
        }

        foreach (var (k, v) in extras)
            d.TryAdd(k, v);

        return d;
    }

    /// <summary>
    /// Looks up pronunciation for a token.
    /// </summary>
    public (string? phonemes, int? rating) GetWord(string word, string? tag, double? stress, TokenContext ctx)
    {
        var (ps, rating) = GetSpecialCase(word, tag, stress, ctx);
        if (ps != null) return (ps, rating);

        var wl = word.ToLowerInvariant();
        if (word.Length > 1 && word.Replace("'", "").All(char.IsLetter) && word != wl
            && (tag != "NNP" || word.Length > 7)
            && !_golds.ContainsKey(word) && !_silvers.ContainsKey(word)
            && (word == word.ToUpperInvariant() || word[1..] == word[1..].ToLowerInvariant())
            && (_golds.ContainsKey(wl) || _silvers.ContainsKey(wl)
                || StemS(wl, tag, stress, ctx).phonemes != null
                || StemEd(wl, tag, stress, ctx).phonemes != null
                || StemIng(wl, tag, stress, ctx).phonemes != null))
        {
            word = wl;
        }

        if (IsKnown(word, tag))
            return Lookup(word, tag, stress, ctx);

        if (word.EndsWith("s'") && IsKnown(word[..^2] + "'s", tag))
            return Lookup(word[..^2] + "'s", tag, stress, ctx);

        if (word.EndsWith("'") && IsKnown(word[..^1], tag))
            return Lookup(word[..^1], tag, stress, ctx);

        var stemS = StemS(word, tag, stress, ctx);
        if (stemS.phonemes != null) return stemS;

        var stemEd = StemEd(word, tag, stress, ctx);
        if (stemEd.phonemes != null) return stemEd;

        var stemIng = StemIng(word, tag, stress ?? 0.5, ctx);
        if (stemIng.phonemes != null) return stemIng;

        return (null, null);
    }

    /// <summary>
    /// Looks up a number and converts to phonemes.
    /// </summary>
    public (string? phonemes, int? rating) GetNumber(string word, string? currency, bool isHead, string numFlags)
    {
        var suffix = SuffixRegex.Match(word);
        string? suffixStr = suffix.Success ? suffix.Value : null;
        if (suffixStr != null) word = word[..^suffixStr.Length];

        var result = new List<(string ps, int rating)>();

        if (word.StartsWith('-'))
        {
            result.Add(Lookup("minus", null, null, null));
            word = word[1..];
        }

        void ExtendNum(string num, bool first = true, bool escape = false)
        {
            string text = escape ? num : NumberToWords.ToCardinal(long.Parse(num));
            var splits = NonAlphaRegex.Split(text).Where(s => !string.IsNullOrEmpty(s)).ToArray();
            for (int i = 0; i < splits.Length; i++)
            {
                var w = splits[i];
                if (w == "and" && !numFlags.Contains('&'))
                {
                    if (numFlags.Contains('n') && result.Count > 0)
                    {
                        var last = result[^1];
                        result[^1] = (last.ps + "ən", last.rating);
                    }
                    continue;
                }
                if (first && i == 0 && splits.Length > 1 && w == "one" && numFlags.Contains('a'))
                {
                    result.Add(("ə", 4));
                }
                else
                {
                    result.Add(Lookup(w, null, w == "point" ? -2.0 : null, null));
                }
            }
        }

        bool IsDigit(string s) => s.All(char.IsAsciiDigit);

        if (IsDigit(word) && suffixStr != null && Ordinals.Contains(suffixStr))
        {
            var ordinal = NumberToWords.ToOrdinal(long.Parse(word));
            ExtendNum(ordinal, escape: true);
        }
        else if (result.Count == 0 && word.Length == 4 && currency == null && IsDigit(word))
        {
            var year = NumberToWords.ToYear(int.Parse(word));
            ExtendNum(year, escape: true);
        }
        else if (!isHead && !word.Contains('.'))
        {
            var num = word.Replace(",", "");
            if (num.StartsWith('0') || num.Length > 3)
            {
                foreach (var n in num)
                    ExtendNum(n.ToString(), first: false);
            }
            else if (num.Length == 3 && !num.EndsWith("00"))
            {
                ExtendNum(num[0].ToString());
                if (num[1] == '0')
                {
                    result.Add(Lookup("O", null, -2.0, null));
                    ExtendNum(num[2].ToString(), first: false);
                }
                else
                {
                    ExtendNum(num[1..], first: false);
                }
            }
            else
            {
                ExtendNum(num);
            }
        }
        else if (currency != null && Currencies.ContainsKey(currency) && IsCurrency(word))
        {
            var parts = word.Replace(",", "").Split('.');
            var pairs = new List<(int num, string unit)>();
            var curr = Currencies[currency];
            if (parts.Length > 0 && int.TryParse(parts[0], out var major))
                pairs.Add((major, curr.major));
            if (parts.Length > 1 && int.TryParse(parts[1], out var minor))
                pairs.Add((minor, curr.minor));

            if (pairs.Count > 1 && pairs[1].num == 0)
                pairs.RemoveAt(1);
            else if (pairs.Count > 1 && pairs[0].num == 0)
                pairs.RemoveAt(0);

            for (int i = 0; i < pairs.Count; i++)
            {
                if (i > 0) result.Add(Lookup("and", null, null, null));
                ExtendNum(pairs[i].num.ToString(), first: i == 0);
                if (Math.Abs(pairs[i].num) != 1 && pairs[i].unit != "pence")
                {
                    var stemResult = StemS(pairs[i].unit + "s", null, null, null);
                    result.Add((stemResult.phonemes ?? "", stemResult.rating ?? 4));
                }
                else
                    result.Add(Lookup(pairs[i].unit, null, null, null));
            }
        }
        else
        {
            if (IsDigit(word))
            {
                ExtendNum(NumberToWords.ToCardinal(long.Parse(word)), escape: true);
            }
            else if (!word.Contains('.'))
            {
                var num = word.Replace(",", "");
                var text = suffixStr != null && Ordinals.Contains(suffixStr)
                    ? NumberToWords.ToOrdinal(long.Parse(num))
                    : NumberToWords.ToCardinal(long.Parse(num));
                ExtendNum(text, escape: true);
            }
            else
            {
                var cleaned = word.Replace(",", "");
                if (cleaned.StartsWith('.'))
                {
                    var text = "point " + string.Join(" ",
                        cleaned[1..].Select(c => NumberToWords.ToCardinal(c - '0')));
                    ExtendNum(text, escape: true);
                }
                else
                {
                    ExtendNum(NumberToWords.ToCardinal(double.Parse(cleaned,
                        System.Globalization.CultureInfo.InvariantCulture)), escape: true);
                }
            }
        }

        if (result.Count == 0) return (null, null);

        var phonemes = string.Join(" ", result.Select(r => r.ps));
        var minRating = result.Min(r => r.rating);

        if (suffixStr is "s" or "'s")
            return (AppendS(phonemes), minRating);
        if (suffixStr is "ed" or "'d")
            return (AppendEd(phonemes), minRating);
        if (suffixStr == "ing")
            return (AppendIng(phonemes), minRating);

        return (phonemes, minRating);
    }

    #region Morphological Stemming

    public (string? phonemes, int? rating) StemS(string word, string? tag, double? stress, TokenContext? ctx)
    {
        if (word.Length < 3 || !word.EndsWith('s')) return (null, null);

        string? stem = null;
        if (!word.EndsWith("ss") && IsKnown(word[..^1], tag))
            stem = word[..^1];
        else if ((word.EndsWith("'s") || (word.Length > 4 && word.EndsWith("es") && !word.EndsWith("ies")))
                 && IsKnown(word[..^2], tag))
            stem = word[..^2];
        else if (word.Length > 4 && word.EndsWith("ies") && IsKnown(word[..^3] + "y", tag))
            stem = word[..^3] + "y";

        if (stem == null) return (null, null);

        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (AppendS(ps), rating);
    }

    public (string? phonemes, int? rating) StemEd(string word, string? tag, double? stress, TokenContext? ctx)
    {
        if (word.Length < 4 || !word.EndsWith('d')) return (null, null);

        string? stem = null;
        if (!word.EndsWith("dd") && IsKnown(word[..^1], tag))
            stem = word[..^1];
        else if (word.Length > 4 && word.EndsWith("ed") && !word.EndsWith("eed") && IsKnown(word[..^2], tag))
            stem = word[..^2];

        if (stem == null) return (null, null);

        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (AppendEd(ps), rating);
    }

    public (string? phonemes, int? rating) StemIng(string word, string? tag, double? stress, TokenContext? ctx)
    {
        if (word.Length < 5 || !word.EndsWith("ing")) return (null, null);

        string? stem = null;
        if (word.Length > 5 && IsKnown(word[..^3], tag))
            stem = word[..^3];
        else if (IsKnown(word[..^3] + "e", tag))
            stem = word[..^3] + "e";
        else if (word.Length > 5 && DoubleConsonantIngRegex.IsMatch(word)
                 && IsKnown(word[..^4], tag))
            stem = word[..^4];

        if (stem == null) return (null, null);

        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (AppendIng(ps), rating);
    }

    #endregion

    #region Suffix Phonology

    private string? AppendS(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char last = stem[^1];
        if ("ptkfθ".Contains(last)) return stem + "s";
        if ("szʃʒʧʤ".Contains(last)) return stem + (_british ? "ɪ" : "ᵻ") + "z";
        return stem + "z";
    }

    private string? AppendEd(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char last = stem[^1];
        if ("pkfθʃsʧ".Contains(last)) return stem + "t";
        if (last == 'd') return stem + (_british ? "ɪ" : "ᵻ") + "d";
        if (last != 't') return stem + "d";
        if (_british || stem.Length < 2) return stem + "ɪd";
        if (UsTaus.Contains(stem[^2])) return stem[..^1] + "ɾᵻd";
        return stem + "ᵻd";
    }

    private string? AppendIng(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        if (_british && stem[^1] is 'ə' or 'ː') return null;
        if (!_british && stem.Length > 1 && stem[^1] == 't' && UsTaus.Contains(stem[^2]))
            return stem[..^1] + "ɾɪŋ";
        return stem + "ɪŋ";
    }

    #endregion

    #region Lookup Helpers

    public bool IsKnown(string word, string? tag)
    {
        if (_golds.ContainsKey(word) || Symbols.ContainsKey(word) || _silvers.ContainsKey(word))
            return true;
        if (!word.All(c => LexiconOrds.Contains(c)))
            return false;
        if (word.Length == 1) return true;
        if (word == word.ToUpperInvariant() && _golds.ContainsKey(word.ToLowerInvariant()))
            return true;
        return word.Length > 1 && word[1..] == word[1..].ToUpperInvariant();
    }

    public (string ps, int rating) Lookup(string word, string? tag, double? stress, TokenContext? ctx)
    {
        bool isNNP = false;
        if (word == word.ToUpperInvariant() && !_golds.ContainsKey(word))
        {
            word = word.ToLowerInvariant();
            isNNP = tag == "NNP";
        }

        string? ps = null;
        int rating = 4;

        if (_golds.TryGetValue(word, out var goldVal))
        {
            ps = ResolveValue(goldVal, tag, ctx);
        }

        if (ps == null && !isNNP && _silvers.TryGetValue(word, out var silverVal))
        {
            ps = ResolveValue(silverVal, tag, ctx);
            rating = 3;
        }

        if (ps == null || (isNNP && !ps.Contains(PrimaryStress)))
        {
            var (nnpPs, nnpRating) = GetNNP(word);
            if (nnpPs != null) return (nnpPs, nnpRating);
        }

        ps ??= "";
        return (ApplyStress(ps, stress), rating);
    }

    private static string? ResolveValue(object value, string? tag, TokenContext? ctx)
    {
        if (value is string s) return s;
        if (value is Dictionary<string, string?> dict)
        {
            if (ctx?.FutureVowel == null && dict.ContainsKey("None"))
                tag = "None";
            else if (tag != null && !dict.ContainsKey(tag))
                tag = GetParentTag(tag);

            if (tag != null && dict.TryGetValue(tag, out var tagVal))
                return tagVal;
            return dict.TryGetValue("DEFAULT", out var def) ? def : null;
        }
        return null;
    }

    private (string? ps, int rating) GetNNP(string word)
    {
        var parts = new List<string>();
        foreach (var c in word.Where(char.IsLetter))
        {
            var upper = c.ToString().ToUpperInvariant();
            if (!_golds.TryGetValue(upper, out var val)) return (null, 3);
            var ps = val is string s ? s : null;
            if (ps == null) return (null, 3);
            parts.Add(ps);
        }
        if (parts.Count == 0) return (null, 3);

        var joined = ApplyStress(string.Concat(parts), 0);
        var lastIdx = joined.LastIndexOf(SecondaryStress);
        if (lastIdx >= 0)
        {
            joined = joined[..lastIdx] + PrimaryStress + joined[(lastIdx + 1)..];
        }
        return (joined, 3);
    }

    private (string? ps, int? rating) GetSpecialCase(string word, string? tag, double? stress, TokenContext ctx)
    {
        if (tag == "ADD" && AddSymbols.TryGetValue(word, out var sym))
            return Lookup(sym, null, -0.5, ctx);

        if (Symbols.TryGetValue(word, out var symWord))
            return Lookup(symWord, null, null, ctx);

        if (word.Contains('.') && !word.StartsWith('.') && !word.EndsWith('.')
            && word.Replace(".", "").All(char.IsLetter)
            && word.Split('.').Max(s => s.Length) < 3)
            return GetNNP(word);

        if (word is "a" or "A")
            return (tag == "DT" ? "ɐ" : "ˈA", 4);

        if (word is "am" or "Am" or "AM")
        {
            if (tag?.StartsWith("NN") == true) return GetNNP(word);
            if (ctx.FutureVowel == null || word != "am" || (stress.HasValue && stress > 0))
            {
                var amVal = _golds.TryGetValue("am", out var v) ? v as string : null;
                return (amVal ?? "æm", 4);
            }
            return ("ɐm", 4);
        }

        if (word is "an" or "An" or "AN")
        {
            if (word == "AN" && tag?.StartsWith("NN") == true) return GetNNP(word);
            return ("ɐn", 4);
        }

        if (word == "I" && tag == "PRP")
            return ($"{SecondaryStress}I", 4);

        if (word is "by" or "By" or "BY" && GetParentTag(tag) == "ADV")
            return ("bˈI", 4);

        if (word is "to" or "To" || (word == "TO" && tag is "TO" or "IN"))
        {
            if (ctx.FutureVowel == null)
            {
                var toVal = _golds.TryGetValue("to", out var v) ? v as string : null;
                return (toVal ?? "tˈu", 4);
            }
            return (ctx.FutureVowel == true ? "tʊ" : "tə", 4);
        }

        if (word is "in" or "In" || (word == "IN" && tag != "NNP"))
        {
            var s = ctx.FutureVowel == null || tag != "IN" ? PrimaryStress.ToString() : "";
            return (s + "ɪn", 4);
        }

        if (word is "the" or "The" || (word == "THE" && tag == "DT"))
            return (ctx.FutureVowel == true ? "ði" : "ðə", 4);

        if (tag == "IN" && VsRegex.IsMatch(word))
            return Lookup("versus", null, null, ctx);

        if (word is "used" or "Used" or "USED")
        {
            if (tag is "VBD" or "JJ" && ctx.FutureTo)
            {
                var usedVal = _golds.TryGetValue("used", out var v) ? v : null;
                if (usedVal is Dictionary<string, string?> d && d.TryGetValue("VBD", out var vbd))
                    return (vbd, 4);
            }
            var defVal = _golds.TryGetValue("used", out var dv) ? dv : null;
            if (defVal is Dictionary<string, string?> dd && dd.TryGetValue("DEFAULT", out var def))
                return (def, 4);
            if (defVal is string ds) return (ds, 4);
        }

        return (null, null);
    }

    private static string GetParentTag(string? tag)
    {
        if (tag == null) return "";
        if (tag.StartsWith("VB")) return "VERB";
        if (tag.StartsWith("NN")) return "NOUN";
        if (tag.StartsWith("ADV") || tag.StartsWith("RB")) return "ADV";
        if (tag.StartsWith("ADJ") || tag.StartsWith("JJ")) return "ADJ";
        return tag;
    }

    #endregion

    #region Stress

    public static string ApplyStress(string ps, double? stress)
    {
        if (stress == null) return ps;

        if (stress < -1)
            return ps.Replace(PrimaryStress.ToString(), "").Replace(SecondaryStress.ToString(), "");

        if (stress == -1 || (stress is 0 or -0.5 && ps.Contains(PrimaryStress)))
            return ps.Replace(SecondaryStress.ToString(), "")
                     .Replace(PrimaryStress, SecondaryStress);

        if (stress is 0 or 0.5 or 1 && !ps.Any(c => Stresses.Contains(c)))
        {
            if (!ps.Any(c => Vowels.Contains(c))) return ps;
            return Restress(SecondaryStress + ps);
        }

        if (stress >= 1 && !ps.Contains(PrimaryStress) && ps.Contains(SecondaryStress))
            return ps.Replace(SecondaryStress, PrimaryStress);

        if (stress > 1 && !ps.Any(c => Stresses.Contains(c)))
        {
            if (!ps.Any(c => Vowels.Contains(c))) return ps;
            return Restress(PrimaryStress + ps);
        }

        return ps;
    }

    private static string Restress(string ps)
    {
        // Move stress marks to just before their target vowel
        Span<char> buf = ps.Length <= 64 ? stackalloc char[ps.Length] : new char[ps.Length];
        int writeIdx = 0;
        int i = 0;

        while (i < ps.Length)
        {
            if (Stresses.Contains(ps[i]))
            {
                char stressChar = ps[i];
                i++;
                // Find the next vowel and insert stress mark just before it
                bool inserted = false;
                int savedStart = writeIdx;
                while (i < ps.Length && !Vowels.Contains(ps[i]) && !Stresses.Contains(ps[i]))
                {
                    buf[writeIdx++] = ps[i];
                    i++;
                }
                if (i < ps.Length && Vowels.Contains(ps[i]))
                {
                    // Shift consonants after stress mark
                    int consonantCount = writeIdx - savedStart;
                    // Insert stress before the vowel (after consonants already written is correct position)
                    buf[writeIdx++] = stressChar;
                    buf[writeIdx++] = ps[i];
                    i++;
                    inserted = true;
                }
                if (!inserted)
                {
                    // No vowel found, put stress at saved position
                    // Shift what we wrote to make room
                    int consonantCount = writeIdx - savedStart;
                    for (int j = writeIdx - 1; j >= savedStart; j--)
                        buf[j + 1] = buf[j];
                    buf[savedStart] = stressChar;
                    writeIdx++;
                }
            }
            else
            {
                buf[writeIdx++] = ps[i];
                i++;
            }
        }

        return new string(buf[..writeIdx]);
    }

    #endregion

    #region Utility

    public static bool IsNumber(string word, bool isHead)
    {
        if (word.All(c => !char.IsAsciiDigit(c))) return false;

        string[] suffixes = ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"];
        foreach (var s in suffixes)
        {
            if (word.EndsWith(s))
            {
                word = word[..^s.Length];
                break;
            }
        }

        for (int i = 0; i < word.Length; i++)
        {
            var c = word[i];
            if (char.IsAsciiDigit(c) || c is ',' or '.') continue;
            if (isHead && i == 0 && c == '-') continue;
            return false;
        }
        return true;
    }

    public static bool IsCurrency(string word)
    {
        if (!word.Contains('.')) return true;
        if (word.Count(c => c == '.') > 1) return false;
        var cents = word.Split('.')[1];
        return cents.Length < 3 || cents.All(c => c == '0');
    }

    #endregion
}

/// <summary>
/// Context for token processing (tracks what comes next in the sentence).
/// </summary>
public sealed class TokenContext
{
    public bool? FutureVowel { get; set; }
    public bool FutureTo { get; set; }
}
