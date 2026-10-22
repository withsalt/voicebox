using System.Reflection;
using System.Text.Json;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Provides pinyin dictionary lookup for Chinese characters and phrases.
/// Loads character and phrase dictionaries from embedded resources lazily and thread-safely.
/// </summary>
public class PinyinLookup
{
    /// <summary>Thread-safe singleton instance.</summary>
    public static PinyinLookup Instance { get; } = new();

    private static readonly string[] TwoCharInitials = { "zh", "ch", "sh" };

    private static readonly HashSet<string> Initials = new(StringComparer.Ordinal)
    {
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "zh", "ch", "sh", "r",
        "z", "c", "s", "j", "q", "x", "y", "w"
    };

    private static readonly Lazy<Dictionary<string, string>> CharDict = new(
        LoadCharDictionary, LazyThreadSafetyMode.ExecutionAndPublication);

    private static readonly Lazy<Dictionary<string, string>> PhraseDict = new(
        LoadPhraseDictionary, LazyThreadSafetyMode.ExecutionAndPublication);

    // Pre-built single-char lookup to avoid ToString() per character
    private static readonly Lazy<Dictionary<char, string>> SingleCharDict = new(
        BuildSingleCharDict, LazyThreadSafetyMode.ExecutionAndPublication);

    private static Dictionary<char, string> BuildSingleCharDict()
    {
        var source = CharDict.Value;
        var dict = new Dictionary<char, string>(source.Count);
        foreach (var (key, value) in source)
        {
            if (key.Length == 1)
                dict[key[0]] = value;
        }
        return dict;
    }

    private static Dictionary<string, string> LoadCharDictionary()
    {
        return LoadEmbeddedJsonDictionary("Misaki.NET.Resources.Chinese.pinyin_tone3.json");
    }

    private static Dictionary<string, string> LoadPhraseDictionary()
    {
        return LoadEmbeddedJsonDictionary("Misaki.NET.Resources.Chinese.phrase_tone3.json");
    }

    private static Dictionary<string, string> LoadEmbeddedJsonDictionary(string resourceName)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException(
                $"Embedded resource '{resourceName}' not found. Available resources: " +
                string.Join(", ", assembly.GetManifestResourceNames()));

        using var reader = new StreamReader(stream);
        var json = reader.ReadToEnd();
        return JsonSerializer.Deserialize<Dictionary<string, string>>(json)
            ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Gets the full tone3 pinyin for each character in the word.
    /// For multi-character words, phrase dictionary is checked first.
    /// </summary>
    /// <param name="word">Chinese word or character(s).</param>
    /// <returns>List of tone3 pinyin strings (e.g., "ni3", "hao3"). Empty string for unknown characters.</returns>
    public List<string> GetTone3(string word) => GetTone3(word, null);

    /// <summary>
    /// Gets all tone3 pinyin readings for a single Chinese character (covers polyphones).
    /// First entry is the primary reading from the dictionary; subsequent entries are
    /// alternative readings in dictionary order. Returns an empty list for unknown characters.
    /// </summary>
    /// <param name="ch">A single Chinese character.</param>
    public IReadOnlyList<string> GetAllTone3(char ch)
    {
        if (!SingleCharDict.Value.TryGetValue(ch, out var charValue) || string.IsNullOrEmpty(charValue))
            return Array.Empty<string>();
        var parts = charValue.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length == 0 ? Array.Empty<string>() : parts;
    }

    /// <summary>
    /// Gets the full tone3 pinyin for each character in the word, consulting an
    /// optional override map first. Override values must be space-separated tone3
    /// pinyin (one syllable per Chinese character in the word, same shape as
    /// <c>phrase_tone3.json</c> entries). When the override count does not match
    /// the word's character count the override is ignored and the built-in
    /// dictionaries are used instead.
    /// </summary>
    public List<string> GetTone3(string word, IReadOnlyDictionary<string, string>? overrides)
    {
        if (string.IsNullOrEmpty(word))
            return new List<string>();

        if (overrides != null && overrides.TryGetValue(word, out var ovValue))
        {
            var ovParts = ovValue.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (ovParts.Length == word.Length)
                return new List<string>(ovParts);
            // mismatch — fall through to built-in dicts
        }

        // Try phrase lookup for multi-character words
        if (word.Length > 1 && PhraseDict.Value.TryGetValue(word, out var phraseValue))
        {
            var parts = phraseValue.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            // Phrase dictionary has space-separated pinyin for each character
            var result = new List<string>(parts.Length);
            foreach (var part in parts)
            {
                result.Add(part);
            }
            return result;
        }

        // Fall back to single character lookup
        var charResults = new List<string>(word.Length);
        var singleDict = SingleCharDict.Value;
        foreach (var ch in word)
        {
            if (singleDict.TryGetValue(ch, out var charValue))
            {
                // Take the first pinyin if multiple are available (comma-separated)
                var commaIdx = charValue.IndexOf(',');
                var pinyin = commaIdx >= 0 ? charValue[..commaIdx] : charValue;
                charResults.Add(pinyin);
            }
            else
            {
                charResults.Add(string.Empty);
            }
        }

        return charResults;
    }

    /// <summary>
    /// Gets the initial consonant for each character in the word.
    /// </summary>
    /// <param name="word">Chinese word or character(s).</param>
    /// <returns>List of initial strings (e.g., "zh", "n"). Empty string if no initial.</returns>
    public List<string> GetInitials(string word)
    {
        var tone3List = GetTone3(word);
        var result = new List<string>(tone3List.Count);

        foreach (var pinyin in tone3List)
        {
            result.Add(ExtractInitial(pinyin));
        }

        return result;
    }

    /// <summary>
    /// Gets the final (vowel part + tone number) for each character in the word.
    /// </summary>
    /// <param name="word">Chinese word or character(s).</param>
    /// <returns>List of final strings with tone number (e.g., "ong1", "i3"). Empty string for unknown characters.</returns>
    public List<string> GetFinals(string word)
    {
        var tone3List = GetTone3(word);
        var result = new List<string>(tone3List.Count);

        for (int i = 0; i < tone3List.Count; i++)
        {
            var pinyin = tone3List[i];

            // Special handling for 嗯: final should be 'n2' (pypinyin quirk after v0.44.0)
            if (i < word.Length && word[i] == '嗯')
            {
                result.Add("n2");
                continue;
            }

            result.Add(ExtractFinal(pinyin));
        }

        return result;
    }

    /// <summary>
    /// Gets initials and finals together in a single pass.
    /// Equivalent to calling <see cref="GetInitials"/> and <see cref="GetFinals"/> back-to-back,
    /// but performs the dictionary lookup and pinyin parse only once per character.
    /// </summary>
    public (List<string> Initials, List<string> Finals) GetInitialsAndFinals(string word)
        => GetInitialsAndFinals(word, null);

    /// <summary>
    /// Override-aware variant: an entry in <paramref name="overrides"/> whose key matches the
    /// whole word is preferred over the embedded character/phrase dictionaries. Useful for
    /// user-provided phrase pronunciations (e.g. polyphone disambiguation).
    /// </summary>
    public (List<string> Initials, List<string> Finals) GetInitialsAndFinals(
        string word, IReadOnlyDictionary<string, string>? overrides)
    {
        var tone3List = GetTone3(word, overrides);
        int n = tone3List.Count;
        var initials = new List<string>(n);
        var finals = new List<string>(n);

        for (int i = 0; i < n; i++)
        {
            var pinyin = tone3List[i];
            initials.Add(ExtractInitial(pinyin));

            if (i < word.Length && word[i] == '嗯')
            {
                finals.Add("n2");
            }
            else
            {
                finals.Add(ExtractFinal(pinyin));
            }
        }

        return (initials, finals);
    }

    /// <summary>
    /// Extracts the initial consonant from a tone3 pinyin string.
    /// Mirrors pypinyin strict mode: y and w are NOT initials — they mark zero-consonant
    /// syllables (yun → vn, yao → iao, wo → uo, wu → u, …).
    /// </summary>
    private static string ExtractInitial(string pinyin)
    {
        if (string.IsNullOrEmpty(pinyin))
            return string.Empty;

        var normalized = NormalizeZeroConsonant(pinyin);

        // Try two-character initials first (zh, ch, sh)
        if (normalized.Length >= 2)
        {
            var twoChar = normalized[..2];
            foreach (var initial in TwoCharInitials)
            {
                if (twoChar == initial)
                    return initial;
            }
        }

        // Try single-character initial (strict: y/w excluded — they were absorbed above)
        var oneChar = normalized[..1];
        if (Initials.Contains(oneChar) && oneChar != "y" && oneChar != "w")
            return oneChar;

        // No initial (starts with vowel)
        return string.Empty;
    }

    /// <summary>
    /// Extracts the final (vowel + tone number) from a tone3 pinyin string.
    /// Applies pypinyin-style zero-consonant restoration (yun→vn, wo→uo, ju→jv, …)
    /// plus FINALS_TONE3 abbreviated final expansion (iu→iou, ui→uei, un→uen).
    /// </summary>
    private static string ExtractFinal(string pinyin)
    {
        if (string.IsNullOrEmpty(pinyin))
            return string.Empty;

        var normalized = NormalizeZeroConsonant(pinyin);
        var initial = ExtractInitial(pinyin);

        // initial length in the normalized form: y/w were absorbed, so initial may be ""
        // even when the original started with y/w.
        var skip = string.IsNullOrEmpty(initial) ? 0 : initial.Length;
        var final = normalized[skip..];

        final = ExpandFinal(final);
        return final;
    }

    /// <summary>
    /// Restores the underlying pinyin form per 《汉语拼音方案》 / pypinyin strict mode:
    ///   y-row: yu*→ü*, yi*→i*, y*→i*
    ///   w-row: wu*→u*, w*→u*
    ///   j/q/x + u → j/q/x + ü
    /// Tone digit (if any) is preserved.
    /// </summary>
    private static string NormalizeZeroConsonant(string pinyin)
    {
        if (string.IsNullOrEmpty(pinyin))
            return pinyin;

        // Strip and preserve a trailing tone digit.
        char lastChar = pinyin[^1];
        bool hasTone = char.IsDigit(lastChar);
        string body = hasTone ? pinyin[..^1] : pinyin;
        string tone = hasTone ? lastChar.ToString() : "";

        if (body.Length == 0)
            return pinyin;

        // y-row
        if (body[0] == 'y')
        {
            if (body.Length == 1)
                body = "i";
            else
            {
                char second = body[1];
                if (second == 'u')
                    body = "ü" + body[2..];        // yu → ü, yue → üe, yuan → üan, yun → ün
                else if (second == 'i')
                    body = body[1..];              // yi → i, yin → in, ying → ing
                else
                    body = "i" + body[1..];        // ya → ia, yao → iao, yang → iang, yong → iong, …
            }
        }
        // w-row
        else if (body[0] == 'w')
        {
            if (body.Length == 1)
                body = "u";
            else
            {
                char second = body[1];
                if (second == 'u')
                    body = body[1..];              // wu → u
                else
                    body = "u" + body[1..];        // wa → ua, wo → uo, wai → uai, wei → uei, …
            }
        }
        // j/q/x + u → j/q/x + ü (covers ju, jue, juan, jun, qu, que, quan, qun, xu, xue, xuan, xun)
        else if ((body[0] == 'j' || body[0] == 'q' || body[0] == 'x')
                 && body.Length >= 2 && body[1] == 'u')
        {
            body = body[..1] + "ü" + body[2..];
        }

        return body + tone;
    }

    /// <summary>
    /// Expands abbreviated pinyin finals to their full form (matching pypinyin FINALS_TONE3 behavior).
    /// iu → iou, ui → uei, un → uen. Also ü → v / üe → ve / üan → van / ün → vn.
    /// Assumes the caller has already applied <see cref="NormalizeZeroConsonant"/>, so
    /// y/w-row pinyin never reaches here in its zero-consonant spelling.
    /// </summary>
    private static string ExpandFinal(string final)
    {
        if (string.IsNullOrEmpty(final))
            return final;

        // Extract tone digit if present
        char lastChar = final[^1];
        bool hasTone = char.IsDigit(lastChar);
        string body = hasTone ? final[..^1] : final;
        string tone = hasTone ? lastChar.ToString() : "";

        string expanded = body switch
        {
            "iu" => "iou",
            "ui" => "uei",
            "un" => "uen",
            "ü" => "v",
            "üe" => "ve",
            "üan" => "van",
            "ün" => "vn",
            _ => body
        };

        return expanded + tone;
    }

    /// <summary>
    /// Checks if a word exists in the phrase dictionary.
    /// </summary>
    public bool IsPhrase(string word)
    {
        return word.Length > 1 && PhraseDict.Value.ContainsKey(word);
    }
}
