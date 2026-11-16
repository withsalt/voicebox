namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Converts pinyin syllables to IPA (International Phonetic Alphabet) transcription.
/// Adapted from pinyin-to-ipa Python library.
/// </summary>
public static class PinyinToIpa
{
    #region Initial Mappings

    private static readonly Dictionary<string, string> InitialMapping = new()
    {
        ["b"] = "p", ["c"] = "ʦʰ", ["ch"] = "\uAB67ʰ", ["d"] = "t",
        ["f"] = "f", ["g"] = "k", ["h"] = "x", ["j"] = "ʨ",
        ["k"] = "kʰ", ["l"] = "l", ["m"] = "m", ["n"] = "n",
        ["p"] = "pʰ", ["q"] = "ʨʰ", ["r"] = "ɻ", ["s"] = "s",
        ["sh"] = "ʂ", ["t"] = "tʰ", ["x"] = "ɕ", ["z"] = "ʦ",
        ["zh"] = "\uAB67",
    };

    #endregion

    #region Final Mappings

    // Finals with tone placeholder '0' indicating where tone mark goes
    private static readonly Dictionary<string, string[]> FinalMapping = new()
    {
        ["a"] = ["a0"],
        ["ai"] = ["ai̯0"],
        ["an"] = ["a0", "n"],
        ["ang"] = ["a0", "ŋ"],
        ["ao"] = ["au̯0"],
        ["e"] = ["ɤ0"],
        ["ei"] = ["ei̯0"],
        ["en"] = ["ə0", "n"],
        ["eng"] = ["ə0", "ŋ"],
        ["i"] = ["i0"],
        ["ia"] = ["j", "a0"],
        ["ian"] = ["j", "ɛ0", "n"],
        ["iang"] = ["j", "a0", "ŋ"],
        ["iao"] = ["j", "au̯0"],
        ["ie"] = ["j", "e0"],
        ["in"] = ["i0", "n"],
        ["iou"] = ["j", "ou̯0"],
        ["ing"] = ["i0", "ŋ"],
        ["iong"] = ["j", "ʊ0", "ŋ"],
        ["ong"] = ["ʊ0", "ŋ"],
        ["ou"] = ["ou̯0"],
        ["u"] = ["u0"],
        ["uei"] = ["w", "ei̯0"],
        ["ua"] = ["w", "a0"],
        ["uai"] = ["w", "ai̯0"],
        ["uan"] = ["w", "a0", "n"],
        ["uen"] = ["w", "ə0", "n"],
        ["uang"] = ["w", "a0", "ŋ"],
        ["ueng"] = ["w", "ə0", "ŋ"],
        ["uo"] = ["w", "o0"],
        ["o"] = ["w", "o0"],
        ["ü"] = ["y0"],
        ["üe"] = ["ɥ", "e0"],
        ["üan"] = ["ɥ", "ɛ0", "n"],
        ["ün"] = ["y0", "n"],
    };

    // Special finals for retroflex consonants (zh, ch, sh, r + i)
    private static readonly Dictionary<string, string[]> RetroflexFinalMapping = new()
    {
        ["i"] = ["ɻ̩0"],
    };

    // Special finals for denti-alveolar consonants (z, c, s + i)
    private static readonly Dictionary<string, string[]> DentiAlveolarFinalMapping = new()
    {
        ["i"] = ["ɹ̩0"],
    };

    #endregion

    #region Tone Mapping

    private static readonly Dictionary<char, string> ToneMapping = new()
    {
        ['1'] = "˥",
        ['2'] = "˧˥",
        ['3'] = "˧˩˧",
        ['4'] = "˥˩",
        ['5'] = "",
    };

    #endregion

    #region Syllabic Consonants and Interjections

    private static readonly Dictionary<string, string[]> SyllabicConsonantMapping = new()
    {
        ["hm"] = ["h", "m̩"],
        ["hng"] = ["h", "ŋ̩"],
        ["m"] = ["m̩"],
        ["n"] = ["n̩"],
        ["ng"] = ["ŋ̩"],
    };

    private static readonly Dictionary<string, string[]> InterjectionMapping = new()
    {
        ["io"] = ["j", "ɔ"],
        ["ê"] = ["ɛ"],
        ["er"] = ["ɚ"],
        ["o"] = ["ɔ"],
    };

    private static readonly HashSet<string> RetroflexInitials = ["zh", "ch", "sh", "r"];
    private static readonly HashSet<string> DentiAlveolarInitials = ["z", "c", "s"];

    #endregion

    /// <summary>
    /// Converts a pinyin syllable (with tone number) to IPA.
    /// </summary>
    /// <param name="pinyin">Pinyin syllable with tone number, e.g., "ma1", "zhong1"</param>
    /// <returns>IPA transcription string.</returns>
    public static string Convert(string pinyin)
    {
        if (string.IsNullOrEmpty(pinyin))
            return string.Empty;

        // Extract tone number (last character if digit)
        char toneChar = pinyin[^1];
        string tone;
        string syllable;

        if (char.IsDigit(toneChar) && ToneMapping.ContainsKey(toneChar))
        {
            tone = ToneMapping[toneChar];
            syllable = pinyin[..^1];
        }
        else
        {
            tone = "";
            syllable = pinyin;
        }

        if (string.IsNullOrEmpty(syllable))
            return string.Empty;

        // Check syllabic consonants
        if (SyllabicConsonantMapping.TryGetValue(syllable, out var scParts))
        {
            return BuildIpa(scParts, tone);
        }

        // Extract initial
        string initial = "";
        string final = syllable;

        // Try two-char initials first
        if (syllable.Length >= 2)
        {
            string twoChar = syllable[..2];
            if (InitialMapping.ContainsKey(twoChar))
            {
                initial = twoChar;
                final = syllable[2..];
            }
            else if (syllable.Length >= 1 && InitialMapping.ContainsKey(syllable[..1]))
            {
                initial = syllable[..1];
                final = syllable[1..];
            }
        }
        else if (syllable.Length == 1 && InitialMapping.ContainsKey(syllable))
        {
            initial = syllable;
            final = "";
        }

        // Handle empty final (just initial)
        if (string.IsNullOrEmpty(final))
        {
            // Check interjections
            if (InterjectionMapping.TryGetValue(syllable, out var intParts))
            {
                return BuildIpa(intParts, tone);
            }
            return string.IsNullOrEmpty(initial) ? "" : (InitialMapping[initial] + tone);
        }

        // Get IPA for initial
        string ipaInitial = string.IsNullOrEmpty(initial) ? "" : InitialMapping[initial];

        // Get IPA for final - check special cases first
        string[] finalParts;
        if (RetroflexInitials.Contains(initial) && RetroflexFinalMapping.TryGetValue(final, out var rfParts))
        {
            finalParts = rfParts;
        }
        else if (DentiAlveolarInitials.Contains(initial) && DentiAlveolarFinalMapping.TryGetValue(final, out var dfParts))
        {
            finalParts = dfParts;
        }
        else if (FinalMapping.TryGetValue(final, out var fParts))
        {
            finalParts = fParts;
        }
        else if (InterjectionMapping.TryGetValue(final, out var ijParts))
        {
            finalParts = ijParts;
        }
        else
        {
            // Unknown final - return as-is
            return ipaInitial + final + tone;
        }

        return ipaInitial + BuildIpa(finalParts, tone);
    }

    private static string BuildIpa(string[] parts, string tone)
    {
        var result = new System.Text.StringBuilder();
        foreach (var part in parts)
        {
            if (part.EndsWith('0'))
            {
                // Replace tone placeholder with actual tone
                result.Append(part[..^1]);
                result.Append(tone);
            }
            else
            {
                result.Append(part);
            }
        }
        return result.ToString();
    }

    /// <summary>
    /// Applies retone transformation (converts IPA tone marks to arrow symbols).
    /// </summary>
    public static string Retone(string ipa)
    {
        if (string.IsNullOrEmpty(ipa))
            return ipa;

        ipa = ipa.Replace("˧˩˧", "↓");  // third tone
        ipa = ipa.Replace("˧˥", "↗");   // second tone
        ipa = ipa.Replace("˥˩", "↘");   // fourth tone
        ipa = ipa.Replace("˥", "→");    // first tone
        // Handle syllabic consonant markers
        ipa = ipa.Replace("\u027B\u0329", "ɨ"); // ɻ̩ → ɨ
        ipa = ipa.Replace("\u0279\u0329", "ɨ"); // ɹ̩ → ɨ
        return ipa;
    }
}
