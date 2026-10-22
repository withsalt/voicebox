using System.Text;

namespace Misaki.NET.Lang.English.Neural;

/// <summary>
/// Maps raw IPA emitted by the DeepPhonemizer en_us_cmudict_ipa model to the
/// Misaki IPA inventory used by Kokoro voices, and injects a default stress mark.
///
/// Differences the mapping handles:
///   1. Affricates: t͡ʃ → ʧ, d͡ʒ → ʤ (also tolerates the missing tie bar).
///   2. Diphthongs: eɪ → A, aɪ → I, oʊ → O (US) / əʊ → Q (GB), aʊ → W, ɔɪ → Y.
///   3. R-colored vowel: ɝ → ɜɹ (stressed) / əɹ (unstressed). With no stress info
///      we default to the stressed form.
///   4. Tie bar U+0361 dropped if it appears outside a recognized affricate.
///
/// The DP en_us model emits no stress markers, so we add a primary-stress prefix
/// (Restress later places it before the first vowel inside EnglishLexicon.ApplyStress).
/// </summary>
internal static class IpaToMisakiMapper
{
    private static readonly HashSet<char> MisakiVowels = new("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");
    private const char PrimaryStress = 'ˈ';

    public static string Map(string rawIpa, bool british)
    {
        if (string.IsNullOrEmpty(rawIpa)) return string.Empty;

        var sb = new StringBuilder(rawIpa.Length);
        int i = 0;
        while (i < rawIpa.Length)
        {
            char c = rawIpa[i];

            // Two-char and three-char digraphs first (longest match wins).
            // Affricates with explicit tie bar t͡ʃ / d͡ʒ (3 codepoints).
            if (i + 2 < rawIpa.Length && rawIpa[i + 1] == '͡')
            {
                char next = rawIpa[i + 2];
                if (c == 't' && next == 'ʃ') { sb.Append('ʧ'); i += 3; continue; }
                if (c == 'd' && next == 'ʒ') { sb.Append('ʤ'); i += 3; continue; }
            }
            // Same affricates without the tie bar.
            if (i + 1 < rawIpa.Length)
            {
                char next = rawIpa[i + 1];
                if (c == 't' && next == 'ʃ') { sb.Append('ʧ'); i += 2; continue; }
                if (c == 'd' && next == 'ʒ') { sb.Append('ʤ'); i += 2; continue; }

                // Diphthongs.
                if (c == 'e' && next == 'ɪ') { sb.Append('A'); i += 2; continue; }
                if (c == 'a' && next == 'ɪ') { sb.Append('I'); i += 2; continue; }
                if (c == 'a' && next == 'ʊ') { sb.Append('W'); i += 2; continue; }
                if (c == 'ɔ' && next == 'ɪ') { sb.Append('Y'); i += 2; continue; }
                if (c == 'o' && next == 'ʊ') { sb.Append(british ? 'Q' : 'O'); i += 2; continue; }
                if (c == 'ə' && next == 'ʊ') { sb.Append(british ? 'Q' : 'O'); i += 2; continue; }
            }

            switch (c)
            {
                case 'ɝ':
                    // Rhotacized vowel: stressed form by default (DP gives no stress).
                    sb.Append(british ? "ɜː" : "ɜɹ");
                    break;
                case 'ɚ':
                    sb.Append(british ? "ə" : "əɹ");
                    break;
                case '͡':
                    // Stray tie bar (no affricate match) — drop.
                    break;
                case 'r':
                    // DP may emit 'r' for the English approximant; Misaki uses ɹ.
                    sb.Append('ɹ');
                    break;
                case 'g':
                    // Misaki uses U+0261 LATIN SMALL LETTER SCRIPT G for the voiced stop.
                    sb.Append('ɡ');
                    break;
                default:
                    sb.Append(c);
                    break;
            }
            i++;
        }

        return InjectDefaultStress(sb.ToString());
    }

    /// <summary>
    /// Prepends a primary stress marker so EnglishLexicon.ApplyStress / Restress places
    /// it before the first vowel. Skips empty strings and tokens with no vowel.
    /// </summary>
    private static string InjectDefaultStress(string ps)
    {
        if (string.IsNullOrEmpty(ps)) return ps;
        foreach (var c in ps)
            if (MisakiVowels.Contains(c))
                return PrimaryStress + ps;
        return ps;
    }
}
