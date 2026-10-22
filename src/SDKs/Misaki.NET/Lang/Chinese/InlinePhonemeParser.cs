using System.Text;
using System.Text.RegularExpressions;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Parses inline pinyin annotations of the form <c>[词|yi1 hang2]</c> embedded in input text.
/// Returns the cleaned text (annotations replaced by their bare word) and a per-call override
/// map keyed by the bare word. Used by <see cref="ChineseG2P"/> for ad-hoc polyphone control
/// without touching the embedded dictionaries.
/// </summary>
public static class InlinePhonemeParser
{
    private static readonly Regex AnnotationRegex = new(
        @"\[([^\[\]\|]+)\|([^\[\]]+)\]",
        RegexOptions.Compiled);

    /// <summary>
    /// Parses <paramref name="text"/>, replacing every <c>[word|tone3 pinyin]</c> annotation
    /// with the bare word and accumulating word→pinyin entries into <paramref name="overrides"/>.
    /// Returns the cleaned text. Annotations whose pinyin token count does not match the
    /// word's character count are silently kept as-is (so they fall through to the regular pipeline).
    /// </summary>
    public static string Parse(string text, out Dictionary<string, string> overrides)
    {
        overrides = new Dictionary<string, string>();
        if (string.IsNullOrEmpty(text) || text.IndexOf('[') < 0)
            return text;

        var sb = new StringBuilder(text.Length);
        int cursor = 0;

        foreach (Match m in AnnotationRegex.Matches(text))
        {
            sb.Append(text, cursor, m.Index - cursor);

            var word = m.Groups[1].Value;
            var pinyin = m.Groups[2].Value.Trim();
            var parts = pinyin.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length == word.Length && parts.Length > 0)
            {
                overrides[word] = string.Join(' ', parts);
                sb.Append(word);
            }
            else
            {
                sb.Append(text, m.Index, m.Length);
            }

            cursor = m.Index + m.Length;
        }

        sb.Append(text, cursor, text.Length - cursor);
        return sb.ToString();
    }
}
