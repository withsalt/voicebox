using System.Text;
using System.Text.RegularExpressions;
using WeTextProcessing.NET;

namespace VoiceBox.TTS.Hojo.Internal;

internal sealed partial class HojoTextNormalizer(bool enabled) : IDisposable
{
    private WetextProcessor? _chinese;
    private WetextProcessor? _english;
    [GeneratedRegex(@"<zh_[^<>]+>")]
    private static partial Regex PinyinTokens();

    public string Normalize(string text, string language)
    {
        text = text.Trim();
        if (!enabled || text.Length == 0) return text;
        bool chinese = language == "zh" || (language == "auto" && text.Any(c => c is >= '\u4e00' and <= '\u9fff'));
        var processor = chinese ? _chinese ??= new(WetextLanguage.Chinese) : _english ??= new(WetextLanguage.English);
        var builder = new StringBuilder(text.Length);
        int start = 0;
        foreach (Match match in PinyinTokens().Matches(text))
        {
            if (match.Index > start) builder.Append(processor.Normalize(text[start..match.Index]));
            builder.Append(match.Value);
            start = match.Index + match.Length;
        }
        if (start < text.Length) builder.Append(processor.Normalize(text[start..]));
        return builder.ToString();
    }
    public void Dispose() { _chinese?.Dispose(); _english?.Dispose(); }
}
