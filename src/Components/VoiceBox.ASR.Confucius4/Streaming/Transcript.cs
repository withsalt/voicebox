using System.Text;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Confucius4.Streaming;

internal static class Transcript
{
    public static string Normalize(string raw)
    {
        int stop = raw.IndexOf('|');
        if (stop >= 0) raw = raw[..stop];
        var text = new StringBuilder(raw.Length);
        char previous = '\0';
        const string western = ",.!?;:()";
        const string chinese = "，。！？；：（）";
        foreach (char original in raw)
        {
            if (original == '\uFFFD') continue;
            char c = original;
            int index = western.IndexOf(c);
            if (previous is >= '\u4e00' and <= '\u9fff' && index >= 0) c = chinese[index];
            else if (char.IsAsciiLetterOrDigit(previous) || previous is '\'' or '"')
            {
                index = chinese.IndexOf(c);
                if (index >= 0) c = western[index];
            }
            text.Append(c);
            if (!char.IsWhiteSpace(c)) previous = c;
        }
        return text.ToString();
    }

    public static (string Language, string Text) Parse(string raw, string? forcedLanguage)
    {
        if (forcedLanguage is not null)
            return (forcedLanguage, forcedLanguage == "English" ? raw.TrimEnd() : raw.Trim());
        int tag = raw.IndexOf("<asr_text>", StringComparison.Ordinal);
        if (tag < 0) return ("", raw.StartsWith("language", StringComparison.OrdinalIgnoreCase) ? "" : raw.Trim());
        string language = raw[..tag].Trim();
        if (language.StartsWith("language ", StringComparison.OrdinalIgnoreCase)) language = language[9..].Trim();
        if (language == "None") language = "";
        return (language, raw[(tag + 10)..].Trim());
    }

    public static string Join(string previous, string next)
    {
        if (previous.Length == 0 || next.Length == 0) return previous + next;
        return previous + (char.IsAsciiLetterOrDigit(previous[^1]) && char.IsAsciiLetterOrDigit(next[0]) ? " " : "") + next;
    }

    public static AsrResult Result(string language, string text, string committed) => new()
    {
        Language = language, Text = text, CommittedText = committed,
        UncommittedText = text.StartsWith(committed, StringComparison.Ordinal) ? text[committed.Length..] : "",
    };
}
