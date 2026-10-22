using System.Text;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Playback.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies;

/// <summary>
/// Keeps the text scheduled for the current TTS response and detects ASR transcripts that
/// are likely a replay of that text. This is a validation guard, not a replacement for AEC.
/// </summary>
internal sealed class PlaybackTextEchoGuard : IScheduledSpeechSink, Contracts.IPlaybackTextEchoResetPort
{
    private const int MaxRememberedCharacters = 8192;
    private const int MinimumExactMatchCharacters = 6;
    private const int MinimumFuzzyMatchCharacters = 8;
    private const double MinimumContiguousCoverage = 0.72;
    private const double MinimumBigramCoverage = 0.78;

    private readonly object _lock = new();
    private readonly StringBuilder _scheduledSpeech = new();
    private long _turnId;

    public void BeginResponse(long turnId)
    {
        lock (_lock)
        {
            _turnId = turnId;
            _scheduledSpeech.Clear();
        }
    }

    public void AppendScheduledSpeech(long turnId, string text)
    {
        if (string.IsNullOrEmpty(text))
            return;

        string normalized = Normalize(text);
        if (normalized.Length == 0)
            return;

        lock (_lock)
        {
            if (_turnId != turnId)
                return;

            _scheduledSpeech.Append(normalized);
            if (_scheduledSpeech.Length > MaxRememberedCharacters)
            {
                _scheduledSpeech.Remove(
                    0,
                    _scheduledSpeech.Length - MaxRememberedCharacters);
            }
        }
    }

    void IScheduledSpeechSink.OnSpeechScheduled(long turnId, string text)
        => AppendScheduledSpeech(turnId, text);

    public PlaybackTextEchoMatch Match(string recognizedText)
    {
        string recognized = Normalize(recognizedText);
        if (recognized.Length < MinimumExactMatchCharacters)
            return NoMatch("too-short");

        string scheduled;
        lock (_lock)
            scheduled = _scheduledSpeech.ToString();

        if (scheduled.Length < MinimumExactMatchCharacters)
            return NoMatch("no-playback-text");

        if (scheduled.Contains(recognized, StringComparison.Ordinal))
            return new PlaybackTextEchoMatch(true, "exact-substring");

        int longest = LongestCommonSubstringLength(recognized, scheduled);
        double contiguousCoverage = longest / (double)recognized.Length;
        if (longest >= MinimumExactMatchCharacters &&
            contiguousCoverage >= MinimumContiguousCoverage)
        {
            return new PlaybackTextEchoMatch(true, "contiguous-coverage");
        }

        double bigramCoverage = recognized.Length >= MinimumFuzzyMatchCharacters
            ? BigramCoverage(recognized, scheduled)
            : 0;
        bool fuzzyMatch =
            recognized.Length >= MinimumFuzzyMatchCharacters &&
            longest >= 4 &&
            bigramCoverage >= MinimumBigramCoverage;
        return new PlaybackTextEchoMatch(
            fuzzyMatch,
            fuzzyMatch ? "bigram-coverage" : "different-text");
    }

    private static PlaybackTextEchoMatch NoMatch(string reason)
        => new(false, reason);

    private static string Normalize(string text)
    {
        var normalized = new StringBuilder(text.Length);
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (Rune.IsLetterOrDigit(rune))
                normalized.Append(Rune.ToLowerInvariant(rune));
        }

        return normalized.ToString();
    }

    private static int LongestCommonSubstringLength(string left, string right)
    {
        var previous = new int[right.Length + 1];
        var current = new int[right.Length + 1];
        int longest = 0;

        for (int i = 1; i <= left.Length; i++)
        {
            for (int j = 1; j <= right.Length; j++)
            {
                int length = left[i - 1] == right[j - 1]
                    ? previous[j - 1] + 1
                    : 0;
                current[j] = length;
                longest = Math.Max(longest, length);
            }

            (previous, current) = (current, previous);
            Array.Clear(current);
        }

        return longest;
    }

    private static double BigramCoverage(string recognized, string scheduled)
    {
        if (recognized.Length < 2 || scheduled.Length < 2)
            return 0;

        var scheduledBigrams = new HashSet<int>();
        for (int i = 1; i < scheduled.Length; i++)
            scheduledBigrams.Add(Bigram(scheduled[i - 1], scheduled[i]));

        int matched = 0;
        int total = recognized.Length - 1;
        for (int i = 1; i < recognized.Length; i++)
        {
            if (scheduledBigrams.Contains(Bigram(recognized[i - 1], recognized[i])))
                matched++;
        }

        return matched / (double)total;
    }

    private static int Bigram(char first, char second)
        => first << 16 | second;
}
