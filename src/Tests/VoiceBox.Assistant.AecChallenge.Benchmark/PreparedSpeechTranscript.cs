namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class PreparedSpeechTranscript
{
    public static string ReadThchs30(string transcriptPath)
        => ReadThchs30(transcriptPath, depth: 0);

    public static string ParseWenet(string raw, string utteranceId)
    {
        ArgumentNullException.ThrowIfNull(raw);
        ArgumentException.ThrowIfNullOrWhiteSpace(utteranceId);

        string text = raw.Trim();
        int timingStart = text.IndexOf("[[", StringComparison.Ordinal);
        if (timingStart >= 0)
            text = text[..timingStart];

        text = text.Trim();
        if (text.StartsWith(utteranceId, StringComparison.Ordinal))
            text = text[utteranceId.Length..];

        text = NormalizeChineseText(text);
        if (text.Length == 0)
        {
            throw new InvalidDataException(
                $"WenetSpeech4TTS transcript '{utteranceId}' does not contain reference text.");
        }
        return text;
    }

    public static string ResolvePreparedReference(
        EvaluationDatasetManifest manifest,
        PreparedEndToEndCase testCase,
        PreparedSpeechAsset? speech)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(testCase);

        string raw = testCase.ReferenceText ?? speech?.Text ?? string.Empty;
        if (speech is null)
            return NormalizeChineseText(raw);

        if (speech.Corpus.StartsWith("wenet-", StringComparison.Ordinal))
            return ParseWenet(raw, speech.UtteranceId);

        if (speech.Corpus.StartsWith("thchs30-", StringComparison.Ordinal) &&
            IsThchs30Pointer(raw))
        {
            if (!manifest.SourceRoots.TryGetValue("kwsAsr", out string? kwsRoot) ||
                string.IsNullOrWhiteSpace(kwsRoot))
            {
                throw new InvalidDataException(
                    $"Prepared THCHS30 asset '{speech.Id}' contains a legacy transcript pointer, " +
                    "but the manifest has no kwsAsr source root.");
            }

            string transcriptPath = Path.Combine(
                kwsRoot,
                "THCHS30_2015",
                "data_thchs30",
                "test",
                speech.UtteranceId + ".wav.trn");
            if (!File.Exists(transcriptPath))
            {
                throw new FileNotFoundException(
                    $"Cannot resolve the legacy THCHS30 transcript for '{speech.Id}'.",
                    transcriptPath);
            }
            return ReadThchs30(transcriptPath);
        }

        return NormalizeChineseText(raw);
    }

    public static string NormalizeChineseText(string text)
        => string.Concat(text.Where(c => !char.IsWhiteSpace(c)));

    private static string ReadThchs30(string transcriptPath, int depth)
    {
        if (depth > 2)
        {
            throw new InvalidDataException(
                $"THCHS30 transcript pointer chain is too deep: {transcriptPath}");
        }

        string firstLine = File.ReadLines(transcriptPath)
            .FirstOrDefault(line => !string.IsNullOrWhiteSpace(line))
            ?.Trim()
            ?? string.Empty;
        if (firstLine.Length == 0)
            throw new InvalidDataException($"THCHS30 transcript is empty: {transcriptPath}");

        if (IsThchs30Pointer(firstLine))
        {
            string directory = Path.GetDirectoryName(Path.GetFullPath(transcriptPath))
                ?? throw new InvalidDataException(
                    $"THCHS30 transcript has no parent directory: {transcriptPath}");
            string target = Path.GetFullPath(
                firstLine.Replace('/', Path.DirectorySeparatorChar),
                directory);
            if (!File.Exists(target))
            {
                throw new FileNotFoundException(
                    $"THCHS30 transcript pointer target was not found: {firstLine}",
                    target);
            }
            return ReadThchs30(target, depth + 1);
        }

        return NormalizeChineseText(firstLine);
    }

    private static bool IsThchs30Pointer(string value)
    {
        string text = value.Trim();
        return text.EndsWith(".trn", StringComparison.OrdinalIgnoreCase) &&
               !text.Any(char.IsWhiteSpace);
    }
}
