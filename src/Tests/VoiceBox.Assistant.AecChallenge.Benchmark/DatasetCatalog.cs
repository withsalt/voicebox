using System.Globalization;
using System.Text;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class DatasetCatalog
{
    public const string WindowsPcDeviceClass = "windows_pc";
    public const string AndroidMobileDeviceClass = "android_mobile";
    public const int DefaultDeviceSampleLimitPerStratum = 50;

    public static IReadOnlyList<string> ExpectedDeviceClasses { get; } =
        [WindowsPcDeviceClass, AndroidMobileDeviceClass];

    public static IReadOnlyList<EvaluationCase> Discover(EvaluationOptions options)
    {
        string root = Path.GetFullPath(options.DatasetRoot);
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"AEC Challenge root not found: {root}");

        options.SkippedIncompletePairs = 0;
        if (options.Suite == EvaluationSuite.DeviceSample &&
            options.LimitPerStratum == 0)
        {
            options.LimitPerStratum = DefaultDeviceSampleLimitPerStratum;
        }

        var cases = new List<EvaluationCase>();
        int effectiveLimit = options.LimitPerStratum;
        switch (options.Suite)
        {
            case EvaluationSuite.Smoke:
                cases.AddRange(DiscoverSynthetic(root, testSplitOnly: true));
                cases.AddRange(DiscoverInitialRealTest(root));
                effectiveLimit = options.LimitPerStratum > 0 ? options.LimitPerStratum : 2;
                break;

            case EvaluationSuite.Standard:
                cases.AddRange(DiscoverSynthetic(root, testSplitOnly: true));
                cases.AddRange(DiscoverInitialRealTest(root));
                break;

            case EvaluationSuite.SyntheticTest:
                cases.AddRange(DiscoverSynthetic(root, testSplitOnly: true));
                break;

            case EvaluationSuite.RealTest:
                cases.AddRange(DiscoverInitialRealTest(root));
                break;

            case EvaluationSuite.Icassp2022:
                cases.AddRange(DiscoverIcassp2022(root));
                break;

            case EvaluationSuite.DeviceSample:
                cases.AddRange(DiscoverDeviceRecordings(root, options));
                break;

            case EvaluationSuite.PreparedSmoke:
                cases.AddRange(DiscoverPrepared(root));
                effectiveLimit = options.LimitPerStratum > 0 ? options.LimitPerStratum : 2;
                break;

            case EvaluationSuite.PreparedPhase1:
                cases.AddRange(DiscoverPrepared(root));
                break;

            case EvaluationSuite.All:
                cases.AddRange(DiscoverSynthetic(root, testSplitOnly: true));
                cases.AddRange(DiscoverInitialRealTest(root));
                cases.AddRange(DiscoverIcassp2022(root));
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(options.Suite), options.Suite, "Unknown evaluation suite.");
        }

        if (!string.IsNullOrWhiteSpace(options.CaseId))
        {
            cases = cases
                .Where(testCase => string.Equals(
                    testCase.Id,
                    options.CaseId,
                    StringComparison.OrdinalIgnoreCase))
                .ToList();
        }
        else if (effectiveLimit > 0)
        {
            cases = SamplePerStratum(cases, effectiveLimit, options.Seed).ToList();
        }

        if (cases.Count == 0)
        {
            if (!string.IsNullOrWhiteSpace(options.CaseId))
            {
                throw new InvalidDataException(
                    $"No case matched --case-id '{options.CaseId}' in suite {options.Suite}.");
            }
            throw new InvalidDataException($"No evaluation cases found for suite {options.Suite} under {root}.");
        }

        return cases;
    }

    private static IEnumerable<EvaluationCase> DiscoverPrepared(string root)
    {
        EvaluationDatasetManifest manifest = EvaluationDatasetManifest.Load(root);
        foreach (PreparedAcousticCase item in manifest.AcousticCases)
        {
            yield return new EvaluationCase(
                Id: item.Id,
                Corpus: item.Corpus,
                Scenario: item.Scenario,
                ExpectedNearEnd: item.ExpectedNearEnd,
                MicrophonePath: EvaluationDatasetManifest.ResolvePath(root, item.MicrophonePath),
                RenderPath: EvaluationDatasetManifest.ResolvePath(root, item.RenderPath),
                NearEndReferencePath: item.NearEndReferencePath is null
                    ? null
                    : EvaluationDatasetManifest.ResolvePath(root, item.NearEndReferencePath),
                NearEndScale: item.NearEndScale,
                SignalToEchoRatioDb: item.SignalToEchoRatioDb,
                FarEndNonlinear: item.FarEndNonlinear,
                FarEndNoisy: item.FarEndNoisy,
                NearEndNoisy: item.NearEndNoisy,
                DeviceClass: item.DeviceClass,
                DeviceSession: item.DeviceSession,
                ClusterId: item.ClusterId);
        }
    }

    private static IEnumerable<EvaluationCase> DiscoverDeviceRecordings(
        string root,
        EvaluationOptions options)
    {
        DeviceCorpus[] corpora =
        [
            new(
                "device_real",
                WindowsPcDeviceClass,
                [Path.Combine(root, "datasets", "real")]),
            new(
                "device_real_doubled",
                WindowsPcDeviceClass,
                [Path.Combine(root, "datasets", "real_doubled")]),
            new(
                "device_fullband",
                WindowsPcDeviceClass,
                [
                    Path.Combine(root, "datasets", "fullband"),
                    Path.Combine(root, "datasets", "real_doubled_fullband"),
                ]),
            new(
                "device_fullband_mobile",
                AndroidMobileDeviceClass,
                [
                    Path.Combine(root, "datasets", "fullband_mobile"),
                    Path.Combine(root, "datasets", "real_doubled_fullband_mobile"),
                ]),
        ];

        foreach (DeviceCorpus corpus in corpora)
        {
            string? directory = corpus.CandidateDirectories
                .FirstOrDefault(ContainsDeviceMicrophoneFiles);
            if (directory is null)
                continue;

            foreach (string microphone in Directory
                         .EnumerateFiles(directory, "*.wav", SearchOption.AllDirectories)
                         .OrderBy(path => path, StringComparer.Ordinal))
            {
                if (!TryClassifyDeviceMicrophonePath(
                        microphone,
                        out string scenario,
                        out bool expectedNearEnd,
                        out string session))
                {
                    continue;
                }

                string render = RenderPathForMicrophone(microphone);
                if (!File.Exists(render))
                {
                    options.SkippedIncompletePairs++;
                    continue;
                }
                yield return new EvaluationCase(
                    Id: $"{corpus.Corpus}-{session}_{scenario}",
                    Corpus: corpus.Corpus,
                    Scenario: scenario,
                    ExpectedNearEnd: expectedNearEnd,
                    MicrophonePath: microphone,
                    RenderPath: render,
                    DeviceClass: corpus.DeviceClass,
                    DeviceSession: session);
            }
        }
    }

    internal static bool TryClassifyDeviceMicrophoneName(
        string name,
        out string scenario,
        out bool expectedNearEnd,
        out string session)
    {
        string stem;
        if (name.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase))
            stem = name[..^"_mic_c.wav".Length];
        else if (name.EndsWith("_mic.wav", StringComparison.OrdinalIgnoreCase))
            stem = name[..^"_mic.wav".Length];
        else
        {
            scenario = string.Empty;
            expectedNearEnd = false;
            session = string.Empty;
            return false;
        }

        string normalizedStem = stem.Replace('-', '_');
        (string Suffix, string Scenario, bool ExpectedNearEnd)[] patterns =
        [
            ("_doubletalk_with_movement", "doubletalk_with_movement", true),
            ("_farend_singletalk_with_movement", "farend_singletalk_with_movement", false),
            ("_doubletalk", "doubletalk", true),
            ("_farend_singletalk", "farend_singletalk", false),
        ];
        foreach ((string suffix, string classifiedScenario, bool classifiedExpectedNearEnd) in patterns)
        {
            if (!normalizedStem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                continue;

            session = stem[..^suffix.Length];
            if (session.Length == 0)
                break;
            scenario = classifiedScenario;
            expectedNearEnd = classifiedExpectedNearEnd;
            return true;
        }

        scenario = string.Empty;
        expectedNearEnd = false;
        session = string.Empty;
        return false;
    }

    private static bool TryClassifyDeviceMicrophonePath(
        string path,
        out string scenario,
        out bool expectedNearEnd,
        out string session)
    {
        string name = Path.GetFileName(path);
        if (TryClassifyDeviceMicrophoneName(name, out scenario, out expectedNearEnd, out session))
            return true;

        string parent = (Path.GetFileName(Path.GetDirectoryName(path)) ?? string.Empty)
            .Replace('-', '_')
            .ToLowerInvariant();
        if (parent is not (
                "doubletalk" or
                "doubletalk_with_movement" or
                "farend_singletalk" or
                "farend_singletalk_with_movement"))
        {
            scenario = string.Empty;
            expectedNearEnd = false;
            session = string.Empty;
            return false;
        }

        string microphoneSuffix = name.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase)
            ? "_mic_c.wav"
            : name.EndsWith("_mic.wav", StringComparison.OrdinalIgnoreCase)
                ? "_mic.wav"
                : string.Empty;
        if (microphoneSuffix.Length == 0)
        {
            scenario = string.Empty;
            expectedNearEnd = false;
            session = string.Empty;
            return false;
        }

        session = name[..^microphoneSuffix.Length];
        if (session.Length == 0)
        {
            scenario = string.Empty;
            expectedNearEnd = false;
            return false;
        }

        expectedNearEnd = parent.StartsWith("doubletalk", StringComparison.Ordinal);
        scenario = parent.EndsWith("_with_movement", StringComparison.Ordinal) ||
                   !session.Contains("movement", StringComparison.OrdinalIgnoreCase)
            ? parent
            : $"{parent}_with_movement";
        return true;
    }

    private static bool ContainsDeviceMicrophoneFiles(string directory)
        => Directory.Exists(directory) &&
           Directory.EnumerateFiles(directory, "*.wav", SearchOption.AllDirectories)
               .Any(path =>
                   path.EndsWith("_mic.wav", StringComparison.OrdinalIgnoreCase) ||
                   path.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase));

    private static string RenderPathForMicrophone(string microphone)
    {
        if (microphone.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase))
            return microphone[..^"_mic_c.wav".Length] + "_lpb.wav";
        if (microphone.EndsWith("_mic.wav", StringComparison.OrdinalIgnoreCase))
            return microphone[..^"_mic.wav".Length] + "_lpb.wav";
        throw new ArgumentException("Expected a microphone path ending in _mic.wav or _mic_c.wav.", nameof(microphone));
    }

    private static IEnumerable<EvaluationCase> DiscoverSynthetic(string root, bool testSplitOnly)
    {
        string dataset = Path.Combine(root, "datasets", "synthetic");
        string metadataPath = Path.Combine(dataset, "meta.csv");
        if (!File.Exists(metadataPath))
            throw new FileNotFoundException("Synthetic metadata not found.", metadataPath);

        using var reader = new StreamReader(metadataPath, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        string? headerLine = reader.ReadLine();
        if (headerLine is null)
            yield break;

        string[] headers = ParseCsvLine(headerLine).ToArray();
        var headerMap = headers
            .Select((name, index) => (name, index))
            .ToDictionary(x => x.name, x => x.index, StringComparer.OrdinalIgnoreCase);

        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] fields = ParseCsvLine(line).ToArray();
            string split = Field(fields, headerMap, "split");
            if (testSplitOnly && !string.Equals(split, "test", StringComparison.OrdinalIgnoreCase))
                continue;

            string id = Field(fields, headerMap, "fileid");
            string mic = Path.Combine(dataset, "nearend_mic_signal", $"nearend_mic_fileid_{id}.wav");
            string render = Path.Combine(dataset, "farend_speech", $"farend_speech_fileid_{id}.wav");
            string nearEnd = Path.Combine(dataset, "nearend_speech", $"nearend_speech_fileid_{id}.wav");
            EnsurePairExists(mic, render, nearEnd);

            yield return new EvaluationCase(
                Id: $"synthetic-{id}",
                Corpus: "synthetic",
                Scenario: "doubletalk",
                ExpectedNearEnd: true,
                MicrophonePath: mic,
                RenderPath: render,
                NearEndReferencePath: nearEnd,
                NearEndScale: ParseDouble(Field(fields, headerMap, "nearend_scale"), 1),
                SignalToEchoRatioDb: ParseNullableDouble(Field(fields, headerMap, "ser")),
                FarEndNonlinear: ParseNullableBool(Field(fields, headerMap, "is_farend_nonlinear")),
                FarEndNoisy: ParseNullableBool(Field(fields, headerMap, "is_farend_noisy")),
                NearEndNoisy: ParseNullableBool(Field(fields, headerMap, "is_nearend_noisy")));
        }
    }

    private static IEnumerable<EvaluationCase> DiscoverInitialRealTest(string root)
    {
        foreach (string quality in new[] { "clean", "noisy" })
        {
            string directory = Path.Combine(root, "datasets", "test_set", quality);
            if (!Directory.Exists(directory))
                continue;

            foreach (string mic in Directory.EnumerateFiles(directory, "*.wav").OrderBy(x => x, StringComparer.Ordinal))
            {
                string name = Path.GetFileName(mic);
                if (!TryClassifyInitialTestName(name, out string scenario, out bool expectedNearEnd))
                    continue;

                string renderName = name.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase)
                    ? name[..^"_mic_c.wav".Length] + "_lpb.wav"
                    : name[..^"_mic.wav".Length] + "_lpb.wav";
                string render = Path.Combine(directory, renderName);
                EnsurePairExists(mic, render);

                string id = renderName[..^"_lpb.wav".Length];
                yield return new EvaluationCase(
                    Id: $"{quality}-{id}",
                    Corpus: $"test_set_{quality}",
                    Scenario: scenario,
                    ExpectedNearEnd: expectedNearEnd,
                    MicrophonePath: mic,
                    RenderPath: render);
            }
        }
    }

    private static IEnumerable<EvaluationCase> DiscoverIcassp2022(string root)
    {
        string baseDirectory = Path.Combine(root, "datasets", "test_set_icassp2022");
        foreach ((string folder, bool expectedNearEnd) in new[]
                 {
                     ("doubletalk", true),
                     ("farend-singletalk", false),
                 })
        {
            string directory = Path.Combine(baseDirectory, folder);
            if (!Directory.Exists(directory))
                continue;

            foreach (string mic in Directory.EnumerateFiles(directory, "*_mic.wav").OrderBy(x => x, StringComparer.Ordinal))
            {
                string render = mic[..^"_mic.wav".Length] + "_lpb.wav";
                EnsurePairExists(mic, render);
                string stem = Path.GetFileNameWithoutExtension(mic);
                string scenario = stem.Contains("movement", StringComparison.OrdinalIgnoreCase)
                    ? $"{folder}_with_movement"
                    : folder.Replace('-', '_');

                yield return new EvaluationCase(
                    Id: $"icassp2022-{stem[..^"_mic".Length]}",
                    Corpus: "test_set_icassp2022",
                    Scenario: scenario,
                    ExpectedNearEnd: expectedNearEnd,
                    MicrophonePath: mic,
                    RenderPath: render);
            }
        }
    }

    private static bool TryClassifyInitialTestName(string name, out string scenario, out bool expectedNearEnd)
    {
        string stem = name;
        if (stem.EndsWith("_mic_c.wav", StringComparison.OrdinalIgnoreCase))
            stem = stem[..^"_mic_c.wav".Length];
        else if (stem.EndsWith("_mic.wav", StringComparison.OrdinalIgnoreCase))
            stem = stem[..^"_mic.wav".Length];
        else
        {
            scenario = string.Empty;
            expectedNearEnd = false;
            return false;
        }

        if (stem.EndsWith("_doubletalk_with_movement", StringComparison.OrdinalIgnoreCase))
        {
            scenario = "doubletalk_with_movement";
            expectedNearEnd = true;
            return true;
        }
        if (stem.EndsWith("_doubletalk", StringComparison.OrdinalIgnoreCase))
        {
            scenario = "doubletalk";
            expectedNearEnd = true;
            return true;
        }
        if (stem.EndsWith("_farend_singletalk_with_movement", StringComparison.OrdinalIgnoreCase))
        {
            scenario = "farend_singletalk_with_movement";
            expectedNearEnd = false;
            return true;
        }
        if (stem.EndsWith("_farend_singletalk", StringComparison.OrdinalIgnoreCase))
        {
            scenario = "farend_singletalk";
            expectedNearEnd = false;
            return true;
        }

        scenario = string.Empty;
        expectedNearEnd = false;
        return false;
    }

    private static IReadOnlyList<EvaluationCase> SamplePerStratum(
        IEnumerable<EvaluationCase> cases,
        int count,
        int seed)
    {
        return cases
            .GroupBy(c => (c.Corpus, c.Scenario, c.ExpectedNearEnd))
            .OrderBy(g => g.Key.Corpus, StringComparer.Ordinal)
            .ThenBy(g => g.Key.Scenario, StringComparer.Ordinal)
            .ThenBy(g => g.Key.ExpectedNearEnd)
            .SelectMany(g => Shuffle(g, StableSeed(seed, g.Key.Corpus, g.Key.Scenario, g.Key.ExpectedNearEnd)).Take(count))
            .ToArray();
    }

    private static IEnumerable<EvaluationCase> Shuffle(IEnumerable<EvaluationCase> source, int seed)
    {
        var values = source.OrderBy(c => c.Id, StringComparer.Ordinal).ToArray();
        var random = new Random(seed);
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
        return values;
    }

    private static int StableSeed(int seed, string corpus, string scenario, bool expectedNearEnd)
    {
        unchecked
        {
            int hash = seed;
            foreach (char c in corpus)
                hash = hash * 31 + c;
            foreach (char c in scenario)
                hash = hash * 31 + c;
            return hash * 31 + (expectedNearEnd ? 1 : 0);
        }
    }

    private static IEnumerable<string> ParseCsvLine(string line)
    {
        var value = new StringBuilder();
        bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                if (quoted && i + 1 < line.Length && line[i + 1] == '"')
                {
                    value.Append('"');
                    i++;
                }
                else
                {
                    quoted = !quoted;
                }
            }
            else if (c == ',' && !quoted)
            {
                yield return value.ToString();
                value.Clear();
            }
            else
            {
                value.Append(c);
            }
        }
        yield return value.ToString();
    }

    private static string Field(string[] fields, IReadOnlyDictionary<string, int> map, string name)
    {
        if (!map.TryGetValue(name, out int index) || index >= fields.Length)
            throw new InvalidDataException($"Synthetic meta.csv is missing column '{name}'.");
        return fields[index];
    }

    private static double ParseDouble(string value, double fallback)
        => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double result)
            ? result
            : fallback;

    private static double? ParseNullableDouble(string value)
        => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double result)
            ? result
            : null;

    private static bool? ParseNullableBool(string value)
        => value.Trim() switch
        {
            "1" => true,
            "0" => false,
            _ => null,
        };

    private static void EnsurePairExists(params string[] paths)
    {
        foreach (string path in paths)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("AEC Challenge pair is incomplete.", path);
        }
    }

    private sealed record DeviceCorpus(
        string Corpus,
        string DeviceClass,
        IReadOnlyList<string> CandidateDirectories);
}
