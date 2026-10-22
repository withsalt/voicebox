using System.Globalization;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal enum DatasetMaterialization
{
    HardLink,
    Copy,
}

internal sealed record PreparedDatasetOptions
{
    public required string AecRoot { get; init; }
    public required string KwsRoot { get; init; }
    public required string ButRirRoot { get; init; }
    public required string ButLibriRoot { get; init; }
    public required string WenetRoot { get; init; }
    public required string OutputRoot { get; init; }
    public int Seed { get; init; } = 20260730;
    public DatasetMaterialization Materialization { get; init; } = DatasetMaterialization.HardLink;
}

internal static class PreparedDatasetBuilder
{
    private const string ConversationCorpus = "aishell3-conversation";
    private const string DailyConversationScenario = "daily-conversation";
    private const string ManualReviewExclusionPolicy = "20260803-v1";

    // Stable corpus IDs rejected during failure-data review. Apply these exclusions after
    // deterministic corpus sampling so every unrelated selected item remains unchanged.
    private static readonly HashSet<string> ManualReviewExcludedAcousticCaseIds = new(StringComparer.Ordinal)
    {
        "device_fullband_mobile-5938a8ab-bffd-4e80-96ce-a2036891fb00_doubletalk",
        "device_fullband_mobile-dcf1e6df-8400-4466-9d21-d88f43c73263_doubletalk",
        "device_fullband_mobile-f2b03727-3ae9-4eea-a64c-ddf9d2b9b76a_doubletalk_with_movement",
        "device_real_doubled-0Od51d4qNUiwVaCKezlZAA_doubletalk",
        "device_real_doubled-Sd5wcbYoZ0yBo2bEbW2NJg_doubletalk",
        "device_real_doubled-UUaPQD7TuEaM0pOXJTJn3Q_doubletalk_with_movement",
        "icassp2022-524a5293-c812-4020-86f8-d8fe7f4e1e9a_doubletalk-with-movement",
    };

    private static readonly HashSet<string> ManualReviewExcludedSpeechAssetIds = new(StringComparer.Ordinal)
    {
        "aishell3-test-SSB05880111",
        "aishell3-test-SSB06070144",
        "wenet-standard-Y0000011994_UvJSJSUC3n0_S00196",
        "xiaoyun-mobile-20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_27",
    };

    // Replacements stay outside the original E2E selection so unaffected case slots retain
    // their IDs and factor assignments. These clips were screened for intelligible daily speech.
    private static readonly IReadOnlyDictionary<string, string> ManualReviewEndToEndReplacementAssetIds =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["aishell3-test-SSB05880111"] = "aishell3-test-SSB09150147",
            ["aishell3-test-SSB06070144"] = "aishell3-test-SSB07840102",
            ["wenet-standard-Y0000011994_UvJSJSUC3n0_S00196"] =
                "wenet-standard-X0000002086_41476701_S00082-S00083",
        };

    // Manually screened for natural conversational content, transcript/audio agreement,
    // distinct speakers, and no overlap with the deterministic AISHELL-3 test slice.
    private static readonly string[] CuratedConversationUtteranceIds =
    [
        "SSB07020058",
        "SSB07110234",
        "SSB07160147",
        "SSB07170106",
        "SSB08090075",
        "SSB08510140",
        "SSB08630058",
        "SSB10010001",
        "SSB10640160",
        "SSB11970024",
        "SSB12740229",
        "SSB14020193",
        "SSB14370016",
        "SSB16300107",
        "SSB18060043",
        "SSB18100086",
    ];

    private static readonly string[] Prompts =
    [
        "正在为你播报今天的日程安排，上午有项目会议，下午需要整理本周的工作记录。",
        "下面继续介绍设备的使用状态，网络连接正常，音频服务已经启动并保持运行。",
        "我正在读取刚才查询到的信息，请稍等片刻，完整结果会按照时间顺序依次播报。",
        "这是用于测试播放期间语音打断的一段中文内容，用户可以在任意位置开始说话。",
        "根据当前设置，系统会持续播放这段提示，并在检测到有效的人声之后进行识别。",
        "现在开始模拟家庭环境中的语音交互，扬声器正在播放，房间中可能存在轻微噪声。",
    ];

    public static bool IsRequested(IReadOnlyList<string> args)
        => args.Contains("--prepare-dataset", StringComparer.Ordinal);

    public static Task<int> RunAsync(string[] args)
    {
        PreparedDatasetOptions options = PreparedDatasetArguments.Parse(args);
        Build(options);
        return Task.FromResult(0);
    }

    private static void Build(PreparedDatasetOptions options)
    {
        ValidateSourceRoot(options.AecRoot, "AEC Challenge");
        ValidateSourceRoot(options.KwsRoot, "KWS/ASR");
        ValidateSourceRoot(options.ButRirRoot, "BUT ReverbDB RIR");
        ValidateSourceRoot(options.ButLibriRoot, "BUT ReverbDB LibriSpeech");
        ValidateSourceRoot(options.WenetRoot, "WenetSpeech4TTS");

        string outputRoot = Path.GetFullPath(options.OutputRoot);
        if (Directory.Exists(outputRoot) &&
            Directory.EnumerateFileSystemEntries(outputRoot).Any())
        {
            throw new InvalidOperationException(
                $"Output directory already exists and is not empty: {outputRoot}");
        }
        Directory.CreateDirectory(outputRoot);

        var materializer = new DatasetMaterializer(outputRoot, options.Materialization);
        IReadOnlyList<EvaluationCase> acousticSource = SelectAcousticCases(options);
        var acoustic = acousticSource
            .Select(c => MaterializeAcoustic(c, materializer))
            .ToArray();
        PreparedSpeechSelection speechSelection =
            SelectSpeechAssets(options, materializer);
        IReadOnlyList<PreparedSpeechAsset> speech = speechSelection.Assets;
        ValidateSpeechAssets(speech);
        (IReadOnlyList<PreparedRirAsset> rirs, IReadOnlyList<PreparedNoiseAsset> roomNoises) =
            SelectButAssets(options, materializer);
        IReadOnlyList<PreparedNoiseAsset> noises = roomNoises
            .Concat(SelectEnvironmentalNoiseAssets(options, materializer))
            .ToArray();
        IReadOnlyList<PreparedLibriSpeechAsset> libriSpeech =
            SelectButLibriSpeech(options, materializer);
        IReadOnlyList<PreparedEndToEndCase> endToEnd =
            BuildEndToEndCases(speechSelection.EndToEndAssets, rirs, noises, options.Seed);
        ValidateEndToEndCases(endToEnd, rirs, noises);
        ValidateManualReviewExclusions(acoustic, speech, endToEnd);

        var manifest = new EvaluationDatasetManifest
        {
            SchemaVersion = EvaluationDatasetManifest.CurrentSchemaVersion,
            DatasetId = "voicebox-assistant-evaluation-v2",
            CreatedUtc = DateTimeOffset.UtcNow,
            Seed = options.Seed,
            Materialization = options.Materialization.ToString().ToLowerInvariant(),
            SourceRoots = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["aecChallenge"] = Path.GetFullPath(options.AecRoot),
                ["kwsAsr"] = Path.GetFullPath(options.KwsRoot),
                ["butRir"] = Path.GetFullPath(options.ButRirRoot),
                ["butLibriSpeech"] = Path.GetFullPath(options.ButLibriRoot),
                ["wenetSpeech4Tts"] = Path.GetFullPath(options.WenetRoot),
            },
            Generator = CreateGeneratorMetadata(options),
            AcousticCases = acoustic,
            SpeechAssets = speech,
            RirAssets = rirs,
            NoiseAssets = noises,
            LibriSpeechAssets = libriSpeech,
            EndToEndCases = endToEnd,
        };
        manifest.Save(outputRoot);
        WriteDatasetReadme(outputRoot, manifest, materializer);

        Console.WriteLine($"Prepared dataset : {outputRoot}");
        Console.WriteLine($"Acoustic cases   : {acoustic.Length}");
        Console.WriteLine($"Speech / RIR     : {speech.Count} / {rirs.Count}");
        Console.WriteLine($"Noise assets     : {noises.Count}");
        Console.WriteLine($"BUT Libri assets: {libriSpeech.Count}");
        Console.WriteLine($"End-to-end cases : {endToEnd.Count}");
        Console.WriteLine($"Logical bytes    : {materializer.LogicalBytes:N0}");
        Console.WriteLine($"Materialization  : {options.Materialization}");
    }

    private static IReadOnlyList<EvaluationCase> SelectAcousticCases(
        PreparedDatasetOptions options)
    {
        var allOptions = new EvaluationOptions
        {
            DatasetRoot = options.AecRoot,
            Suite = EvaluationSuite.All,
            Seed = options.Seed,
        };
        EvaluationCase[] all = DatasetCatalog.Discover(allOptions).ToArray();
        EvaluationCase[] negatives = all
            .Where(c => !c.ExpectedNearEnd)
            .OrderBy(c => c.Id, StringComparer.Ordinal)
            .ToArray();

        var selected = new List<EvaluationCase>(negatives);
        selected.AddRange(SelectSyntheticBySer(
            all.Where(c => c.Corpus == "synthetic"),
            countPerBucket: 24,
            options.Seed));
        selected.AddRange(SelectPerStratum(
            all.Where(c => c.ExpectedNearEnd && c.Corpus != "synthetic"),
            countPerStratum: 30,
            options.Seed));

        var deviceOptions = new EvaluationOptions
        {
            DatasetRoot = options.AecRoot,
            Suite = EvaluationSuite.DeviceSample,
            LimitPerStratum = 24,
            Seed = options.Seed,
        };
        selected.AddRange(DatasetCatalog.Discover(deviceOptions));

        return selected
            .GroupBy(c => c.Id, StringComparer.Ordinal)
            .Select(g => g.First())
            .Where(c => !ManualReviewExcludedAcousticCaseIds.Contains(c.Id))
            .OrderBy(c => c.Corpus, StringComparer.Ordinal)
            .ThenBy(c => c.Scenario, StringComparer.Ordinal)
            .ThenBy(c => c.Id, StringComparer.Ordinal)
            .ToArray();
    }

    private static IEnumerable<EvaluationCase> SelectSyntheticBySer(
        IEnumerable<EvaluationCase> source,
        int countPerBucket,
        int seed)
    {
        return source
            .GroupBy(c => SerBucket(c.SignalToEchoRatioDb ?? 0))
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .SelectMany(g => StableShuffle(g, StableSeed(seed, g.Key)).Take(countPerBucket));
    }

    private static IEnumerable<EvaluationCase> SelectPerStratum(
        IEnumerable<EvaluationCase> source,
        int countPerStratum,
        int seed)
    {
        return source
            .GroupBy(c => $"{c.Corpus}/{c.Scenario}/{c.DeviceClass}")
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .SelectMany(g => StableShuffle(g, StableSeed(seed, g.Key)).Take(countPerStratum));
    }

    private static PreparedAcousticCase MaterializeAcoustic(
        EvaluationCase source,
        DatasetMaterializer materializer)
    {
        string directory = $"phase1/{Safe(source.Corpus)}/{Safe(source.Id)}";
        return new PreparedAcousticCase
        {
            Id = source.Id,
            Corpus = source.Corpus,
            Scenario = source.Scenario,
            ExpectedNearEnd = source.ExpectedNearEnd,
            MicrophonePath = materializer.Add(
                source.MicrophonePath,
                $"{directory}/microphone.wav"),
            RenderPath = materializer.Add(
                source.RenderPath,
                $"{directory}/render.wav"),
            NearEndReferencePath = source.NearEndReferencePath is null
                ? null
                : materializer.Add(
                    source.NearEndReferencePath,
                    $"{directory}/nearend-reference.wav"),
            NearEndScale = source.NearEndScale,
            SignalToEchoRatioDb = source.SignalToEchoRatioDb,
            FarEndNonlinear = source.FarEndNonlinear,
            FarEndNoisy = source.FarEndNoisy,
            NearEndNoisy = source.NearEndNoisy,
            DeviceClass = source.DeviceClass,
            DeviceSession = source.DeviceSession,
            ClusterId = InferAcousticCluster(source),
            Split = "test",
            MicrophoneSha256 = materializer.GetSha256(source.MicrophonePath),
            RenderSha256 = materializer.GetSha256(source.RenderPath),
            NearEndReferenceSha256 = source.NearEndReferencePath is null
                ? null
                : materializer.GetSha256(source.NearEndReferencePath),
        };
    }

    private static PreparedSpeechSelection SelectSpeechAssets(
        PreparedDatasetOptions options,
        DatasetMaterializer materializer)
    {
        var candidates = new List<SpeechCandidate>();
        candidates.AddRange(ReadAishell1(options.KwsRoot));
        candidates.AddRange(ReadAishell3(options.KwsRoot));
        candidates.AddRange(ReadThchs30(options.KwsRoot));

        var selected = new List<SpeechCandidate>();
        selected.AddRange(SelectSpeakers(
            candidates.Where(c => c.Corpus == "aishell1-test"),
            speakerCount: 20,
            utterancesPerSpeaker: 2,
            options.Seed));
        SpeechCandidate[] aishell3Test = SelectSpeakers(
            candidates.Where(c => c.Corpus == "aishell3-test"),
            speakerCount: 48,
            utterancesPerSpeaker: 1,
            options.Seed)
            .ToArray();
        selected.AddRange(aishell3Test);
        selected.AddRange(SelectCuratedConversations(candidates, aishell3Test));

        SpeechCandidate[] thchsClean = StableShuffle(
                candidates.Where(c => c.Corpus == "thchs30-test-clean"),
                StableSeed(options.Seed, "thchs30"))
            .Take(12)
            .ToArray();
        selected.AddRange(thchsClean);
        foreach (SpeechCandidate clean in thchsClean)
        {
            SpeechCandidate? noisy = candidates.FirstOrDefault(
                c => c.Corpus == "thchs30-test-noisy" &&
                     c.UtteranceId == clean.UtteranceId);
            if (noisy is not null)
                selected.Add(noisy);
        }

        selected.AddRange(ReadWenet(
            options.WenetRoot,
            countPerTier: 32,
            options.Seed));

        SpeechCandidate[] ordered = selected
            .OrderBy(candidate => candidate.Corpus, StringComparer.Ordinal)
            .ThenBy(GetSpeechAssetId, StringComparer.Ordinal)
            .ToArray();
        SpeechCandidate[] endToEndCandidates =
        [
            .. SelectForEndToEnd(ordered, "aishell1-test", 16, options.Seed),
            .. SelectForEndToEnd(ordered, "aishell3-test", 16, options.Seed),
            .. SelectForEndToEnd(ordered, "thchs30-test-clean", 8, options.Seed),
            .. SelectForEndToEnd(ordered, "thchs30-test-noisy", 8, options.Seed),
            .. SelectForEndToEnd(ordered, ConversationCorpus, 12, options.Seed),
            .. SelectForEndToEnd(ordered, "wenet-premium", 16, options.Seed),
            .. SelectForEndToEnd(ordered, "wenet-standard", 16, options.Seed),
        ];

        PreparedSpeechAsset[] assets = ordered
            .Where(candidate =>
                !ManualReviewExcludedSpeechAssetIds.Contains(GetSpeechAssetId(candidate)))
            .Select(candidate =>
            {
                string id = GetSpeechAssetId(candidate);
                return new PreparedSpeechAsset
                {
                    Id = id,
                    Corpus = candidate.Corpus,
                    SpeakerId = candidate.SpeakerId,
                    UtteranceId = candidate.UtteranceId,
                    ClusterId = candidate.ClusterId,
                    Text = NormalizeChineseText(candidate.Text),
                    AudioPath = materializer.Add(
                        candidate.Path,
                        $"phase2/speech/{Safe(candidate.Corpus)}/{Safe(id)}.wav"),
                    Noisy = candidate.Noisy,
                    Split = "test",
                    DurationMs = GetWaveDurationMs(candidate.Path),
                    AudioSha256 = materializer.GetSha256(candidate.Path),
                };
            })
            .OrderBy(a => a.Corpus, StringComparer.Ordinal)
            .ThenBy(a => a.Id, StringComparer.Ordinal)
            .ToArray();
        IReadOnlyDictionary<string, PreparedSpeechAsset> assetsById = assets.ToDictionary(
            asset => asset.Id,
            StringComparer.Ordinal);
        PreparedSpeechAsset[] endToEndAssets = endToEndCandidates
            .Select(candidate => assetsById[GetSpeechAssetId(candidate)])
            .ToArray();
        return new PreparedSpeechSelection(assets, endToEndAssets);
    }

    private static string GetSpeechAssetId(SpeechCandidate candidate)
        => $"{candidate.Corpus}-{candidate.UtteranceId}";

    private static IEnumerable<SpeechCandidate> SelectSpeakers(
        IEnumerable<SpeechCandidate> source,
        int speakerCount,
        int utterancesPerSpeaker,
        int seed)
    {
        SpeechCandidate[] values = source.ToArray();
        string[] speakers = StableShuffle(
                values.Select(v => v.SpeakerId).Distinct(StringComparer.Ordinal),
                seed)
            .Take(speakerCount)
            .ToArray();
        foreach (string speaker in speakers)
        {
            foreach (SpeechCandidate candidate in StableShuffle(
                         values.Where(v => v.SpeakerId == speaker),
                         StableSeed(seed, speaker)).Take(utterancesPerSpeaker))
            {
                yield return candidate;
            }
        }
    }

    private static IReadOnlyList<SpeechCandidate> SelectCuratedConversations(
        IEnumerable<SpeechCandidate> source,
        IReadOnlyCollection<SpeechCandidate> aishell3Test)
    {
        IReadOnlyDictionary<string, SpeechCandidate> byUtterance = source
            .Where(candidate => candidate.Corpus == "aishell3-test")
            .ToDictionary(candidate => candidate.UtteranceId, StringComparer.Ordinal);
        HashSet<string> testSpeakers = aishell3Test
            .Select(candidate => candidate.SpeakerId)
            .ToHashSet(StringComparer.Ordinal);
        var selected = new List<SpeechCandidate>(CuratedConversationUtteranceIds.Length);
        foreach (string utteranceId in CuratedConversationUtteranceIds)
        {
            if (!byUtterance.TryGetValue(utteranceId, out SpeechCandidate? candidate))
            {
                throw new InvalidDataException(
                    $"Curated AISHELL-3 conversation '{utteranceId}' was not found.");
            }
            if (testSpeakers.Contains(candidate.SpeakerId))
            {
                throw new InvalidDataException(
                    $"Curated AISHELL-3 speaker '{candidate.SpeakerId}' overlaps the test slice.");
            }
            selected.Add(candidate with { Corpus = ConversationCorpus });
        }
        return selected;
    }

    private static IEnumerable<SpeechCandidate> ReadAishell1(string kwsRoot)
    {
        string root = Path.Combine(kwsRoot, "speech_asr_aishell1_trainsets");
        string csv = Path.Combine(root, "speech_asr_aishell_testsets.csv");
        foreach (string line in File.ReadLines(csv, Encoding.UTF8).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            int comma = line.IndexOf(',');
            if (comma <= 0)
                continue;
            string relative = line[..comma].Replace('/', Path.DirectorySeparatorChar);
            string path = Path.Combine(root, relative);
            string id = Path.GetFileNameWithoutExtension(path);
            string speaker = Path.GetFileName(Path.GetDirectoryName(path)) ?? "unknown";
            yield return new SpeechCandidate(
                "aishell1-test",
                speaker,
                id,
                $"aishell1/{id}",
                line[(comma + 1)..],
                path,
                Noisy: false);
        }
    }

    private static IEnumerable<SpeechCandidate> ReadAishell3(string kwsRoot)
    {
        string root = Path.Combine(kwsRoot, "aishell3_16k", "test");
        foreach (string line in File.ReadLines(Path.Combine(root, "content.txt"), Encoding.UTF8))
        {
            string[] fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (fields.Length < 3)
                continue;
            string file = fields[0];
            string id = Path.GetFileNameWithoutExtension(file);
            string speaker = id.Length >= 7 ? id[..7] : id;
            string text = string.Concat(fields.Skip(1).Where((_, index) => index % 2 == 0));
            yield return new SpeechCandidate(
                "aishell3-test",
                speaker,
                id,
                $"aishell3/{id}",
                text,
                Path.Combine(root, "wav", speaker, file),
                Noisy: false);
        }
    }

    private static IEnumerable<SpeechCandidate> ReadThchs30(string kwsRoot)
    {
        string root = Path.Combine(kwsRoot, "THCHS30_2015");
        string cleanRoot = Path.Combine(root, "data_thchs30", "test");
        string[] noiseKinds = ["cafe", "car", "white"];
        int index = 0;
        foreach (string clean in Directory
                     .EnumerateFiles(cleanRoot, "*.wav")
                     .OrderBy(path => path, StringComparer.Ordinal))
        {
            string id = Path.GetFileNameWithoutExtension(clean);
            string speaker = id.Split('_')[0];
            string transcript = PreparedSpeechTranscript.ReadThchs30(clean + ".trn");
            yield return new SpeechCandidate(
                "thchs30-test-clean",
                speaker,
                id,
                $"thchs30/{id}",
                transcript,
                clean,
                Noisy: false);

            string noise = noiseKinds[index++ % noiseKinds.Length];
            string noisy = Path.Combine(root, "test-noise", "0db", noise, Path.GetFileName(clean));
            if (File.Exists(noisy))
            {
                yield return new SpeechCandidate(
                    "thchs30-test-noisy",
                    speaker,
                    id,
                    $"thchs30/{id}",
                    transcript,
                    noisy,
                    Noisy: true);
            }
        }
    }

    private static IEnumerable<SpeechCandidate> ReadWenet(
        string root,
        int countPerTier,
        int seed)
    {
        foreach (string tier in new[] { "Premium", "Standard" })
        {
            var candidates = new List<(string Wav, string Txt, string Id, string Cluster)>();
            string tierRoot = Path.Combine(root, tier);
            if (!Directory.Exists(tierRoot))
                continue;

            foreach (string wav in Directory.EnumerateFiles(
                         tierRoot,
                         "*.wav",
                         SearchOption.AllDirectories))
            {
                string? wavDirectory = Path.GetDirectoryName(wav);
                string? packageDirectory = wavDirectory is null
                    ? null
                    : Path.GetDirectoryName(wavDirectory);
                if (packageDirectory is null)
                    continue;
                string id = Path.GetFileNameWithoutExtension(wav);
                string txt = Path.Combine(packageDirectory, "txts", id + ".txt");
                if (!File.Exists(txt))
                    continue;
                string cluster = id.Split("_S", 2, StringSplitOptions.None)[0];
                candidates.Add((wav, txt, id, cluster));
            }

            foreach ((string wav, string txt, string id, string cluster) in StableShuffle(
                         candidates,
                         StableSeed(seed, $"wenet-{tier}"))
                     .GroupBy(item => item.Cluster, StringComparer.Ordinal)
                     .Select(group => group.First())
                     .Take(countPerTier))
            {
                string text = PreparedSpeechTranscript.ParseWenet(
                    File.ReadAllText(txt, Encoding.UTF8),
                    id);
                string corpus = $"wenet-{tier.ToLowerInvariant()}";
                yield return new SpeechCandidate(
                    corpus,
                    cluster,
                    id,
                    $"wenet/{cluster}",
                    text,
                    wav,
                    Noisy: tier == "Standard");
            }
        }
    }

    private static (
        IReadOnlyList<PreparedRirAsset> Rirs,
        IReadOnlyList<PreparedNoiseAsset> Noises) SelectButAssets(
        PreparedDatasetOptions options,
        DatasetMaterializer materializer)
    {
        var rirs = new List<PreparedRirAsset>();
        var noises = new List<PreparedNoiseAsset>();
        ButRirSource[] sources = Directory
            .EnumerateFiles(options.ButRirRoot, "*.wav", SearchOption.AllDirectories)
            .Where(path => path.Contains(
                $"{Path.DirectorySeparatorChar}RIR{Path.DirectorySeparatorChar}",
                StringComparison.OrdinalIgnoreCase))
            .Select(path => ParseButRirSource(options.ButRirRoot, path))
            .Where(source => source is not null)
            .Select(source => source!)
            .ToArray();

        foreach (IGrouping<string, ButRirSource> roomGroup in sources
                     .GroupBy(source => source.RoomId, StringComparer.Ordinal)
                     .OrderBy(group => group.Key, StringComparer.Ordinal))
        {
            IGrouping<string, ButRirSource>[] geometries = StableShuffle(
                    roomGroup
                        .GroupBy(source => source.GeometryId, StringComparer.Ordinal)
                        .Where(group => group
                            .Select(source => source.SpeakerSetupId)
                            .Distinct(StringComparer.Ordinal)
                            .Count() >= 2),
                    StableSeed(options.Seed, $"rir/{roomGroup.Key}"))
                .Take(6)
                .ToArray();
            foreach (IGrouping<string, ButRirSource> geometry in geometries)
            {
                ButRirSource[] pair = StableShuffle(
                        geometry
                            .GroupBy(source => source.SpeakerSetupId, StringComparer.Ordinal)
                            .Select(group => group.First()),
                        StableSeed(options.Seed, geometry.Key))
                    .Take(2)
                    .ToArray();
                foreach (ButRirSource source in pair)
                {
                    string id =
                        $"{Safe(source.RoomId)}-{Safe(source.MicrophoneSetupId)}-" +
                        $"{Safe(source.MicrophoneChannelId)}-{Safe(source.SpeakerSetupId)}";
                    rirs.Add(new PreparedRirAsset
                    {
                        Id = id,
                        RoomId = source.RoomId,
                        GeometryId = source.GeometryId,
                        MicrophoneSetupId = source.MicrophoneSetupId,
                        MicrophoneChannelId = source.MicrophoneChannelId,
                        SpeakerSetupId = source.SpeakerSetupId,
                        Path = materializer.Add(source.Path, $"phase2/rir/{id}.wav"),
                        AudioSha256 = materializer.GetSha256(source.Path),
                    });
                }

                string positionDirectory = Path.GetDirectoryName(
                    Path.GetDirectoryName(pair[0].Path)!)!;
                string silenceDirectory = Path.Combine(positionDirectory, "silence");
                string? noise = Directory.Exists(silenceDirectory)
                    ? Directory
                        .EnumerateFiles(silenceDirectory, "*.wav")
                        .OrderBy(path => path, StringComparer.Ordinal)
                        .FirstOrDefault()
                    : null;
                if (noise is null)
                    continue;
                string noiseId =
                    $"{Safe(pair[0].RoomId)}-{Safe(pair[0].MicrophoneSetupId)}-" +
                    $"{Safe(pair[0].MicrophoneChannelId)}-noise";
                noises.Add(new PreparedNoiseAsset
                {
                    Id = noiseId,
                    RoomId = pair[0].RoomId,
                    Corpus = "but-reverbdb",
                    Kind = "room-ambience",
                    GeometryId = pair[0].GeometryId,
                    Path = materializer.Add(noise, $"phase2/noise/{noiseId}.wav"),
                    AudioSha256 = materializer.GetSha256(noise),
                });
            }
        }
        return (rirs, noises);
    }

    private static IReadOnlyList<PreparedNoiseAsset> SelectEnvironmentalNoiseAssets(
        PreparedDatasetOptions options,
        DatasetMaterializer materializer)
    {
        var selected = new List<(string Path, string Corpus, string Kind)>();
        foreach ((string relativeRoot, string corpus, string kind, int count) in new[]
                 {
                     (Path.Combine("musan", "music"), "musan", "music", 12),
                     (Path.Combine("musan", "noise"), "musan", "noise", 12),
                     (
                         Path.Combine(
                             "ICASSP_2021_DNS_Challenge",
                             "datasets",
                             "datasets",
                             "noise"),
                         "dns-challenge",
                         "noise",
                         12),
                 })
        {
            string root = Path.Combine(options.KwsRoot, relativeRoot);
            if (!Directory.Exists(root))
                continue;
            selected.AddRange(StableShuffle(
                    Directory.EnumerateFiles(root, "*.wav", SearchOption.AllDirectories),
                    StableSeed(options.Seed, $"{corpus}/{kind}"))
                .Take(count)
                .Select(path => (path, corpus, kind)));
        }

        return selected
            .Select((source, index) =>
            {
                string id =
                    $"{source.Corpus}-{source.Kind}-{index:D2}-" +
                    $"{Safe(Path.GetFileNameWithoutExtension(source.Path))}";
                return new PreparedNoiseAsset
                {
                    Id = id,
                    RoomId = "external",
                    Corpus = source.Corpus,
                    Kind = source.Kind,
                    GeometryId = null,
                    Path = materializer.Add(
                        source.Path,
                        $"phase2/noise/{Safe(source.Corpus)}/{id}.wav"),
                    AudioSha256 = materializer.GetSha256(source.Path),
                };
            })
            .ToArray();
    }

    private static IReadOnlyList<PreparedLibriSpeechAsset> SelectButLibriSpeech(
        PreparedDatasetOptions options,
        DatasetMaterializer materializer)
    {
        var result = new List<PreparedLibriSpeechAsset>();
        foreach (string source in Directory
                     .EnumerateFiles(options.ButLibriRoot, "*.wav", SearchOption.AllDirectories)
                     .OrderBy(path => path, StringComparer.Ordinal)
                     .Where((_, index) => index % 100 == 0)
                     .Take(24))
        {
            string room = GetRoomId(source);
            string utterance = Path.GetFileNameWithoutExtension(source);
            string id = $"but-libri-{result.Count:D2}-{utterance}";
            result.Add(new PreparedLibriSpeechAsset
            {
                Id = id,
                RoomId = room,
                UtteranceId = utterance,
                Path = materializer.Add(source, $"phase1/but-librispeech/{id}.wav"),
                AudioSha256 = materializer.GetSha256(source),
            });
        }
        return result;
    }

    private static string GetRoomId(string path)
        => Path.GetFileName(
            path.Split(
                [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
                StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault(part => part.StartsWith("VUT_", StringComparison.OrdinalIgnoreCase))
            ?? Path.GetDirectoryName(path)
            ?? "unknown");

    private static ButRirSource? ParseButRirSource(string root, string path)
    {
        string[] parts = Path.GetRelativePath(root, path).Split(
            [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
            StringSplitOptions.RemoveEmptyEntries);
        int rirIndex = Array.FindIndex(
            parts,
            part => part.Equals("RIR", StringComparison.OrdinalIgnoreCase));
        if (rirIndex < 4)
            return null;
        string room = parts[0];
        string microphone = parts[1];
        string speaker = parts[2];
        string channel = parts[3];
        return new ButRirSource(
            path,
            room,
            $"{room}/{microphone}/{channel}",
            microphone,
            channel,
            speaker);
    }

    private static IReadOnlyList<PreparedEndToEndCase> BuildEndToEndCases(
        IReadOnlyList<PreparedSpeechAsset> positive,
        IReadOnlyList<PreparedRirAsset> rirs,
        IReadOnlyList<PreparedNoiseAsset> noises,
        int seed)
    {
        if (rirs.Count == 0)
            throw new InvalidDataException("No BUT RIR assets were selected.");

        var result = new List<PreparedEndToEndCase>();
        for (int i = 0; i < positive.Count; i++)
        {
            PreparedSpeechAsset asset = positive[i];
            result.Add(CreateEndToEndCase(
                $"e2e-{i:D3}-{asset.Id}",
                asset,
                rirs,
                noises,
                i,
                seed));
        }
        for (int i = 0; i < 72; i++)
        {
            result.Add(CreateEndToEndCase(
                $"e2e-echo-only-{i:D2}",
                speech: null,
                rirs,
                noises,
                positive.Count + i,
                seed));
        }
        return result;
    }

    private static IReadOnlyList<SpeechCandidate> SelectForEndToEnd(
        IEnumerable<SpeechCandidate> speech,
        string corpus,
        int count,
        int seed)
    {
        SpeechCandidate[] ranked = StableShuffle(
                speech.Where(s => s.Corpus == corpus),
                StableSeed(seed, corpus))
            .GroupBy(s => s.ClusterId, StringComparer.Ordinal)
            .Select(g => g.First())
            .ToArray();
        if (ranked.Length < count)
        {
            throw new InvalidDataException(
                $"Corpus '{corpus}' has only {ranked.Length} independent E2E candidates; {count} required.");
        }

        SpeechCandidate[] selected = ranked.Take(count).ToArray();
        HashSet<string> assignedIds = selected
            .Where(candidate =>
                !ManualReviewExcludedSpeechAssetIds.Contains(GetSpeechAssetId(candidate)))
            .Select(GetSpeechAssetId)
            .ToHashSet(StringComparer.Ordinal);
        for (int i = 0; i < selected.Length; i++)
        {
            string excludedId = GetSpeechAssetId(selected[i]);
            if (!ManualReviewExcludedSpeechAssetIds.Contains(excludedId))
                continue;
            if (!ManualReviewEndToEndReplacementAssetIds.TryGetValue(
                    excludedId,
                    out string? replacementId))
            {
                throw new InvalidDataException(
                    $"E2E replacement for manually excluded asset '{excludedId}' is not configured.");
            }
            SpeechCandidate? replacement = ranked
                .Skip(count)
                .SingleOrDefault(candidate => GetSpeechAssetId(candidate) == replacementId);
            if (replacement is null ||
                ManualReviewExcludedSpeechAssetIds.Contains(replacementId) ||
                !assignedIds.Add(replacementId))
            {
                throw new InvalidDataException(
                    $"E2E replacement '{replacementId}' for '{excludedId}' is unavailable, excluded, or duplicated.");
            }
            selected[i] = replacement;
        }
        return selected;
    }

    private static PreparedEndToEndCase CreateEndToEndCase(
        string id,
        PreparedSpeechAsset? speech,
        IReadOnlyList<PreparedRirAsset> rirs,
        IReadOnlyList<PreparedNoiseAsset> noises,
        int index,
        int seed)
    {
        IGrouping<string, PreparedRirAsset>[] geometryGroups = rirs
            .GroupBy(rir => rir.GeometryId, StringComparer.Ordinal)
            .Where(group => group.Count() >= 2)
            .OrderBy(group => group.Key, StringComparer.Ordinal)
            .ToArray();
        if (geometryGroups.Length == 0)
        {
            throw new InvalidDataException(
                "End-to-end cases require at least one geometry with two RIRs.");
        }

        string scenario = speech switch
        {
            null => (index % 4) switch
            {
                0 => "tts-echo-only",
                1 => "tts-echo-with-room-noise",
                2 => "tts-echo-with-music",
                _ => "tts-echo-with-environment-noise",
            },
            { Corpus: ConversationCorpus } => DailyConversationScenario,
            { Corpus: "wenet-premium" or "wenet-standard" } => "natural-query",
            { Noisy: true } => "near-speech-with-source-noise",
            _ when index % 11 == 0 => "near-tts-text-overlap",
            _ when index % 7 == 0 => "startup-guard-overlap",
            _ => "valid-command",
        };
        int[] onsets = scenario == "startup-guard-overlap"
            ? [200, 350, 450]
            : [700, 900, 1400, 2200, 3000];
        double[] sers = [-12, -6, 0, 6, 12];
        double[] snrs = [8, 15, 25, 35];
        int[] delays = [40, 80, 120, 180];
        int[] drifts = [-80, 0, 80];
        int[] callbacks = [160, 256, 480];
        IGrouping<string, PreparedRirAsset> geometry = Pick(
            geometryGroups,
            seed,
            id,
            "geometry");
        PreparedRirAsset[] geometryRirs = geometry
            .OrderBy(rir => rir.Id, StringComparer.Ordinal)
            .ToArray();
        PreparedRirAsset echoRir = Pick(geometryRirs, seed, id, "echo-rir");
        PreparedRirAsset? nearRir = speech is null
            ? null
            : geometryRirs.First(rir => rir.Id != echoRir.Id);
        PreparedNoiseAsset? matchingNoise = noises.FirstOrDefault(
            noise => noise.GeometryId == geometry.Key);
        PreparedNoiseAsset[] musicNoises = noises
            .Where(noise => noise.Kind == "music")
            .ToArray();
        PreparedNoiseAsset[] externalNoises = noises
            .Where(noise => noise.GeometryId is null && noise.Kind != "music")
            .ToArray();
        PreparedNoiseAsset? selectedNoise = scenario switch
        {
            "tts-echo-with-room-noise" => matchingNoise,
            "tts-echo-with-music" when musicNoises.Length > 0 =>
                Pick(musicNoises, seed, id, "music-noise"),
            "tts-echo-with-environment-noise" when externalNoises.Length > 0 =>
                Pick(externalNoises, seed, id, "environment-noise"),
            _ when speech is not null &&
                   Pick([false, true, true], seed, id, "noise") =>
                Pick(
                    noises.Where(noise =>
                            noise.GeometryId is null ||
                            noise.GeometryId == geometry.Key)
                        .ToArray(),
                    seed,
                    id,
                    "positive-noise"),
            _ => null,
        };
        int onset = Pick(onsets, seed, id, "onset");
        bool expected = speech is not null;
        string prompt = scenario == "near-tts-text-overlap" && speech is not null
            ? BuildOverlapPrompt(speech.Text)
            : Pick(Prompts, seed, id, "prompt");
        int? renderGapStartMs = index % 11 == 0
            ? Pick([600, 1200, 1800], seed, id, "render-gap-start")
            : null;
        return new PreparedEndToEndCase
        {
            Id = id,
            ExpectedNearEnd = expected,
            SpeechAssetId = speech?.Id,
            PromptText = prompt,
            EchoRirId = echoRir.Id,
            NearRirId = nearRir?.Id,
            NoiseAssetId = selectedNoise?.Id,
            NearEndOnsetMs = onset,
            SignalToEchoRatioDb = Pick(sers, seed, id, "ser"),
            SignalToNoiseRatioDb = Pick(snrs, seed, id, "snr"),
            EchoDelayMs = Pick(delays, seed, id, "echo-delay"),
            ClockDriftPpm = Pick(drifts, seed, id, "clock-drift"),
            FarEndNonlinear = Pick([false, false, false, true], seed, id, "nonlinear"),
            CallbackSamples = Pick(callbacks, seed, id, "callback"),
            ClusterId = speech?.ClusterId ??
                        $"echo/{geometry.Key}/prompt-{Array.IndexOf(Prompts, prompt):D2}",
            Scenario = scenario,
            Split = "test",
            Suites = BuildSuites(id, seed),
            GeometryId = geometry.Key,
            SpeakerPolicy = "any-speaker",
            ReferenceText = speech?.Text,
            ExpectHeard = expected,
            ExpectDuck = expected,
            ExpectAccepted = expected,
            ExpectPlaybackStop = expected,
            ExpectPlaybackResume = false,
            OracleSpeechIntervals = speech is null
                ? []
                : [new OracleSpeechInterval(onset, onset + speech.DurationMs)],
            RenderGapStartMs = renderGapStartMs,
            RenderGapDurationMs = renderGapStartMs.HasValue
                ? Pick([20, 40, 80], seed, id, "render-gap-duration")
                : 0,
        };
    }

    private static IReadOnlyList<string> BuildSuites(string caseId, int seed)
    {
        var suites = new List<string> { "nightly-e2e" };
        if (Pick([false, true], seed, caseId, "suite-release"))
            suites.Add("release-e2e");
        if (Pick([false, false, false, false, true], seed, caseId, "suite-pr"))
            suites.Add("pr-e2e");
        if (Pick(
                Enumerable.Repeat(false, 16).Append(true).ToArray(),
                seed,
                caseId,
                "suite-smoke"))
        {
            suites.Add("smoke");
        }
        return suites;
    }

    private static string BuildOverlapPrompt(string referenceText)
    {
        string normalized = NormalizeChineseText(referenceText);
        string overlap = normalized[..Math.Min(10, normalized.Length)];
        return $"刚才提到的内容是{overlap}，我会继续说明后面的信息和相关安排。";
    }

    private static void ValidateEndToEndCases(
        IReadOnlyList<PreparedEndToEndCase> cases,
        IReadOnlyList<PreparedRirAsset> rirs,
        IReadOnlyList<PreparedNoiseAsset> noises)
    {
        int dailyConversationCount = cases.Count(testCase =>
            testCase.Scenario == DailyConversationScenario);
        if (dailyConversationCount != 12 ||
            cases.Any(testCase => testCase.Scenario == "wake-word-interrupt"))
        {
            throw new InvalidDataException(
                "End-to-end cases must contain 12 daily conversations and no wake-word cases.");
        }

        IReadOnlyDictionary<string, PreparedRirAsset> rirById = rirs.ToDictionary(
            item => item.Id,
            StringComparer.Ordinal);
        IReadOnlyDictionary<string, PreparedNoiseAsset> noiseById = noises.ToDictionary(
            item => item.Id,
            StringComparer.Ordinal);
        foreach (PreparedEndToEndCase testCase in cases)
        {
            PreparedRirAsset echo = rirById[testCase.EchoRirId];
            if (echo.GeometryId != testCase.GeometryId)
            {
                throw new InvalidDataException(
                    $"Case '{testCase.Id}' echo RIR does not match its geometry.");
            }
            if (testCase.NearRirId is string nearId)
            {
                PreparedRirAsset near = rirById[nearId];
                if (near.GeometryId != testCase.GeometryId ||
                    near.Id == echo.Id)
                {
                    throw new InvalidDataException(
                        $"Case '{testCase.Id}' has an invalid near/echo RIR pair.");
                }
            }
            if (testCase.NoiseAssetId is string noiseId &&
                noiseById[noiseId].GeometryId is string noiseGeometry &&
                noiseGeometry != testCase.GeometryId)
            {
                throw new InvalidDataException(
                    $"Case '{testCase.Id}' noise does not match its RIR geometry.");
            }
            if (testCase.OracleExpectHeard &&
                (testCase.SpeechAssetId is null ||
                 testCase.OracleSpeechIntervals.Count == 0))
            {
                throw new InvalidDataException(
                    $"Case '{testCase.Id}' lacks positive speech truth.");
            }
        }

        bool onsetIsDecoupled = cases
            .Where(testCase => testCase.OracleExpectHeard)
            .GroupBy(testCase => testCase.NearEndOnsetMs)
            .Any(group => group
                .Select(testCase => testCase.SignalToNoiseRatioDb)
                .Distinct()
                .Count() > 1);
        bool callbackIsDecoupled = cases
            .GroupBy(testCase => testCase.CallbackSamples)
            .Any(group => group
                .Select(testCase => testCase.ClockDriftPpm)
                .Distinct()
                .Count() > 1);
        if (!onsetIsDecoupled || !callbackIsDecoupled)
        {
            throw new InvalidDataException(
                "End-to-end factor assignment is confounded; independent seeds were not effective.");
        }
    }

    internal static void ValidateSpeechAssets(
        IReadOnlyList<PreparedSpeechAsset> speech)
    {
        PreparedSpeechAsset[] wakeWordAssets = speech
            .Where(asset =>
                asset.Corpus.Contains("xiaoyun", StringComparison.OrdinalIgnoreCase) ||
                asset.Corpus.Contains("wake-word", StringComparison.OrdinalIgnoreCase) ||
                NormalizeChineseText(asset.Text) == "小云小云")
            .ToArray();
        if (wakeWordAssets.Length != 0)
        {
            throw new InvalidDataException(
                $"Wake-word speech assets are not allowed: {string.Join(", ", wakeWordAssets.Select(asset => asset.Id))}");
        }

        PreparedSpeechAsset[] conversations = speech
            .Where(asset => asset.Corpus == ConversationCorpus)
            .ToArray();
        if (conversations.Length != CuratedConversationUtteranceIds.Length ||
            conversations.Select(asset => asset.SpeakerId).Distinct(StringComparer.Ordinal).Count() !=
            conversations.Length ||
            conversations.Any(asset =>
                !CuratedConversationUtteranceIds.Contains(
                    asset.UtteranceId,
                    StringComparer.Ordinal)))
        {
            throw new InvalidDataException(
                "The curated daily-conversation slice is incomplete, duplicated, or unexpected.");
        }
    }

    private static void ValidateManualReviewExclusions(
        IReadOnlyList<PreparedAcousticCase> acoustic,
        IReadOnlyList<PreparedSpeechAsset> speech,
        IReadOnlyList<PreparedEndToEndCase> endToEnd)
    {
        string[] violations =
        [
            .. acoustic
                .Where(testCase => ManualReviewExcludedAcousticCaseIds.Contains(testCase.Id))
                .Select(testCase => $"acoustic:{testCase.Id}"),
            .. speech
                .Where(asset => ManualReviewExcludedSpeechAssetIds.Contains(asset.Id))
                .Select(asset => $"speech:{asset.Id}"),
            .. endToEnd
                .Where(testCase => testCase.SpeechAssetId is not null &&
                    ManualReviewExcludedSpeechAssetIds.Contains(testCase.SpeechAssetId))
                .Select(testCase => $"e2e:{testCase.Id}"),
        ];
        if (violations.Length != 0)
        {
            throw new InvalidDataException(
                $"Manual-review exclusions leaked into the prepared dataset: {string.Join(", ", violations)}");
        }
    }

    private static T Pick<T>(
        IReadOnlyList<T> values,
        int seed,
        string caseId,
        string factor)
    {
        if (values.Count == 0)
            throw new ArgumentException("Cannot select from an empty factor.", nameof(values));
        uint hash = unchecked((uint)StableSeed(seed, $"{factor}/{caseId}"));
        return values[(int)(hash % (uint)values.Count)];
    }

    private static string InferAcousticCluster(EvaluationCase source)
    {
        if (!string.IsNullOrWhiteSpace(source.DeviceSession))
            return $"{source.Corpus}/{source.DeviceSession}";

        string id = source.Id;
        string[] suffixes =
        [
            "_doubletalk_with_movement",
            "_farend_singletalk_with_movement",
            "_doubletalk",
            "_farend_singletalk",
        ];
        foreach (string suffix in suffixes)
        {
            int index = id.LastIndexOf(suffix, StringComparison.OrdinalIgnoreCase);
            if (index > 0)
                return $"{source.Corpus}/{id[..index]}";
        }
        return $"{source.Corpus}/{id}";
    }

    private static IEnumerable<T> StableShuffle<T>(IEnumerable<T> source, int seed)
    {
        T[] values = source.ToArray();
        var random = new Random(seed);
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
        return values;
    }

    private static IEnumerable<T> EvenlySpaced<T>(IReadOnlyList<T> source, int count)
    {
        if (source.Count <= count)
            return source;
        return Enumerable.Range(0, count)
            .Select(i => source[(int)Math.Round(
                i * (source.Count - 1) / (double)(count - 1),
                MidpointRounding.AwayFromZero)]);
    }

    private static string NormalizeChineseText(string text)
        => PreparedSpeechTranscript.NormalizeChineseText(text);

    private static double GetWaveDurationMs(string path)
    {
        WaveAudio audio = WaveAudioLoader.LoadMono(
            path,
            EvaluationOptions.TargetSampleRate);
        return audio.Samples.Length * 1000.0 / EvaluationOptions.TargetSampleRate;
    }

    private static string SerBucket(double value)
        => value switch
        {
            <= -6 => "very-low",
            <= -1 => "low",
            <= 4 => "medium",
            _ => "high",
        };

    private static int StableSeed(int seed, string value)
    {
        unchecked
        {
            int hash = seed;
            foreach (char c in value)
                hash = hash * 31 + c;
            return hash;
        }
    }

    private static string Safe(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (char c in value)
        {
            builder.Append(char.IsAsciiLetterOrDigit(c) || c is '-' or '_'
                ? c
                : '_');
        }
        return builder.ToString();
    }

    private static void ValidateSourceRoot(string root, string name)
    {
        if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root))
            throw new DirectoryNotFoundException($"{name} root not found: {root}");
    }

    private static DatasetGeneratorMetadata CreateGeneratorMetadata(
        PreparedDatasetOptions options)
    {
        string commit = ReadGit("rev-parse", "HEAD") ?? "unknown";
        bool dirty = !string.IsNullOrWhiteSpace(
            ReadGit("status", "--porcelain", "--untracked-files=no"));
        return new DatasetGeneratorMetadata
        {
            Commit = commit,
            Dirty = dirty,
            Options = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["seed"] = options.Seed.ToString(CultureInfo.InvariantCulture),
                ["materialization"] = options.Materialization.ToString().ToLowerInvariant(),
                ["rirGeometriesPerRoom"] = "6",
                ["rirSpeakersPerGeometry"] = "2",
                ["wenetAssetsPerTier"] = "32",
                ["conversationSelection"] = "curated-aishell3-v1",
                ["wakeWordPolicy"] = "excluded",
                ["manualReviewExclusionPolicy"] = ManualReviewExclusionPolicy,
                ["factorAssignment"] = "independent-stable-seeds",
            },
        };
    }

    private static string? ReadGit(params string[] arguments)
    {
        try
        {
            var startInfo = new ProcessStartInfo("git")
            {
                WorkingDirectory = Environment.CurrentDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (string argument in arguments)
                startInfo.ArgumentList.Add(argument);
            using Process? process = Process.Start(startInfo);
            if (process is null)
                return null;
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return process.ExitCode == 0 ? output.Trim() : null;
        }
        catch
        {
            return null;
        }
    }

    private static void WriteDatasetReadme(
        string root,
        EvaluationDatasetManifest manifest,
        DatasetMaterializer materializer)
    {
        double negativeHours = manifest.AcousticCases
            .Where(c => !c.ExpectedNearEnd)
            .Sum(c => materializer.TryGetWaveDurationHours(c.MicrophonePath));
        File.WriteAllText(
            Path.Combine(root, "README.md"),
            $"""
            # VoiceBox Assistant evaluation dataset

            Dataset id: `{manifest.DatasetId}`

            This is a deterministic, evaluation-only subset. Files are `{manifest.Materialization}`
            materializations of the source corpora. Statistical independence is defined by each
            manifest item's `clusterId`; acoustic variants and callback-size variants must not be
            counted as independent observations.

            - Phase 1 acoustic cases: {manifest.AcousticCases.Count}
            - Phase 1 negative exposure: {negativeHours.ToString("F3", CultureInfo.InvariantCulture)} hours
            - Phase 2 speech assets: {manifest.SpeechAssets.Count}
            - Curated daily-conversation assets: {manifest.SpeechAssets.Count(asset => asset.Corpus == ConversationCorpus)}
            - Wake-word assets: excluded
            - Manual-review exclusions: {ManualReviewExcludedAcousticCaseIds.Count} acoustic / {ManualReviewExcludedSpeechAssetIds.Count} speech assets
            - Phase 2 RIR/noise assets: {manifest.RirAssets.Count}/{manifest.NoiseAssets.Count}
            - BUT LibriSpeech-only assets (not scored without original render): {manifest.LibriSpeechAssets.Count}
            - Phase 2 end-to-end cases: {manifest.EndToEndCases.Count}

            The source datasets retain their original licenses. Consult the source roots recorded in
            `manifest.json` before redistributing this subset.
            """,
            Encoding.UTF8);
    }

    private sealed record SpeechCandidate(
        string Corpus,
        string SpeakerId,
        string UtteranceId,
        string ClusterId,
        string Text,
        string Path,
        bool Noisy);

    private sealed record PreparedSpeechSelection(
        IReadOnlyList<PreparedSpeechAsset> Assets,
        IReadOnlyList<PreparedSpeechAsset> EndToEndAssets);

    private sealed record ButRirSource(
        string Path,
        string RoomId,
        string GeometryId,
        string MicrophoneSetupId,
        string MicrophoneChannelId,
        string SpeakerSetupId);
}

internal sealed class DatasetMaterializer
{
    private readonly string _root;
    private readonly DatasetMaterialization _mode;
    private readonly Dictionary<string, string> _sourceToRelative =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _sourceHashes =
        new(StringComparer.OrdinalIgnoreCase);

    public DatasetMaterializer(string root, DatasetMaterialization mode)
    {
        _root = Path.GetFullPath(root);
        _mode = mode;
    }

    public long LogicalBytes { get; private set; }

    public string Add(string source, string relativePath)
    {
        source = Path.GetFullPath(source);
        if (!File.Exists(source))
            throw new FileNotFoundException("Selected source asset was not found.", source);
        if (_sourceToRelative.TryGetValue(source, out string? existing))
            return existing;

        relativePath = relativePath.Replace('\\', '/');
        string destination = EvaluationDatasetManifest.ResolvePath(_root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        if (File.Exists(destination))
            throw new IOException($"Prepared dataset path collision: {destination}");

        if (_mode == DatasetMaterialization.HardLink)
            CreateHardLink(destination, source);
        else
            File.Copy(source, destination, overwrite: false);

        LogicalBytes = checked(LogicalBytes + new FileInfo(source).Length);
        _sourceToRelative[source] = relativePath;
        return relativePath;
    }

    public string GetSha256(string source)
    {
        source = Path.GetFullPath(source);
        if (_sourceHashes.TryGetValue(source, out string? existing))
            return existing;
        using FileStream stream = File.OpenRead(source);
        string hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        _sourceHashes[source] = hash;
        return hash;
    }

    private static void CreateHardLink(string destination, string source)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException(
                "Hard-link materialization currently requires Windows; use --materialization copy.");
        if (!CreateHardLinkWindows(destination, source, IntPtr.Zero))
        {
            int error = Marshal.GetLastWin32Error();
            throw new IOException(
                $"Could not create hard link '{destination}' -> '{source}' (Win32 error {error}).");
        }
    }

    [DllImport(
        "Kernel32.dll",
        EntryPoint = "CreateHardLinkW",
        CharSet = CharSet.Unicode,
        SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateHardLinkWindows(
        string newFileName,
        string existingFileName,
        IntPtr securityAttributes);

    public double TryGetWaveDurationHours(string relativePath)
    {
        try
        {
            WaveAudio audio = WaveAudioLoader.LoadMono(
                EvaluationDatasetManifest.ResolvePath(_root, relativePath),
                EvaluationOptions.TargetSampleRate);
            return audio.Samples.Length /
                   (double)EvaluationOptions.TargetSampleRate /
                   3600;
        }
        catch
        {
            return 0;
        }
    }
}

internal static class PreparedDatasetArguments
{
    public static PreparedDatasetOptions Parse(string[] args)
    {
        string? aec = Environment.GetEnvironmentVariable("VOICEBOX_AEC_CHALLENGE");
        string? kws = Environment.GetEnvironmentVariable("VOICEBOX_KWS_DATASETS");
        string? rir = Environment.GetEnvironmentVariable("VOICEBOX_BUT_RIR");
        string? libri = Environment.GetEnvironmentVariable("VOICEBOX_BUT_LIBRISPEECH");
        string? wenet = Environment.GetEnvironmentVariable("VOICEBOX_WENET_SPEECH4TTS");
        string output = string.Empty;
        int seed = 20260730;
        DatasetMaterialization materialization = DatasetMaterialization.HardLink;

        for (int i = 0; i < args.Length; i++)
        {
            string argument = args[i];
            switch (argument)
            {
                case "--prepare-dataset":
                    break;
                case "--aec-root":
                    aec = Required(args, ref i, argument);
                    break;
                case "--kws-root":
                    kws = Required(args, ref i, argument);
                    break;
                case "--but-rir-root":
                    rir = Required(args, ref i, argument);
                    break;
                case "--but-libri-root":
                    libri = Required(args, ref i, argument);
                    break;
                case "--wenet-root":
                    wenet = Required(args, ref i, argument);
                    break;
                case "--prepared-output":
                    output = Required(args, ref i, argument);
                    break;
                case "--seed":
                    seed = int.Parse(Required(args, ref i, argument), CultureInfo.InvariantCulture);
                    break;
                case "--materialization":
                    materialization = Required(args, ref i, argument).ToLowerInvariant() switch
                    {
                        "hardlink" => DatasetMaterialization.HardLink,
                        "copy" => DatasetMaterialization.Copy,
                        string value => throw new ArgumentException(
                            $"Unknown materialization '{value}', expected hardlink|copy."),
                    };
                    break;
                case "-h":
                case "--help":
                    break;
                default:
                    throw new ArgumentException(
                        $"Unknown prepare-dataset argument: {argument}");
            }
        }

        if (string.IsNullOrWhiteSpace(output))
            throw new ArgumentException("--prepared-output is required.");
        return new PreparedDatasetOptions
        {
            AecRoot = aec ?? string.Empty,
            KwsRoot = kws ?? string.Empty,
            ButRirRoot = rir ?? string.Empty,
            ButLibriRoot = libri ?? string.Empty,
            WenetRoot = wenet ?? string.Empty,
            OutputRoot = output,
            Seed = seed,
            Materialization = materialization,
        };
    }

    private static string Required(string[] args, ref int index, string flag)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"Missing value for {flag}.");
        return args[++index];
    }
}
