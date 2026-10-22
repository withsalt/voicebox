using System.Diagnostics;
using System.Globalization;
using VoiceBox.Core.Onnx;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class EndToEndProgram
{
    public static bool IsRequested(IReadOnlyList<string> args)
    {
        if (args.Contains("--e2e", StringComparer.Ordinal))
            return true;
        for (int i = 0; i + 1 < args.Count; i++)
        {
            if (args[i] == "--mode" &&
                string.Equals(args[i + 1], "e2e", StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    public static async Task<int> RunAsync(
        string[] args,
        CancellationToken cancellationToken = default)
    {
        EndToEndOptions options = Parse(args);
        EvaluationDatasetManifest manifest = EvaluationDatasetManifest.Load(options.DatasetRoot);
        IEnumerable<PreparedEndToEndCase> selected = manifest.EndToEndCases;
        if (manifest.SchemaVersion == EvaluationDatasetManifest.CurrentSchemaVersion &&
            !options.Suite.Equals("all", StringComparison.OrdinalIgnoreCase))
        {
            selected = selected.Where(testCase => testCase.Suites.Contains(
                options.Suite,
                StringComparer.OrdinalIgnoreCase));
        }
        if (!string.IsNullOrWhiteSpace(options.CaseId))
        {
            selected = selected.Where(c => c.Id.Equals(
                options.CaseId,
                StringComparison.OrdinalIgnoreCase));
        }
        if (options.Limit is > 0)
            selected = selected.Take(options.Limit.Value);
        PreparedEndToEndCase[] cases = selected.ToArray();
        if (cases.Length == 0)
            throw new InvalidDataException("No end-to-end cases matched the requested selection.");

        Console.WriteLine("VoiceBox Assistant Chinese TTS-AEC-ASR end-to-end benchmark");
        Console.WriteLine($"Dataset       : {Path.GetFullPath(options.DatasetRoot)}");
        Console.WriteLine($"Cases         : {cases.Length}");
        Console.WriteLine($"Suite         : {options.Suite}");
        Console.WriteLine($"ASR model     : {options.AsrModelPath}");
        Console.WriteLine($"TTS model     : {options.TtsModelPath}");
        Console.WriteLine($"ASR/VAD EP    : {options.ExecutionProvider}");
        Console.WriteLine($"TTS EP        : {options.TtsExecutionProvider}");

        var results = new List<EndToEndCaseResult>(cases.Length);
        var stopwatch = Stopwatch.StartNew();
        try
        {
            using var evaluator = new EndToEndEvaluator(options);
            for (int i = 0; i < cases.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                EndToEndCaseResult result = await evaluator
                    .EvaluateAsync(cases[i], cancellationToken)
                    .ConfigureAwait(false);
                results.Add(result);
                if (options.Verbose || result.Error is not null ||
                    (result.ExpectHeard && !result.Heard) ||
                    (result.ExpectAccepted && !result.Accepted) ||
                    result.EarlyFalseTrigger ||
                    (!result.ExpectAccepted && result.Accepted))
                {
                    PrintCase(result);
                }
                Console.WriteLine($"Progress      : {i + 1}/{cases.Length}");
            }
        }
        finally
        {
            stopwatch.Stop();
        }

        EndToEndSummary overall = EndToEndMetrics.Aggregate(results);
        IReadOnlyList<QualityGate> gates = EndToEndMetrics.EvaluateGates(overall);
        double audioSeconds = results.Sum(c => c.AudioDurationMs) / 1000.0;
        var report = new EndToEndReport
        {
            // 3.0: the level model, latency reference, device output model and gate set all changed.
            // Reports from 2.2 and earlier are not comparable case-for-case with these and must not
            // be strict-compared against them; the 2.2 baseline stays valid as its own frozen
            // identity.
            SchemaVersion = "3.0",
            CreatedUtc = DateTimeOffset.UtcNow,
            DatasetRoot = Path.GetFullPath(options.DatasetRoot),
            DatasetId = manifest.DatasetId,
            AsrModelPath = Path.GetFullPath(options.AsrModelPath),
            TtsModelPath = Path.GetFullPath(options.TtsModelPath),
            ExecutionProvider = options.ExecutionProvider.ToString(),
            TtsExecutionProvider = options.TtsExecutionProvider.ToString(),
            WallTimeSeconds = stopwatch.Elapsed.TotalSeconds,
            AudioTimeSeconds = audioSeconds,
            RealTimeFactor = audioSeconds > 0
                ? stopwatch.Elapsed.TotalSeconds / audioSeconds
                : 0,
            Overall = overall,
            Slices = EndToEndMetrics.BuildSlices(results),
            Gates = gates,
            Cases = results,
        };
        (string jsonPath, string csvPath) = EndToEndReportWriter.Write(
            report,
            options.OutputPath);
        PrintSummary(report, jsonPath, csvPath);
        return options.EnforceGates && gates.Any(g => g.Status != GateStatus.Passed)
            ? 1
            : 0;
    }

    private static EndToEndOptions Parse(string[] args)
    {
        string dataset = Environment.GetEnvironmentVariable("VOICEBOX_EVAL_DATASET") ?? string.Empty;
        string asr = Environment.GetEnvironmentVariable("VOICEBOX_QWEN3_ASR_MODEL") ?? string.Empty;
        string tts = Environment.GetEnvironmentVariable("VOICEBOX_VOXCPM_TTS_MODEL") ??
                     Environment.GetEnvironmentVariable("VOXCPM_MODEL") ??
                     string.Empty;
        string? voice = null;
        string? vadDirectory =
            Environment.GetEnvironmentVariable("VOICEBOX_FIRERED_VAD_MODEL_DIR");
        string suite = "release-e2e";
        string? caseId = null;
        string? output = null;
        int? limit = null;
        int timeout = 120;
        bool verbose = false;
        bool enforceGates = false;
        OnnxExecutionProvider provider = OnnxExecutionProvider.Cuda;

        for (int i = 0; i < args.Length; i++)
        {
            string argument = args[i];
            switch (argument)
            {
                case "--e2e":
                case "--mode":
                    if (argument == "--mode")
                    {
                        string mode = Required(args, ref i, argument);
                        if (!string.Equals(mode, "e2e", StringComparison.OrdinalIgnoreCase))
                            throw new ArgumentException($"Expected --mode e2e, got '{mode}'.");
                    }
                    break;
                case "--dataset":
                case "--prepared-dataset":
                    dataset = Required(args, ref i, argument);
                    break;
                case "--asr-model":
                    asr = Required(args, ref i, argument);
                    break;
                case "--tts-model":
                    tts = Required(args, ref i, argument);
                    break;
                case "--voice":
                    voice = Required(args, ref i, argument);
                    break;
                case "--vad-model-dir":
                    vadDirectory = Required(args, ref i, argument);
                    break;
                case "--suite":
                    suite = Required(args, ref i, argument);
                    break;
                case "--provider":
                    provider = ParseProvider(Required(args, ref i, argument));
                    break;
                case "--limit":
                    limit = ParsePositiveInt(Required(args, ref i, argument), argument);
                    break;
                case "--case-id":
                    caseId = Required(args, ref i, argument);
                    break;
                case "--asr-timeout-seconds":
                    timeout = ParsePositiveInt(Required(args, ref i, argument), argument);
                    break;
                case "--output":
                case "-o":
                    output = Required(args, ref i, argument);
                    break;
                case "--verbose":
                case "-v":
                    verbose = true;
                    break;
                case "--enforce-gates":
                    enforceGates = true;
                    break;
                case "-h":
                case "--help":
                    PrintHelp();
                    throw new OperationCanceledException();
                default:
                    throw new ArgumentException($"Unknown end-to-end argument: {argument}");
            }
        }
        if (string.IsNullOrWhiteSpace(dataset))
            throw new ArgumentException("--dataset is required for --mode e2e.");
        if (string.IsNullOrWhiteSpace(asr))
            throw new ArgumentException("--asr-model or VOICEBOX_QWEN3_ASR_MODEL is required.");
        if (string.IsNullOrWhiteSpace(tts))
            throw new ArgumentException(
                "--tts-model, VOICEBOX_VOXCPM_TTS_MODEL, or VOXCPM_MODEL is required.");
        return new EndToEndOptions
        {
            DatasetRoot = dataset,
            AsrModelPath = asr,
            TtsModelPath = tts,
            TtsVoice = voice,
            VadModelDirectory = vadDirectory,
            ExecutionProvider = provider,
            Suite = suite,
            Limit = limit,
            CaseId = caseId,
            OutputPath = output,
            AsrTimeoutSeconds = timeout,
            Verbose = verbose,
            EnforceGates = enforceGates,
        };
    }

    private static OnnxExecutionProvider ParseProvider(string value)
        => value.Trim().ToLowerInvariant() switch
        {
            "cpu" => OnnxExecutionProvider.Cpu,
            "cuda" or "gpu" => OnnxExecutionProvider.Cuda,
            "directml" or "dml" => OnnxExecutionProvider.DirectML,
            _ => throw new ArgumentException(
                $"Unknown provider '{value}', expected cpu|cuda|directml."),
        };

    private static int ParsePositiveInt(string value, string flag)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ||
            parsed <= 0)
        {
            throw new ArgumentException($"{flag} expects a positive integer.");
        }
        return parsed;
    }

    private static string Required(string[] args, ref int index, string flag)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"Missing value for {flag}.");
        return args[++index];
    }

    private static void PrintCase(EndToEndCaseResult result)
    {
        Console.WriteLine(
            $"[{(result.Error is null ? "CASE" : "ERROR"),-5}] {result.Id} " +
            $"scenario={result.Scenario} " +
            $"expected={(result.ExpectAccepted ? "accept" : "reject")} " +
            $"heard={result.Heard} accepted={result.Accepted} rejected={result.Rejected} " +
            $"early={result.EarlyFalseTrigger} " +
            $"cer={result.CharacterErrorRate?.ToString("F3", CultureInfo.InvariantCulture) ?? "-"} " +
            $"heardLatency={result.HeardLatencyMs?.ToString("F0", CultureInfo.InvariantCulture) ?? "-"}ms " +
            $"reason={result.Error ?? result.ValidationReason ?? "-"}");
    }

    private static void PrintSummary(
        EndToEndReport report,
        string jsonPath,
        string csvPath)
    {
        EndToEndSummary s = report.Overall;
        Console.WriteLine();
        Console.WriteLine("End-to-end summary");
        Console.WriteLine($"  Cases/clusters          : {s.Total}/{s.ClusterCount}");
        Console.WriteLine($"  Positive/negative       : {s.Positive}/{s.Negative} clusters");
        Console.WriteLine($"  Heard recall            : {Percent(s.HeardRecall)}");
        Console.WriteLine($"  Duck recall             : {Percent(s.DuckRecall)}");
        Console.WriteLine($"  Accepted recall         : {Percent(s.AcceptedRecall)}");
        Console.WriteLine($"  Playback stop recall    : {Percent(s.PlaybackStopRecall)}");
        Console.WriteLine($"  Early false triggers    : {s.EarlyFalseTrigger}");
        Console.WriteLine($"  Echo-only heard/accepted: {s.EchoOnlyHeard}/{s.EchoOnlyAccepted}");
        Console.WriteLine($"  Aggregate CER           : {Percent(s.AggregateCharacterErrorRate)}");
        Console.WriteLine($"  Exact match             : {Percent(s.ExactMatchRate)}");
        Console.WriteLine($"  Empty recognition       : {Percent(s.EmptyRecognitionRate)}");
        Console.WriteLine($"  Accept via speaker/acoustic: {Percent(s.TextValidatedAcceptRate)}/{Percent(s.TimeoutAcceptRate)}");
        Console.WriteLine(
            $"  False accept per hour   : {s.FalseAcceptPerHour?.ToString("F1", CultureInfo.InvariantCulture) ?? "n/a"}" +
            $" over {s.NegativeAudioHours:F3}h echo-only");
        Console.WriteLine($"  Heard latency p50/p95   : {Ms(s.HeardLatencyP50Ms)}/{Ms(s.HeardLatencyP95Ms)}");
        Console.WriteLine($"  Duck ctrl/audible p95   : {Ms(s.DuckLatencyP95Ms)}/{Ms(s.AudibleDuckLatencyP95Ms)}");
        Console.WriteLine($"  Control stop p95        : {Ms(s.StopLatencyP95Ms)}");
        Console.WriteLine($"  Device drain done p95   : {Ms(s.DrainCompleteLatencyP95Ms)}");
        Console.WriteLine($"  Device tail p95         : {Ms(s.DeviceTailLatencyP95Ms)} (irreducible)");
        Console.WriteLine($"  Audible stop p95        : {Ms(s.AudibleStopLatencyP95Ms)} (incl. room reverb)");
        Console.WriteLine($"  Audible tail p95        : {Ms(s.AudibleTailLatencyP95Ms)}");
        Console.WriteLine(
            $"  Latency censored H/S/AS/AT: {s.HeardLatencyCensored}/" +
            $"{s.StopLatencyCensored}/{s.AudibleStopLatencyCensored}/" +
            $"{s.AudibleTailLatencyCensored}");
        Console.WriteLine($"  RTF                     : {report.RealTimeFactor:F3}");
        foreach (QualityGate gate in report.Gates)
        {
            Console.WriteLine(
                $"  Gate {gate.Status,-12}: {gate.Name}, " +
                $"actual={gate.Actual?.ToString("0.###", CultureInfo.InvariantCulture) ?? "n/a"} " +
                $"{gate.Comparison} {gate.Required:G6}; {gate.Detail}");
        }
        foreach (string tracked in EndToEndMetrics.TrackedNotEnforced)
            Console.WriteLine($"  Tracked (not enforced): {tracked}");
        Console.WriteLine($"  JSON                    : {jsonPath}");
        Console.WriteLine($"  Cases CSV               : {csvPath}");
    }

    private static string Percent(double? value)
        => value.HasValue ? value.Value.ToString("P2", CultureInfo.InvariantCulture) : "n/a";

    private static string Ms(double? value)
        => value.HasValue
            ? $"{value.Value.ToString("0", CultureInfo.InvariantCulture)}ms"
            : "n/a";

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            Chinese end-to-end benchmark:
              dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- \
                --mode e2e --dataset <prepared-root> --asr-model <qwen3-asr> \
                --tts-model <voxcpm> [--provider cpu|cuda|directml]

            Options:
              --provider <name>            Qwen3-ASR/FireRed VAD EP. VoxCPM TTS is always CPU.
              --limit <N>                 Run the first N deterministic manifest cases.
              --case-id <id>              Run one manifest case.
              --suite <name|all>          Manifest suite. Default: release-e2e.
              --vad-model-dir <path>      FireRed VAD model directory.
              --asr-timeout-seconds <N>   Harness wait timeout. Default: 120.
              --output <path>             JSON output; sibling .cases.csv is written.
              --verbose                   Print every case.
              --enforce-gates             Exit 1 on failed or inconclusive gates.
            """);
    }
}
