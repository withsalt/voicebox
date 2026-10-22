using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        try { Console.OutputEncoding = Encoding.UTF8; } catch { }

        if (ComparisonProgram.IsRequested(args))
        {
            try
            {
                return ComparisonProgram.Run(args);
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"[ERROR] {ex.Message}");
                Console.Error.WriteLine("Use --mode compare --help for usage.");
                return 2;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"[ERROR] Benchmark comparison failed: {ex.GetType().Name}: {ex.Message}");
                return 1;
            }
        }

        if (EndToEndProgram.IsRequested(args))
        {
            try
            {
                return await EndToEndProgram.RunAsync(args, Cancellation.Token)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (args.Any(a => a is "-h" or "--help"))
            {
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"[ERROR] End-to-end benchmark failed: {ex.GetType().Name}: {ex.Message}");
                return 1;
            }
        }

        if (PreparedDatasetBuilder.IsRequested(args))
        {
            try
            {
                return await PreparedDatasetBuilder.RunAsync(args).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"[ERROR] Prepared dataset creation failed: {ex.GetType().Name}: {ex.Message}");
                return 1;
            }
        }

        EvaluationOptions options;
        try
        {
            options = ParseArguments(args);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine($"[ERROR] {ex.Message}");
            Console.Error.WriteLine("Use --help for usage.");
            return 2;
        }

        if (args.Any(a => a is "-h" or "--help"))
        {
            PrintHelp();
            return 0;
        }

        if (options.SelfTest)
        {
            try
            {
                SelfChecks.Run();
                Console.WriteLine("[PASS] Deterministic evaluator self-checks passed.");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[FAIL] {ex.Message}");
                return 1;
            }
        }

        if (string.IsNullOrWhiteSpace(options.DatasetRoot))
        {
            Console.Error.WriteLine(
                "[ERROR] Dataset root is required. Use --dataset <path> or set VOICEBOX_AEC_CHALLENGE.");
            return 2;
        }

        IReadOnlyList<EvaluationCase> cases;
        try
        {
            cases = DatasetCatalog.Discover(options);
            if (!string.IsNullOrWhiteSpace(options.CaseId))
            {
                cases = cases
                    .Where(testCase => string.Equals(
                        testCase.Id,
                        options.CaseId,
                        StringComparison.OrdinalIgnoreCase))
                    .ToArray();
                if (cases.Count == 0)
                    throw new InvalidOperationException(
                        $"No case matched --case-id '{options.CaseId}' in suite {options.Suite}.");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Dataset discovery failed: {ex.Message}");
            return 2;
        }

        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            Cancellation.Cancel();
        };

        int totalRuns = checked(cases.Count * options.CallbackSamples.Length);
        Console.WriteLine("VoiceBox Assistant AEC Challenge barge-in benchmark");
        Console.WriteLine($"Dataset       : {Path.GetFullPath(options.DatasetRoot)}");
        Console.WriteLine($"Suite         : {options.Suite}");
        Console.WriteLine($"Cases / runs  : {cases.Count} / {totalRuns}");
        Console.WriteLine($"Callbacks     : {string.Join(", ", options.CallbackSamples)} samples");
        if (options.Suite == EvaluationSuite.DeviceSample)
        {
            PrintDeviceCoverage(cases);
            Console.WriteLine($"Dataset pairs : skipped incomplete={options.SkippedIncompletePairs}");
        }
        Console.WriteLine(
            $"Decision      : guard={options.PlaybackStartupGuardMs}ms, " +
            $"window={options.ActiveNearEndWindowMs}ms, min={options.MinDurationMs}ms/{options.MinRms:F4} RMS");

        var results = new List<CaseResult>(totalRuns);
        var stopwatch = Stopwatch.StartNew();
        int completed = 0;
        try
        {
            using var evaluator = new BargeInCorpusEvaluator(options);
            foreach (EvaluationCase testCase in cases)
            {
                foreach (int callbackSamples in options.CallbackSamples)
                {
                    Cancellation.Token.ThrowIfCancellationRequested();
                    try
                    {
                        CaseResult result = await evaluator
                            .EvaluateAsync(testCase, callbackSamples, Cancellation.Token)
                            .ConfigureAwait(false);
                        results.Add(result);
                        if (options.Verbose || result.FalsePositive || result.FalseNegative || result.EarlyFalseTrigger)
                            PrintCase(result);
                    }
                    catch (Exception ex)
                    {
                        results.Add(ErrorResult(testCase, callbackSamples, ex));
                        Console.Error.WriteLine(
                            $"[ERROR] {testCase.Id} callback={callbackSamples}: {ex.GetType().Name}: {ex.Message}");
                    }

                    completed++;
                    if (!options.Verbose && (completed == totalRuns || completed % 25 == 0))
                        Console.WriteLine($"Progress      : {completed}/{totalRuns}");
                }
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("[INFO] Evaluation canceled.");
            return 130;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Evaluation pipeline failed: {ex}");
            return 1;
        }
        finally
        {
            stopwatch.Stop();
        }

        MetricSlice overall = MetricAggregator.Aggregate("overall", results);
        IReadOnlyList<QualityGate> gates = MetricAggregator.EvaluateGates(results, options);
        double audioSeconds = results.Sum(r => r.EvaluatedDurationMs) / 1000.0;
        var report = new EvaluationReport
        {
            SchemaVersion = "2.1",
            CreatedUtc = DateTimeOffset.UtcNow,
            DatasetRoot = Path.GetFullPath(options.DatasetRoot),
            Suite = options.Suite.ToString(),
            Scope = options.Suite == EvaluationSuite.DeviceSample
                ? "Offline device-sample runtime-path benchmark over anonymous real recording " +
                  "sessions. Device classes are gated independently; missing expected classes are " +
                  "inconclusive. Hardware driver timing and exact product device models remain out of scope."
                : "Offline runtime-path benchmark. It replaces the hardware duplex service with " +
                  "dataset-backed ingress, then runs the production VoiceAssistant, WebRTC APM, " +
                  "the render-aware DTD, barge-in decider, coordinator, " +
                  "and state machine. A shared virtual TimeProvider advances only with dataset audio.",
            RuntimeVersion = RuntimeInformation.FrameworkDescription,
            OperatingSystem = RuntimeInformation.OSDescription,
            GitCommit = TryGetGitCommit(),
            GitWorktreeState = TryGetGitWorktreeState(),
            DatasetGitCommit = TryGetGitCommit(options.DatasetRoot),
            DatasetGitWorktreeState = TryGetGitWorktreeState(options.DatasetRoot),
            Configuration = new ReportConfiguration
            {
                TargetSampleRate = EvaluationOptions.TargetSampleRate,
                CallbackSamples = options.CallbackSamples,
                PlaybackStartupGuardMs = options.PlaybackStartupGuardMs,
                ActiveNearEndWindowMs = options.ActiveNearEndWindowMs,
                MinDurationMs = options.MinDurationMs,
                MinRms = options.MinRms,
                MainAecWebRtc = true,
                MainAecNoiseSuppression = options.EnableNoiseSuppression,
                OracleDefinition =
                    "Synthetic nearend_speech * nearend_scale; 20 ms RMS frames; " +
                    "threshold=max(0.0005, 3% of clip peak frame RMS); bridge <=100 ms gaps; " +
                    "require configured minimum speech after startup guard.",
                ConfidenceLevel = 0.95,
                RecallConfidenceMethod =
                    "Deterministic device-session/utterance cluster bootstrap; " +
                    "Wilson fallback when fewer than five clusters are available.",
                FalseTriggerConfidenceMethod =
                    "Conservative maximum of exact one-sided Poisson and clustered bootstrap upper bounds.",
                LatencyCensoring =
                    "Missed and early detections are right-censored above the clip window; " +
                    "reported percentiles are null when the requested rank intersects censored samples.",
                Seed = options.Seed,
                LimitPerStratum = options.LimitPerStratum,
                SkippedIncompletePairs = options.SkippedIncompletePairs,
            },
            WallTimeSeconds = stopwatch.Elapsed.TotalSeconds,
            AudioTimeSeconds = audioSeconds,
            RealTimeFactor = audioSeconds > 0 ? stopwatch.Elapsed.TotalSeconds / audioSeconds : 0,
            Overall = overall,
            Slices = MetricAggregator.BuildSlices(results),
            Gates = gates,
            Cases = results,
        };

        (string jsonPath, string csvPath) = ReportWriter.Write(report, options.OutputPath);
        PrintSummary(report, jsonPath, csvPath);

        return options.EnforceGates && gates.Any(g => g.Status != GateStatus.Passed)
            ? 1
            : 0;
    }

    private static EvaluationOptions ParseArguments(string[] args)
    {
        var options = new EvaluationOptions
        {
            DatasetRoot =
                Environment.GetEnvironmentVariable("VOICEBOX_AEC_CHALLENGE")
                ?? (Directory.Exists(@"F:\Temp\AEC-Challenge") ? @"F:\Temp\AEC-Challenge" : string.Empty),
        };

        for (int i = 0; i < args.Length; i++)
        {
            string argument = args[i];
            switch (argument)
            {
                case "-h":
                case "--help":
                    break;
                case "--self-test":
                    options.SelfTest = true;
                    break;
                case "--dataset":
                    options.DatasetRoot = Required(args, ref i, argument);
                    break;
                case "--suite":
                    options.Suite = ParseSuite(Required(args, ref i, argument));
                    break;
                case "--case-id":
                    options.CaseId = Required(args, ref i, argument);
                    break;
                case "--limit-per-stratum":
                case "--limit":
                    options.LimitPerStratum = ParseNonNegativeInt(Required(args, ref i, argument), argument);
                    break;
                case "--seed":
                    options.Seed = int.Parse(Required(args, ref i, argument), CultureInfo.InvariantCulture);
                    break;
                case "--callback-samples":
                    options.CallbackSamples = ParsePositiveIntList(Required(args, ref i, argument), argument);
                    break;
                case "--guard-ms":
                    options.PlaybackStartupGuardMs = ParseNonNegativeInt(Required(args, ref i, argument), argument);
                    break;
                case "--window-ms":
                    options.ActiveNearEndWindowMs = ParsePositiveInt(Required(args, ref i, argument), argument);
                    break;
                case "--min-duration-ms":
                    options.MinDurationMs = ParseNonNegativeInt(Required(args, ref i, argument), argument);
                    break;
                case "--min-rms":
                    options.MinRms = ParseNonNegativeFloat(Required(args, ref i, argument), argument);
                    break;
                case "--noise-suppression":
                    options.EnableNoiseSuppression = true;
                    break;
                case "--output":
                case "-o":
                    options.OutputPath = Required(args, ref i, argument);
                    break;
                case "--minimum-recall":
                    options.MinimumRecall = ParseRatio(Required(args, ref i, argument), argument);
                    break;
                case "--maximum-false-triggers-per-hour":
                    options.MaximumFalseTriggersPerHour = ParseNonNegativeDouble(
                        Required(args, ref i, argument),
                        argument);
                    break;
                case "--maximum-p95-latency-ms":
                    options.MaximumP95EligibleLatencyMs = ParseNonNegativeDouble(
                        Required(args, ref i, argument),
                        argument);
                    break;
                case "--enforce-gates":
                    options.EnforceGates = true;
                    break;
                case "--verbose":
                case "-v":
                    options.Verbose = true;
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: {argument}");
            }
        }

        return options;
    }

    private static EvaluationSuite ParseSuite(string value)
        => value.Trim().ToLowerInvariant() switch
        {
            "smoke" => EvaluationSuite.Smoke,
            "standard" => EvaluationSuite.Standard,
            "synthetic-test" or "synthetic" => EvaluationSuite.SyntheticTest,
            "real-test" or "real" => EvaluationSuite.RealTest,
            "icassp2022" or "2022" => EvaluationSuite.Icassp2022,
            "device-sample" or "device" => EvaluationSuite.DeviceSample,
            "prepared-smoke" => EvaluationSuite.PreparedSmoke,
            "prepared-phase1" or "phase1" => EvaluationSuite.PreparedPhase1,
            "all" => EvaluationSuite.All,
            _ => throw new ArgumentException(
                $"Unknown suite '{value}'. Expected " +
                "smoke|standard|synthetic-test|real-test|icassp2022|device-sample|" +
                "prepared-smoke|prepared-phase1|all."),
        };

    private static string Required(string[] args, ref int index, string flag)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"Missing value for {flag}.");
        return args[++index];
    }

    private static int ParsePositiveInt(string value, string flag)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
            throw new ArgumentException($"{flag} expects a positive integer.");
        return parsed;
    }

    private static int ParseNonNegativeInt(string value, string flag)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed < 0)
            throw new ArgumentException($"{flag} expects a non-negative integer.");
        return parsed;
    }

    private static int[] ParsePositiveIntList(string value, string flag)
    {
        string[] parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            throw new ArgumentException($"{flag} expects one or more comma-separated positive integers.");
        return parts.Select(part => ParsePositiveInt(part, flag)).Distinct().ToArray();
    }

    private static float ParseNonNegativeFloat(string value, string flag)
    {
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ||
            !float.IsFinite(parsed) ||
            parsed < 0)
        {
            throw new ArgumentException($"{flag} expects a finite non-negative number.");
        }
        return parsed;
    }

    private static double ParseNonNegativeDouble(string value, string flag)
    {
        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ||
            !double.IsFinite(parsed) ||
            parsed < 0)
        {
            throw new ArgumentException($"{flag} expects a finite non-negative number.");
        }
        return parsed;
    }

    private static double ParseRatio(string value, string flag)
    {
        double parsed = ParseNonNegativeDouble(value, flag);
        if (parsed > 1)
            throw new ArgumentException($"{flag} expects a number in [0, 1].");
        return parsed;
    }

    private static CaseResult ErrorResult(EvaluationCase testCase, int callbackSamples, Exception ex)
        => new()
        {
            Id = testCase.Id,
            Corpus = testCase.Corpus,
            Scenario = testCase.Scenario,
            DeviceClass = testCase.DeviceClass,
            DeviceSession = testCase.DeviceSession,
            ClusterId = testCase.ClusterId ?? testCase.DeviceSession ?? testCase.Id,
            ExpectedNearEnd = testCase.ExpectedNearEnd,
            CallbackSamples = callbackSamples,
            SourceSampleRate = 0,
            EvaluationSampleRate = EvaluationOptions.TargetSampleRate,
            DurationMs = 0,
            EvaluatedDurationMs = 0,
            Resampled = false,
            Detected = false,
            DetectionTimeMs = null,
            DecisionReason = "error",
            DecisionFrameCount = 0,
            DecisionAcceptedFrameCount = 0,
            DecisionEchoLikeFrameCount = 0,
            DecisionAcceptedDurationMs = 0,
            DecisionScoreDurationMs = 0,
            DecisionPeakScoreDurationMs = 0,
            PeakNearEndProbability = 0,
            PeakSmoothedNearEndProbability = 0,
            Oracle = OracleActivity.Unavailable,
            EligibleLatencyMs = null,
            EarlyFalseTrigger = false,
            TruePositive = false,
            FalseNegative = false,
            FalsePositive = false,
            SignalToEchoRatioDb = testCase.SignalToEchoRatioDb,
            FarEndNonlinear = testCase.FarEndNonlinear,
            FarEndNoisy = testCase.FarEndNoisy,
            NearEndNoisy = testCase.NearEndNoisy,
            Error = $"{ex.GetType().Name}: {ex.Message}",
        };

    private static void PrintCase(CaseResult result)
    {
        string outcome = result.Error is not null
            ? "ERROR"
            : result.TruePositive
                ? "TP"
                : result.FalsePositive
                    ? "FP"
                    : result.FalseNegative
                        ? "FN"
                        : result.EarlyFalseTrigger
                            ? "EARLY-FP"
                            : "TN/SKIP";
        Console.WriteLine(
            $"[{outcome,-8}] {result.Id} cb={result.CallbackSamples} " +
            $"det={result.DetectionTimeMs?.ToString("F0", CultureInfo.InvariantCulture) ?? "-"}ms " +
            $"p={result.PeakSmoothedNearEndProbability:F3} reason={result.DecisionReason}" +
            (result.FalseNegative &&
             string.Equals(result.DecisionReason, "confirmed-nearend-window", StringComparison.Ordinal)
                ? $" coord={result.CoordinatorDiagnostic}"
                : string.Empty));
    }

    private static void PrintDeviceCoverage(IReadOnlyList<EvaluationCase> cases)
    {
        foreach (string deviceClass in DatasetCatalog.ExpectedDeviceClasses)
        {
            EvaluationCase[] deviceCases = cases
                .Where(c => string.Equals(c.DeviceClass, deviceClass, StringComparison.Ordinal))
                .ToArray();
            int sessions = deviceCases
                .Select(c => c.DeviceSession)
                .Where(session => session is not null)
                .Distinct(StringComparer.Ordinal)
                .Count();
            string marker = deviceCases.Length == 0 ? " [MISSING]" : string.Empty;
            Console.WriteLine(
                $"Device class  : {deviceClass}={deviceCases.Length} cases/{sessions} sessions{marker}");
        }
    }

    private static void PrintSummary(EvaluationReport report, string jsonPath, string csvPath)
    {
        MetricSlice s = report.Overall;
        Console.WriteLine();
        Console.WriteLine("Summary");
        Console.WriteLine($"  TP/FN/FP/TN      : {s.TruePositive}/{s.FalseNegative}/{s.FalsePositive}/{s.TrueNegative}");
        Console.WriteLine($"  Ungradeable      : {s.Ungradeable}");
        Console.WriteLine($"  Early false      : {s.EarlyFalseTrigger}");
        Console.WriteLine(
            $"  Recall           : {Percent(s.Recall)} " +
            $"(95% CI {Interval(s.RecallLower95, s.RecallUpper95)})");
        Console.WriteLine(
            $"  Precision        : {Percent(s.Precision)} " +
            $"(95% CI {Interval(s.PrecisionLower95, s.PrecisionUpper95)})");
        Console.WriteLine(
            $"  Specificity      : {Percent(s.Specificity)} " +
            $"(95% CI {Interval(s.SpecificityLower95, s.SpecificityUpper95)})");
        Console.WriteLine(
            $"  False triggers/h : {Number(s.FalseTriggersPerHour)}, " +
            $"one-sided 95% upper={Number(s.FalseTriggersPerHourUpper95)} " +
            $"({s.NegativeExposureHours:F3} negative audio hours)");
        Console.WriteLine(
            $"  Eligible latency : p50={Milliseconds(s.LatencyP50Ms)}, " +
            $"p95={Milliseconds(s.LatencyP95Ms)} " +
            $"(observed={s.LatencyCount}, censored={s.LatencyCensored}, oracle={s.OraclePositive})");
        Console.WriteLine($"  RTF              : {report.RealTimeFactor:F3}");
        foreach (QualityGate gate in report.Gates)
        {
            Console.WriteLine(
                $"  Gate {gate.Status,-12}: {gate.Name}, " +
                $"actual={Number(gate.Actual)} {gate.Comparison} {gate.Required:G6}; {gate.Detail}");
        }
        Console.WriteLine($"  JSON             : {jsonPath}");
        Console.WriteLine($"  Cases CSV        : {csvPath}");
    }

    private static string Percent(double? value)
        => value.HasValue ? $"{value.Value:P2}" : "n/a";

    private static string Number(double? value)
        => value?.ToString("0.###", CultureInfo.InvariantCulture) ?? "n/a";

    private static string Milliseconds(double? value)
        => value.HasValue ? $"{Number(value)}ms" : "n/a";

    private static string Interval(double? lower, double? upper)
        => lower.HasValue && upper.HasValue
            ? $"[{lower.Value:P2}, {upper.Value:P2}]"
            : "n/a";

    private static string? TryGetGitCommit(string? startDirectory = null)
    {
        string? root = FindRepositoryRoot(startDirectory);
        if (root is null)
            return null;

        return TryRunGit(root, "rev-parse", "HEAD");
    }

    private static GitWorktreeState TryGetGitWorktreeState(string? startDirectory = null)
    {
        string? root = FindRepositoryRoot(startDirectory);
        if (root is null)
            return GitWorktreeState.Unavailable;

        string? status = TryRunGit(root, "status", "--porcelain", "--untracked-files=normal");
        if (status is null)
            return GitWorktreeState.Unavailable;
        return status.Length > 0 ? GitWorktreeState.Dirty : GitWorktreeState.Clean;
    }

    private static string? TryRunGit(string root, params string[] arguments)
    {
        try
        {
            var startInfo = new ProcessStartInfo("git")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = root,
            };
            foreach (string argument in arguments)
                startInfo.ArgumentList.Add(argument);
            using Process? process = Process.Start(startInfo);
            if (process is null)
                return null;
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(3000);
            return process.ExitCode == 0 && output.Length > 0 ? output : null;
        }
        catch
        {
            return null;
        }
    }

    private static string? FindRepositoryRoot(string? requestedStart)
    {
        IEnumerable<string> starts = string.IsNullOrWhiteSpace(requestedStart)
            ? [Environment.CurrentDirectory, AppContext.BaseDirectory]
            : [requestedStart];
        foreach (string start in starts)
        {
            string? cursor = Path.GetFullPath(start);
            for (int i = 0; i < 10 && cursor is not null; i++)
            {
                if (Directory.Exists(Path.Combine(cursor, ".git")))
                    return cursor;
                cursor = Path.GetDirectoryName(cursor);
            }
        }
        return null;
    }

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            VoiceBox Assistant AEC Challenge barge-in benchmark

            Usage:
              dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- [options]

            Suites:
              smoke             Two deterministic samples per corpus/scenario stratum.
              standard          Synthetic test split plus 2021 clean/noisy real test sets.
              synthetic-test    The 500-row synthetic test split with clean near-end oracle.
              real-test         2021 clean/noisy double-talk and far-end single-talk pairs.
              icassp2022        48 kHz ICASSP 2022 pairs, resampled to the production 16 kHz.
              device-sample     Deterministic real/real_doubled/fullband/mobile device sample.
                                Defaults to 50 cases per corpus/scenario/label stratum.
              prepared-smoke    Two cases per stratum from a prepared evaluation dataset.
              prepared-phase1   Complete acoustic suite from a prepared evaluation dataset.
              all               standard plus ICASSP 2022.

            Options:
              --dataset <path>                  AEC-Challenge root. Defaults to
                                                VOICEBOX_AEC_CHALLENGE or F:\Temp\AEC-Challenge.
              --suite <name>                    Default: smoke.
              --case-id <id>                    Run one exact discovered case id.
              --limit-per-stratum <N>           Deterministic cap for each corpus/scenario/label.
              --callback-samples <N[,N,...]>    Default: 256, matching the current demo maximum.
              --seed <N>                        Sampling seed. Default: 42.
              --guard-ms <N>                    Default comes from VoiceAssistantBargeInOptions.
              --window-ms <N>                   Default comes from VoiceAssistantBargeInOptions.
              --min-duration-ms <N>             Default comes from VoiceAssistantBargeInOptions.
              --min-rms <value>                 Default comes from VoiceAssistantBargeInOptions.
              --noise-suppression               Enable WebRTC NS in addition to AEC3.
              --output, -o <path>               JSON path; a sibling .cases.csv is also written.
              --minimum-recall <0..1>           Gate default: 0.90.
              --maximum-false-triggers-per-hour Gate default: 1.0.
              --maximum-p95-latency-ms <N>      Gate default: 700.
              --enforce-gates                   Exit 1 when any gate fails or is inconclusive.
              --verbose, -v                     Print every case.
              --self-test                       Run deterministic evaluator checks without models/data.
              --help, -h                        Show help.

            Prepare a fixed evaluation dataset:
              --prepare-dataset
              --aec-root <path> --kws-root <path> --but-rir-root <path>
              --but-libri-root <path> --wenet-root <path> --prepared-output <path>
              [--materialization hardlink|copy] [--seed <N>]

            Chinese end-to-end:
              --mode e2e --dataset <prepared-root> --asr-model <path> --tts-model <voxcpm-path>
              (see the end-to-end help for the full option list)

            Strict baseline comparison:
              --mode compare --baseline-acoustic <path> --candidate-acoustic <path>
              --baseline-e2e <path> --candidate-e2e <path>
              --baseline-evidence <directory> --candidate-evidence <directory>
              --output <comparison.json> [--csv <comparison.csv>] [--enforce-comparison]

            Recommended:
              dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- \
                --dataset F:\Temp\AEC-Challenge --suite standard --output src/TestResults/barge-in.json

            This tool measures the production acoustic "Heard" decision. Run
            VoiceBox.Assistant.Tests as the separate control-plane gate for ASR validation,
            cancellation, state transition, ducking, and playback queue clearing.
            """);
    }

    private static class Cancellation
    {
        private static readonly CancellationTokenSource Source = new();
        public static CancellationToken Token => Source.Token;
        public static void Cancel() => Source.Cancel();
    }
}
