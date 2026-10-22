using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal enum ComparisonStatus
{
    Passed,
    Regressed,
    Incomparable,
}

internal sealed record ComparisonFinding(
    string Kind,
    string Scope,
    string Key,
    string Metric,
    string Baseline,
    string Candidate,
    string Detail);

internal sealed record ComparisonMetricRow(
    string Scope,
    string Key,
    string Metric,
    string Direction,
    string Baseline,
    string Candidate,
    string Status,
    string Detail);

internal sealed record ComparisonReport
{
    public required string SchemaVersion { get; init; }
    public required DateTimeOffset CreatedUtc { get; init; }
    public required string Status { get; init; }
    public string? Scope { get; init; }
    public string? BaselineAcoustic { get; init; }
    public string? CandidateAcoustic { get; init; }
    public string? BaselineE2e { get; init; }
    public string? CandidateE2e { get; init; }
    public required IReadOnlyList<string> EvidenceChecks { get; init; }
    public required IReadOnlyList<ComparisonFinding> Findings { get; init; }
    public required IReadOnlyList<ComparisonMetricRow> Metrics { get; init; }
}

internal static class ComparisonProgram
{
    private const string ComparisonSchemaVersion = "1.0";
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    private static readonly JsonSerializerOptions OutputJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    private static readonly string[] AcousticTruthFields =
    [
        "id", "corpus", "scenario", "deviceClass", "deviceSession", "clusterId",
        "expectedNearEnd", "callbackSamples", "sourceSampleRate", "evaluationSampleRate",
         "durationMs", "signalToEchoRatioDb", "farEndNonlinear",
        "farEndNoisy", "nearEndNoisy", "oracle"
    ];

    private static readonly string[] E2eTruthFields =
    [
        "id", "clusterId", "corpus", "speechAssetId", "expectedNearEnd", "scenario",
        "geometryId", "speakerPolicy", "expectHeard", "expectDuck", "expectAccepted",
        "expectPlaybackStop", "expectPlaybackResume", "referenceText", "callbackSamples",
        "audioDurationMs", "signalToEchoRatioDb", "signalToNoiseRatioDb", "echoDelayMs",
        "clockDriftPpm", "farEndNonlinear", "renderGapStartMs", "renderGapDurationMs",
        "echoRirId", "nearRirId", "noiseAssetId", "noiseKind", "noiseCorpus"
    ];

    private static readonly string[] AcousticConfigurationFields =
    [
        "targetSampleRate", "callbackSamples", "playbackStartupGuardMs", "activeNearEndWindowMs",
        "minDurationMs", "minRms", "mainAecWebRtc", "mainAecNoiseSuppression",
        "oracleDefinition", "confidenceLevel", "recallConfidenceMethod",
        "falseTriggerConfidenceMethod", "latencyCensoring", "seed", "limitPerStratum",
        "skippedIncompletePairs"
    ];

    private static readonly string[] AcousticHigherMetrics =
    [
        "truePositive", "trueNegative", "recall", "recallLower95", "recallUpper95",
        "precision", "precisionLower95", "precisionUpper95", "specificity",
        "specificityLower95", "specificityUpper95"
    ];

    private static readonly string[] AcousticLowerMetrics =
    [
        "falseNegative", "falsePositive", "earlyFalseTrigger", "errorCount", "latencyCensored",
        "falseTriggersPerHour", "falseTriggersPerHourUpper95", "latencyP50Ms", "latencyP95Ms"
    ];

    private static readonly string[] E2eHigherMetrics =
    [
        "heardPositive", "acceptedPositive", "duckedPositive", "stoppedPositive", "resumedPositive",
        "heardRecall", "acceptedRecall", "duckRecall", "playbackStopRecall", "playbackResumeRecall",
        "exactMatchRate"
    ];

    private static readonly string[] E2eLowerMetrics =
    [
        "rejectedPositive", "earlyFalseTrigger", "echoOnlyHeard", "echoOnlyAccepted", "errorCount",
        "aggregateCharacterErrorRate", "emptyRecognitionRate", "heardLatencyP50Ms", "heardLatencyP95Ms",
        "stopLatencyP95Ms", "audibleStopLatencyP95Ms", "audibleTailLatencyP95Ms", "heardLatencyCensored",
        "stopLatencyCensored", "audibleStopLatencyCensored", "audibleTailLatencyCensored"
    ];

    public static bool IsRequested(IReadOnlyList<string> args)
    {
        for (int i = 0; i + 1 < args.Count; i++)
        {
            if (string.Equals(args[i], "--mode", StringComparison.Ordinal) &&
                string.Equals(args[i + 1], "compare", StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    public static int Run(string[] args)
    {
        ComparisonOptions options = Parse(args);
        if (options.Help)
        {
            PrintHelp();
            return 0;
        }

        ComparisonReport report = Compare(
            options.BaselineAcoustic,
            options.CandidateAcoustic,
            options.AcousticOnly ? null : options.BaselineE2e,
            options.AcousticOnly ? null : options.CandidateE2e,
            options.BaselineEvidence!,
            options.CandidateEvidence!,
            options.E2eOnly);
        string output = ResolveOutput(options.Output, options.CandidateEvidence!);
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        File.WriteAllText(output, JsonSerializer.Serialize(report, OutputJsonOptions), new UTF8Encoding(false));
        string csv = string.IsNullOrWhiteSpace(options.Csv)
            ? Path.ChangeExtension(output, ".csv")
            : Path.GetFullPath(options.Csv!);
        WriteCsv(report, csv);
        PrintSummary(report, output, csv);
        return options.Enforce && !string.Equals(report.Status, nameof(ComparisonStatus.Passed), StringComparison.Ordinal)
            ? 1
            : 0;
    }

    internal static ComparisonReport Compare(
        string? baselineAcousticPath,
        string? candidateAcousticPath,
        string? baselineE2ePath,
        string? candidateE2ePath,
        string baselineEvidencePath,
        string candidateEvidencePath,
        bool e2eOnly = false)
    {
        bool compareAcoustic = !e2eOnly;
        if (compareAcoustic && (baselineAcousticPath is null || candidateAcousticPath is null))
            throw new ArgumentException("Both acoustic reports are required when acoustic comparison is enabled.");
        if (!e2eOnly && (baselineE2ePath is null || candidateE2ePath is null))
            throw new ArgumentException("Both E2E reports are required when E2E comparison is enabled.");
        EvaluationReport? baselineAcoustic = compareAcoustic ? Load<EvaluationReport>(baselineAcousticPath!) : null;
        EvaluationReport? candidateAcoustic = compareAcoustic ? Load<EvaluationReport>(candidateAcousticPath!) : null;
        bool compareE2e = baselineE2ePath is not null && candidateE2ePath is not null;
        if (!compareE2e && (baselineE2ePath is not null || candidateE2ePath is not null) && !e2eOnly)
            throw new ArgumentException("Both E2E reports are required when E2E comparison is enabled.");
        if (e2eOnly && !compareE2e)
            throw new ArgumentException("Both E2E reports are required in E2E-only comparison mode.");

        EndToEndReport? baselineE2e = compareE2e ? Load<EndToEndReport>(baselineE2ePath!) : null;
        EndToEndReport? candidateE2e = compareE2e ? Load<EndToEndReport>(candidateE2ePath!) : null;
        EvidenceSnapshot baselineEvidence = EvidenceSnapshot.Load(baselineEvidencePath, compareE2e);
        EvidenceSnapshot candidateEvidence = EvidenceSnapshot.Load(candidateEvidencePath, compareE2e);

        var findings = new List<ComparisonFinding>();
        var metrics = new List<ComparisonMetricRow>();
        var evidenceChecks = new List<string>();

        CompareEvidence(baselineEvidence, candidateEvidence, findings, evidenceChecks);
        if (compareAcoustic)
        {
            CompareAcousticReportIdentity(baselineAcoustic!, candidateAcoustic!, findings);
            CompareAcousticCases(baselineAcoustic!, candidateAcoustic!, findings);
            CompareAcousticSlices(baselineAcoustic!, candidateAcoustic!, metrics, findings);
            CompareGates("acoustic", baselineAcoustic!.Gates, candidateAcoustic!.Gates, findings);
        }
        if (compareE2e)
        {
            CompareE2eReportIdentity(baselineE2e!, candidateE2e!, findings);
            CompareE2eCases(baselineE2e!, candidateE2e!, findings);
            CompareE2eSlices(baselineE2e!, candidateE2e!, metrics, findings);
            CompareGates("e2e", baselineE2e!.Gates, candidateE2e!.Gates, findings);
        }
        else
        {
            evidenceChecks.Add("acoustic:not-executed-by-explicit-e2e-only-mode");
        }

        ComparisonStatus status = findings.Any(f => f.Kind == "incomparable")
            ? ComparisonStatus.Incomparable
            : findings.Count > 0
                ? ComparisonStatus.Regressed
                : ComparisonStatus.Passed;
        return new ComparisonReport
        {
            SchemaVersion = ComparisonSchemaVersion,
            CreatedUtc = DateTimeOffset.UtcNow,
            Status = status.ToString(),
            Scope = e2eOnly ? "e2e" : compareE2e ? "acoustic+e2e" : "acoustic",
            BaselineAcoustic = baselineAcousticPath is null ? null : Path.GetFullPath(baselineAcousticPath),
            CandidateAcoustic = candidateAcousticPath is null ? null : Path.GetFullPath(candidateAcousticPath),
            BaselineE2e = compareE2e ? Path.GetFullPath(baselineE2ePath!) : null,
            CandidateE2e = compareE2e ? Path.GetFullPath(candidateE2ePath!) : null,
            EvidenceChecks = evidenceChecks,
            Findings = findings,
            Metrics = metrics,
        };
    }

    internal static (string Status, string Detail) CompareScalarForSelfTest(
        double? baseline,
        double? candidate,
        bool higherIsBetter)
    {
        ComparisonValue result = CompareValue(baseline, candidate, higherIsBetter);
        return (result.Status, result.Detail);
    }

    internal static bool ModelHashesMatchForSelfTest(
        IReadOnlyDictionary<string, string> baseline,
        IReadOnlyDictionary<string, string> candidate)
        => baseline.Count == candidate.Count &&
           baseline.All(pair => candidate.TryGetValue(pair.Key, out string? hash) &&
                                string.Equals(hash, pair.Value, StringComparison.OrdinalIgnoreCase));

    internal static string NormalizeEnvironmentForSelfTest(string name, string value)
        => EvidenceSnapshot.NormalizeEnvironmentEvidence(name, value);

    internal static bool AcousticOnlyArgumentsAreAcceptedForSelfTest()
    {
        try
        {
            return Parse([
                "--mode", "compare",
                "--acoustic-only",
                "--baseline-acoustic", "baseline-acoustic.json",
                "--candidate-acoustic", "candidate-acoustic.json",
                "--baseline-evidence", "baseline-evidence",
                "--candidate-evidence", "candidate-evidence",
            ]).AcousticOnly;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    internal static bool AcousticOnlyArgumentsRejectE2eForSelfTest()
    {
        try
        {
            _ = Parse([
                "--mode", "compare",
                "--acoustic-only",
                "--baseline-acoustic", "baseline-acoustic.json",
                "--candidate-acoustic", "candidate-acoustic.json",
                "--baseline-e2e", "baseline-e2e.json",
                "--baseline-evidence", "baseline-evidence",
                "--candidate-evidence", "candidate-evidence",
            ]);
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    internal static bool AcousticOnlyCommandIdentityIgnoresUnexecutedE2eForSelfTest()
    {
        const string root = "C:\\evidence";
        const string acoustic =
            "dotnet run -- --dataset D:\\dataset --suite prepared-phase1 --callback-samples 256 --output A";
        const string e2e =
            "dotnet run -- --mode e2e --dataset D:\\dataset --suite release-e2e --output E";
        return string.Equals(
            EvidenceSnapshot.NormalizeCommandText(acoustic + "\n" + e2e, root, includeE2e: false),
            EvidenceSnapshot.NormalizeCommandText(acoustic, root, includeE2e: false),
            StringComparison.Ordinal);
    }

    private static void CompareEvidence(
        EvidenceSnapshot baseline,
        EvidenceSnapshot candidate,
        ICollection<ComparisonFinding> findings,
        ICollection<string> checks)
    {
        CompareEqual("evidence", "dataset-manifest", baseline.DatasetManifestHash,
            candidate.DatasetManifestHash, "Dataset manifest hash must be identical.", findings, checks);
        CompareEqual("evidence", "commands", baseline.Commands, candidate.Commands,
            "Benchmark commands differ after output-path normalization.", findings, checks);
        CompareEqual("evidence", "execution-backends", baseline.ExecutionBackends,
            candidate.ExecutionBackends, "Execution backend differs.", findings, checks);
        CompareEqual("evidence", "environment", baseline.Environment,
            candidate.Environment, "Runtime environment differs.", findings, checks);
        if (!ModelHashesMatchForSelfTest(baseline.SharedModelHashes, candidate.SharedModelHashes))
        {
            findings.Add(new ComparisonFinding(
                "incomparable", "evidence", "models", "shared-model-hash", "different", "different",
                "Shared ASR/TTS/VAD model files must have identical hashes."));
        }
        else
        {
            checks.Add("shared-model-hashes:match");
        }
        checks.Add("algorithm-model-hashes:allowed-to-change");
    }

    private static void CompareAcousticReportIdentity(
        EvaluationReport baselineAcoustic,
        EvaluationReport candidateAcoustic,
        ICollection<ComparisonFinding> findings)
    {
        CompareEqual("acoustic", "suite", baselineAcoustic.Suite, candidateAcoustic.Suite,
            "Acoustic suite differs.", findings);
        CompareEqual("acoustic", "dataset", baselineAcoustic.DatasetRoot, candidateAcoustic.DatasetRoot,
            "Acoustic dataset differs.", findings);
        CompareJsonFields(
            "acoustic", "configuration", baselineAcoustic.Configuration, candidateAcoustic.Configuration,
            AcousticConfigurationFields, findings);
    }

    private static void CompareE2eReportIdentity(
        EndToEndReport baselineE2e,
        EndToEndReport candidateE2e,
        ICollection<ComparisonFinding> findings)
    {
        CompareEqual("e2e", "dataset-id", baselineE2e.DatasetId, candidateE2e.DatasetId,
            "End-to-end dataset id differs.", findings);
        CompareEqual("e2e", "execution-provider", baselineE2e.ExecutionProvider,
            candidateE2e.ExecutionProvider, "ASR/VAD execution provider differs.", findings);
        CompareEqual("e2e", "tts-execution-provider", baselineE2e.TtsExecutionProvider,
            candidateE2e.TtsExecutionProvider, "TTS execution provider differs.", findings);
        CompareEqual("e2e", "asr-model-path", NormalizePath(baselineE2e.AsrModelPath),
            NormalizePath(candidateE2e.AsrModelPath), "ASR model path differs.", findings);
        CompareEqual("e2e", "tts-model-path", NormalizePath(baselineE2e.TtsModelPath),
            NormalizePath(candidateE2e.TtsModelPath), "TTS model path differs.", findings);
    }

    private static void CompareAcousticCases(
        EvaluationReport baseline,
        EvaluationReport candidate,
        ICollection<ComparisonFinding> findings)
    {
        Dictionary<string, JsonElement> before = LoadCaseElements(baseline.Cases);
        Dictionary<string, JsonElement> after = LoadCaseElements(candidate.Cases);
        CompareCaseKeys("acoustic", before.Keys, after.Keys, findings);
        foreach (string key in before.Keys.Intersect(after.Keys, StringComparer.Ordinal))
        {
            string left = TruthFingerprint(before[key], AcousticTruthFields);
            string right = TruthFingerprint(after[key], AcousticTruthFields);
            if (!string.Equals(left, right, StringComparison.Ordinal))
            {
                findings.Add(new ComparisonFinding(
                    "incomparable", "acoustic-case", key, "truth", left, right,
                    "Case identity or truth fields differ."));
                continue;
            }

            CaseResult b = baseline.Cases.Single(c => c.Id == key);
            CaseResult c = candidate.Cases.Single(item => item.Id == key);
            CompareCaseBoolean("acoustic-case", key, "error", b.Error is not null, c.Error is not null,
                c.Error is not null && b.Error is null, "Candidate introduced a case error.", findings);
            CompareCaseBoolean("acoustic-case", key, "false-positive", b.FalsePositive, c.FalsePositive,
                c.FalsePositive && !b.FalsePositive, "Candidate introduced a false trigger.", findings);
            CompareCaseBoolean("acoustic-case", key, "early-trigger", b.EarlyFalseTrigger, c.EarlyFalseTrigger,
                c.EarlyFalseTrigger && !b.EarlyFalseTrigger, "Candidate introduced an early trigger.", findings);
            CompareCaseBoolean("acoustic-case", key, "false-negative", b.FalseNegative, c.FalseNegative,
                c.FalseNegative && !b.FalseNegative, "Candidate introduced a miss.", findings);
        }
    }

    private static void CompareE2eCases(
        EndToEndReport baseline,
        EndToEndReport candidate,
        ICollection<ComparisonFinding> findings)
    {
        Dictionary<string, JsonElement> before = LoadCaseElements(baseline.Cases);
        Dictionary<string, JsonElement> after = LoadCaseElements(candidate.Cases);
        CompareCaseKeys("e2e", before.Keys, after.Keys, findings);
        foreach (string key in before.Keys.Intersect(after.Keys, StringComparer.Ordinal))
        {
            string left = TruthFingerprint(before[key], E2eTruthFields);
            string right = TruthFingerprint(after[key], E2eTruthFields);
            if (!string.Equals(left, right, StringComparison.Ordinal))
            {
                findings.Add(new ComparisonFinding(
                    "incomparable", "e2e-case", key, "truth", left, right,
                    "Case identity or layered truth fields differ."));
                continue;
            }

            EndToEndCaseResult b = baseline.Cases.Single(c => c.Id == key);
            EndToEndCaseResult c = candidate.Cases.Single(item => item.Id == key);
            bool negative = !b.ExpectedNearEnd;
            CompareCaseBoolean("e2e-case", key, "error", b.Error is not null, c.Error is not null,
                c.Error is not null && b.Error is null, "Candidate introduced a case error.", findings);
            CompareCaseBoolean("e2e-case", key, "early-trigger", b.EarlyFalseTrigger, c.EarlyFalseTrigger,
                c.EarlyFalseTrigger && !b.EarlyFalseTrigger, "Candidate introduced an early trigger.", findings);
            CompareCaseBoolean("e2e-case", key, "heard", b.Heard, c.Heard,
                b.ExpectHeard && !c.Heard && b.Heard, "Candidate lost Heard on a positive case.", findings);
            CompareCaseBoolean("e2e-case", key, "accepted", b.Accepted, c.Accepted,
                b.ExpectAccepted && !c.Accepted && b.Accepted, "Candidate lost Accepted on a positive case.", findings);
            if (negative)
            {
                CompareCaseBoolean("e2e-case", key, "remote-single-talk-heard", b.Heard, c.Heard,
                    c.Heard && !b.Heard, "Candidate introduced a remote single-talk Heard trigger.", findings);
                CompareCaseBoolean("e2e-case", key, "remote-single-talk-accepted", b.Accepted, c.Accepted,
                    c.Accepted && !b.Accepted, "Candidate introduced a remote single-talk Accepted trigger.", findings);
            }
        }
    }

    private static void CompareAcousticSlices(
        EvaluationReport baseline,
        EvaluationReport candidate,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        Dictionary<string, MetricSlice> before = baseline.Slices.Concat([baseline.Overall])
            .GroupBy(s => s.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);
        Dictionary<string, MetricSlice> after = candidate.Slices.Concat([candidate.Overall])
            .GroupBy(s => s.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);
        CompareSliceKeys("acoustic", before.Keys, after.Keys, findings);
        foreach (string key in before.Keys.Intersect(after.Keys, StringComparer.Ordinal))
        {
            MetricSlice b = before[key];
            MetricSlice c = after[key];
            CompareEqualMetric("acoustic-slice", key, "total", b.Total, c.Total, rows, findings);
            CompareEqualMetric("acoustic-slice", key, "positive", b.Positive, c.Positive, rows, findings);
            CompareEqualMetric("acoustic-slice", key, "negative", b.Negative, c.Negative, rows, findings);
            CompareEqualMetric("acoustic-slice", key, "ungradeable", b.Ungradeable, c.Ungradeable, rows, findings);
            CompareEqualMetric("acoustic-slice", key, "clusterCount", b.ClusterCount, c.ClusterCount, rows, findings);
            CompareEqualMetric("acoustic-slice", key, "oraclePositive", b.OraclePositive, c.OraclePositive, rows, findings);
            AddMetricResult("acoustic-slice", key, "negativeExposureHours", b.NegativeExposureHours,
                c.NegativeExposureHours, higherIsBetter: false, rows, findings);
            foreach (string metric in AcousticHigherMetrics)
                CompareAcousticMetric(key, metric, b, c, true, rows, findings);
            foreach (string metric in AcousticLowerMetrics)
                CompareAcousticMetric(key, metric, b, c, false, rows, findings);
        }
    }

    private static void CompareE2eSlices(
        EndToEndReport baseline,
        EndToEndReport candidate,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        Dictionary<string, EndToEndSummary> before = baseline.Slices.Concat([
            new EndToEndSummarySlice { Name = "overall", Summary = baseline.Overall }
        ]).GroupBy(s => s.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First().Summary, StringComparer.Ordinal);
        Dictionary<string, EndToEndSummary> after = candidate.Slices.Concat([
            new EndToEndSummarySlice { Name = "overall", Summary = candidate.Overall }
        ]).GroupBy(s => s.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First().Summary, StringComparer.Ordinal);
        CompareSliceKeys("e2e", before.Keys, after.Keys, findings);
        foreach (string key in before.Keys.Intersect(after.Keys, StringComparer.Ordinal))
        {
            EndToEndSummary b = before[key];
            EndToEndSummary c = after[key];
            CompareEqualMetric("e2e-slice", key, "total", b.Total, c.Total, rows, findings);
            CompareEqualMetric("e2e-slice", key, "positive", b.Positive, c.Positive, rows, findings);
            CompareEqualMetric("e2e-slice", key, "negative", b.Negative, c.Negative, rows, findings);
            CompareEqualMetric("e2e-slice", key, "clusterCount", b.ClusterCount, c.ClusterCount, rows, findings);
            foreach (string metric in E2eHigherMetrics)
                CompareE2eMetric(key, metric, b, c, true, rows, findings);
            foreach (string metric in E2eLowerMetrics)
                CompareE2eMetric(key, metric, b, c, false, rows, findings);
        }
    }

    private static void CompareAcousticMetric(
        string key,
        string metric,
        MetricSlice baseline,
        MetricSlice candidate,
        bool higherIsBetter,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        double? left = metric switch
        {
            "truePositive" => baseline.TruePositive,
            "trueNegative" => baseline.TrueNegative,
            "recall" => baseline.Recall,
            "recallLower95" => baseline.RecallLower95,
            "recallUpper95" => baseline.RecallUpper95,
            "precision" => baseline.Precision,
            "precisionLower95" => baseline.PrecisionLower95,
            "precisionUpper95" => baseline.PrecisionUpper95,
            "specificity" => baseline.Specificity,
            "specificityLower95" => baseline.SpecificityLower95,
            "specificityUpper95" => baseline.SpecificityUpper95,
            "falseNegative" => baseline.FalseNegative,
            "falsePositive" => baseline.FalsePositive,
            "earlyFalseTrigger" => baseline.EarlyFalseTrigger,
            "errorCount" => baseline.ErrorCount,
            "latencyCensored" => baseline.LatencyCensored,
            "falseTriggersPerHour" => baseline.FalseTriggersPerHour,
            "falseTriggersPerHourUpper95" => baseline.FalseTriggersPerHourUpper95,
            "latencyP50Ms" => baseline.LatencyP50Ms,
            "latencyP95Ms" => baseline.LatencyP95Ms,
            _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null),
        };
        double? right = metric switch
        {
            "truePositive" => candidate.TruePositive,
            "trueNegative" => candidate.TrueNegative,
            "recall" => candidate.Recall,
            "recallLower95" => candidate.RecallLower95,
            "recallUpper95" => candidate.RecallUpper95,
            "precision" => candidate.Precision,
            "precisionLower95" => candidate.PrecisionLower95,
            "precisionUpper95" => candidate.PrecisionUpper95,
            "specificity" => candidate.Specificity,
            "specificityLower95" => candidate.SpecificityLower95,
            "specificityUpper95" => candidate.SpecificityUpper95,
            "falseNegative" => candidate.FalseNegative,
            "falsePositive" => candidate.FalsePositive,
            "earlyFalseTrigger" => candidate.EarlyFalseTrigger,
            "errorCount" => candidate.ErrorCount,
            "latencyCensored" => candidate.LatencyCensored,
            "falseTriggersPerHour" => candidate.FalseTriggersPerHour,
            "falseTriggersPerHourUpper95" => candidate.FalseTriggersPerHourUpper95,
            "latencyP50Ms" => candidate.LatencyP50Ms,
            "latencyP95Ms" => candidate.LatencyP95Ms,
            _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null),
        };
        AddMetricResult("acoustic-slice", key, metric, left, right, higherIsBetter, rows, findings);
    }

    private static void CompareE2eMetric(
        string key,
        string metric,
        EndToEndSummary baseline,
        EndToEndSummary candidate,
        bool higherIsBetter,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        double? left = GetE2eMetric(baseline, metric);
        double? right = GetE2eMetric(candidate, metric);
        AddMetricResult("e2e-slice", key, metric, left, right, higherIsBetter, rows, findings);
    }

    private static double? GetE2eMetric(EndToEndSummary value, string metric)
        => metric switch
        {
            "heardPositive" => value.HeardPositive,
            "acceptedPositive" => value.AcceptedPositive,
            "duckedPositive" => value.DuckedPositive,
            "stoppedPositive" => value.StoppedPositive,
            "resumedPositive" => value.ResumedPositive,
            "heardRecall" => value.HeardRecall,
            "acceptedRecall" => value.AcceptedRecall,
            "duckRecall" => value.DuckRecall,
            "playbackStopRecall" => value.PlaybackStopRecall,
            "playbackResumeRecall" => value.PlaybackResumeRecall,
            "exactMatchRate" => value.ExactMatchRate,
            "rejectedPositive" => value.RejectedPositive,
            "earlyFalseTrigger" => value.EarlyFalseTrigger,
            "echoOnlyHeard" => value.EchoOnlyHeard,
            "echoOnlyAccepted" => value.EchoOnlyAccepted,
            "errorCount" => value.ErrorCount,
            "aggregateCharacterErrorRate" => value.AggregateCharacterErrorRate,
            "emptyRecognitionRate" => value.EmptyRecognitionRate,
            "heardLatencyP50Ms" => value.HeardLatencyP50Ms,
            "heardLatencyP95Ms" => value.HeardLatencyP95Ms,
            "stopLatencyP95Ms" => value.StopLatencyP95Ms,
            "audibleStopLatencyP95Ms" => value.AudibleStopLatencyP95Ms,
            "audibleTailLatencyP95Ms" => value.AudibleTailLatencyP95Ms,
            "heardLatencyCensored" => value.HeardLatencyCensored,
            "stopLatencyCensored" => value.StopLatencyCensored,
            "audibleStopLatencyCensored" => value.AudibleStopLatencyCensored,
            "audibleTailLatencyCensored" => value.AudibleTailLatencyCensored,
            _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null),
        };

    private static void AddMetricResult(
        string scope,
        string key,
        string metric,
        double? baseline,
        double? candidate,
        bool higherIsBetter,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        ComparisonValue value = CompareValue(baseline, candidate, higherIsBetter);
        rows.Add(new ComparisonMetricRow(
            scope, key, metric, higherIsBetter ? "higher" : "lower", Format(baseline), Format(candidate),
            value.Status, value.Detail));
        if (value.Status == "regressed")
        {
            findings.Add(new ComparisonFinding(
                "regression", scope, key, metric, Format(baseline), Format(candidate), value.Detail));
        }
    }

    private static ComparisonValue CompareValue(double? baseline, double? candidate, bool higherIsBetter)
    {
        if (!baseline.HasValue && !candidate.HasValue)
            return new("same", "Both values are unavailable.");
        if (!baseline.HasValue)
            return new("improved", "Candidate provides an observed value where baseline was censored/unavailable.");
        if (!candidate.HasValue)
            return new("regressed", "Candidate became censored/unavailable.");
        if (!double.IsFinite(baseline.Value) || !double.IsFinite(candidate.Value))
            return new("incomparable", "Metric contains a non-finite value.");
        const double epsilon = 1e-12;
        bool worse = higherIsBetter
            ? candidate.Value + epsilon < baseline.Value
            : candidate.Value > baseline.Value + epsilon;
        bool better = higherIsBetter
            ? candidate.Value > baseline.Value + epsilon
            : candidate.Value + epsilon < baseline.Value;
        return worse
            ? new("regressed", higherIsBetter
                ? "Candidate is lower than baseline."
                : "Candidate is higher than baseline.")
            : better
                ? new("improved", "Candidate improved over baseline.")
                : new("same", "Candidate matches baseline within tolerance.");
    }

    private static void CompareGates(
        string scope,
        IReadOnlyList<QualityGate> baseline,
        IReadOnlyList<QualityGate> candidate,
        ICollection<ComparisonFinding> findings)
    {
        Dictionary<string, QualityGate> before = baseline.ToDictionary(g => g.Name, StringComparer.Ordinal);
        Dictionary<string, QualityGate> after = candidate.ToDictionary(g => g.Name, StringComparer.Ordinal);
        CompareKeySets($"{scope}-gates", before.Keys, after.Keys, findings);
        foreach (string key in before.Keys.Intersect(after.Keys, StringComparer.Ordinal))
        {
            GateStatus left = before[key].Status;
            GateStatus right = after[key].Status;
            if (GateRank(right) > GateRank(left))
            {
                findings.Add(new ComparisonFinding(
                    "regression", $"{scope}-gate", key, "status", left.ToString(), right.ToString(),
                    "Candidate downgraded a baseline gate."));
            }
        }
    }

    private static int GateRank(GateStatus status)
        => status switch
        {
            GateStatus.Passed => 0,
            GateStatus.Inconclusive => 1,
            GateStatus.Failed => 2,
            _ => 3,
        };

    private static void CompareCaseBoolean(
        string scope,
        string key,
        string metric,
        bool baseline,
        bool candidate,
        bool regression,
        string detail,
        ICollection<ComparisonFinding> findings)
    {
        if (regression)
        {
            findings.Add(new ComparisonFinding(
                "regression", scope, key, metric, baseline.ToString(), candidate.ToString(), detail));
        }
    }

    private static void CompareCaseKeys(
        string scope,
        IEnumerable<string> baseline,
        IEnumerable<string> candidate,
        ICollection<ComparisonFinding> findings)
        => CompareKeySets($"{scope}-cases", baseline, candidate, findings);

    private static void CompareSliceKeys(
        string scope,
        IEnumerable<string> baseline,
        IEnumerable<string> candidate,
        ICollection<ComparisonFinding> findings)
        => CompareKeySets($"{scope}-slices", baseline, candidate, findings);

    private static void CompareKeySets(
        string scope,
        IEnumerable<string> baseline,
        IEnumerable<string> candidate,
        ICollection<ComparisonFinding> findings)
    {
        HashSet<string> left = baseline.ToHashSet(StringComparer.Ordinal);
        HashSet<string> right = candidate.ToHashSet(StringComparer.Ordinal);
        foreach (string missing in left.Except(right, StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal))
        {
            findings.Add(new ComparisonFinding(
                "incomparable", scope, missing, "presence", "present", "missing",
                "Candidate is missing a baseline case, slice, or gate."));
        }
        foreach (string added in right.Except(left, StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal))
        {
            findings.Add(new ComparisonFinding(
                "incomparable", scope, added, "presence", "missing", "present",
                "Candidate added a case, slice, or gate not present in baseline."));
        }
    }

    private static void CompareEqualMetric(
        string scope,
        string key,
        string metric,
        int baseline,
        int candidate,
        ICollection<ComparisonMetricRow> rows,
        ICollection<ComparisonFinding> findings)
    {
        string status = baseline == candidate ? "same" : "incomparable";
        string detail = baseline == candidate
            ? "Structural count matches baseline."
            : "Structural count differs; the compared slice is not equivalent.";
        rows.Add(new ComparisonMetricRow(scope, key, metric, "equal", baseline.ToString(CultureInfo.InvariantCulture),
            candidate.ToString(CultureInfo.InvariantCulture), status, detail));
        if (baseline != candidate)
        {
            findings.Add(new ComparisonFinding(
                "incomparable", scope, key, metric, baseline.ToString(CultureInfo.InvariantCulture),
                candidate.ToString(CultureInfo.InvariantCulture), detail));
        }
    }

    private static void CompareEqual(
        string scope,
        string key,
        string baseline,
        string candidate,
        string detail,
        ICollection<ComparisonFinding> findings,
        ICollection<string>? checks = null)
    {
        if (string.Equals(baseline, candidate, StringComparison.Ordinal))
        {
            checks?.Add($"{key}:match");
            return;
        }
        findings.Add(new ComparisonFinding("incomparable", scope, key, "identity", baseline, candidate, detail));
    }

    private static void CompareJsonFields<T>(
        string scope,
        string key,
        T baseline,
        T candidate,
        IReadOnlyList<string> fields,
        ICollection<ComparisonFinding> findings)
    {
        using JsonDocument left = JsonDocument.Parse(JsonSerializer.Serialize(baseline, JsonOptions));
        using JsonDocument right = JsonDocument.Parse(JsonSerializer.Serialize(candidate, JsonOptions));
        foreach (string field in fields)
        {
            JsonElement l = GetProperty(left.RootElement, field);
            JsonElement r = GetProperty(right.RootElement, field);
            string leftValue = Canonical(l);
            string rightValue = Canonical(r);
            if (!string.Equals(leftValue, rightValue, StringComparison.Ordinal))
            {
                findings.Add(new ComparisonFinding(
                    "incomparable", scope, key, field, leftValue, rightValue,
                    "Shared evaluation configuration differs."));
            }
        }
    }

    private static Dictionary<string, JsonElement> LoadCaseElements<T>(IReadOnlyList<T> cases)
    {
        using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(cases, JsonOptions));
        var result = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (JsonElement item in document.RootElement.EnumerateArray())
        {
            string id = item.GetProperty("id").GetString()
                ?? throw new InvalidDataException("A benchmark case has a null id.");
            if (!result.TryAdd(id, item.Clone()))
                throw new InvalidDataException($"Duplicate benchmark case id '{id}'.");
        }
        return result;
    }

    private static string TruthFingerprint(JsonElement item, IReadOnlyList<string> fields)
        => string.Join('\u001f', fields.Select(field => Canonical(GetProperty(item, field))));

    private static JsonElement GetProperty(JsonElement item, string name)
        => item.TryGetProperty(name, out JsonElement value) ? value : default;

    private static string Canonical(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Undefined)
            return "<missing>";
        return value.ValueKind switch
        {
            JsonValueKind.Object => "{" + string.Join(",", value.EnumerateObject()
                .OrderBy(p => p.Name, StringComparer.Ordinal)
                .Select(p => JsonSerializer.Serialize(p.Name) + ":" + Canonical(p.Value))) + "}",
            JsonValueKind.Array => "[" + string.Join(",", value.EnumerateArray().Select(Canonical)) + "]",
            _ => value.GetRawText(),
        };
    }

    private static T Load<T>(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Comparison input path cannot be empty.");
        string full = Path.GetFullPath(path);
        if (!File.Exists(full))
            throw new FileNotFoundException("Comparison report not found.", full);
        try
        {
            return JsonSerializer.Deserialize<T>(File.ReadAllText(full), JsonOptions)
                ?? throw new InvalidDataException($"Empty comparison report: {full}");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Invalid comparison report: {full}: {ex.Message}", ex);
        }
    }

    private static string ResolveOutput(string? output, string candidateEvidence)
        => string.IsNullOrWhiteSpace(output)
            ? Path.Combine(Path.GetFullPath(candidateEvidence), "comparison.json")
            : Path.GetFullPath(output);

    private static string NormalizePath(string value)
        => Path.GetFullPath(value).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

    private static string Format(double? value)
        => value.HasValue ? value.Value.ToString("R", CultureInfo.InvariantCulture) : "null";

    private static void WriteCsv(ComparisonReport report, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
        writer.WriteLine("scope,key,metric,direction,baseline,candidate,status,detail");
        foreach (ComparisonMetricRow row in report.Metrics)
        {
            writer.WriteLine(string.Join(",", Csv(row.Scope), Csv(row.Key), Csv(row.Metric), Csv(row.Direction),
                Csv(row.Baseline), Csv(row.Candidate), Csv(row.Status), Csv(row.Detail)));
        }
        foreach (ComparisonFinding finding in report.Findings)
        {
            writer.WriteLine(string.Join(",", Csv(finding.Scope), Csv(finding.Key), Csv(finding.Metric),
                Csv("finding"), Csv(finding.Baseline), Csv(finding.Candidate), Csv(finding.Kind), Csv(finding.Detail)));
        }
    }

    private static void PrintSummary(ComparisonReport report, string output, string csv)
    {
        Console.WriteLine("VoiceBox Assistant strict baseline comparison");
        Console.WriteLine($"Status        : {report.Status}");
        Console.WriteLine($"Findings      : {report.Findings.Count}");
        foreach (ComparisonFinding finding in report.Findings.Take(20))
            Console.WriteLine($"  {finding.Kind,-12} {finding.Scope}/{finding.Key}/{finding.Metric}: {finding.Detail}");
        if (report.Findings.Count > 20)
            Console.WriteLine($"  ... {report.Findings.Count - 20} more findings");
        Console.WriteLine($"JSON          : {output}");
        Console.WriteLine($"CSV           : {csv}");
    }

    private static ComparisonOptions Parse(string[] args)
    {
        var result = new ComparisonOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "-h":
                case "--help":
                    result.Help = true;
                    break;
                case "--mode":
                    string mode = Required(args, ref i, arg);
                    if (!string.Equals(mode, "compare", StringComparison.OrdinalIgnoreCase))
                        throw new ArgumentException($"Expected --mode compare, got '{mode}'.");
                    break;
                case "--baseline-acoustic":
                    result.BaselineAcoustic = SetOnce(result.BaselineAcoustic, Required(args, ref i, arg), arg);
                    break;
                case "--candidate-acoustic":
                    result.CandidateAcoustic = SetOnce(result.CandidateAcoustic, Required(args, ref i, arg), arg);
                    break;
                case "--baseline-e2e":
                    result.BaselineE2e = SetOnce(result.BaselineE2e, Required(args, ref i, arg), arg);
                    break;
                case "--candidate-e2e":
                    result.CandidateE2e = SetOnce(result.CandidateE2e, Required(args, ref i, arg), arg);
                    break;
                case "--acoustic-only":
                    if (result.AcousticOnly)
                        throw new ArgumentException("Duplicate --acoustic-only.");
                    if (result.E2eOnly)
                        throw new ArgumentException("--acoustic-only cannot be combined with --e2e-only.");
                    result.AcousticOnly = true;
                    break;
                case "--e2e-only":
                    if (result.E2eOnly)
                        throw new ArgumentException("Duplicate --e2e-only.");
                    if (result.AcousticOnly)
                        throw new ArgumentException("--e2e-only cannot be combined with --acoustic-only.");
                    result.E2eOnly = true;
                    break;
                case "--baseline-evidence":
                    result.BaselineEvidence = SetOnce(result.BaselineEvidence, Required(args, ref i, arg), arg);
                    break;
                case "--candidate-evidence":
                    result.CandidateEvidence = SetOnce(result.CandidateEvidence, Required(args, ref i, arg), arg);
                    break;
                case "--output":
                    result.Output = SetOnce(result.Output, Required(args, ref i, arg), arg);
                    break;
                case "--csv":
                    result.Csv = SetOnce(result.Csv, Required(args, ref i, arg), arg);
                    break;
                case "--enforce-comparison":
                    result.Enforce = true;
                    break;
                default:
                    throw new ArgumentException($"Unknown comparison argument: {arg}");
            }
        }
        if (result.Help)
            return result;
        if (!result.E2eOnly)
        {
            Require(result.BaselineAcoustic, "--baseline-acoustic");
            Require(result.CandidateAcoustic, "--candidate-acoustic");
        }
        if (result.AcousticOnly && (result.BaselineE2e is not null || result.CandidateE2e is not null))
            throw new ArgumentException("--acoustic-only cannot be combined with --baseline-e2e or --candidate-e2e.");
        if (!result.AcousticOnly)
        {
            Require(result.BaselineE2e, "--baseline-e2e");
            Require(result.CandidateE2e, "--candidate-e2e");
        }
        Require(result.BaselineEvidence, "--baseline-evidence");
        Require(result.CandidateEvidence, "--candidate-evidence");
        return result;
    }

    private static string SetOnce(string? current, string value, string flag)
        => current is null ? value : throw new ArgumentException($"Duplicate {flag}.");

    private static void Require(string? value, string flag)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"{flag} is required for comparison.");
    }

    private static string Required(string[] args, ref int index, string flag)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"Missing value for {flag}.");
        return args[++index];
    }

    private static string Csv(string value) => $"\"{value.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";

    private static void PrintHelp()
    {
        Console.WriteLine(
            """
            Strict baseline comparison:
              --mode compare --baseline-acoustic <path> --candidate-acoustic <path>
                --baseline-e2e <path> --candidate-e2e <path>
                --baseline-evidence <directory> --candidate-evidence <directory>
                [--output <comparison.json>] [--csv <comparison.csv>]
                [--enforce-comparison]

              --mode compare --acoustic-only
                --baseline-acoustic <path> --candidate-acoustic <path>
                --baseline-evidence <directory> --candidate-evidence <directory>
                [--output <comparison.json>] [--csv <comparison.csv>]
                [--enforce-comparison]

              --mode compare --e2e-only
                --baseline-e2e <path> --candidate-e2e <path>
                --baseline-evidence <directory> --candidate-evidence <directory>
                [--output <comparison.json>] [--csv <comparison.csv>]
                [--enforce-comparison]

            The comparison verifies shared experiment identity, then compares every
            serialized slice, gate, and case-level safety outcome for each explicitly
            requested suite. Acoustic-only mode records E2E as not executed; it does not
            substitute baseline E2E output for a candidate run.
            """);
    }

    private sealed class ComparisonOptions
    {
        public string? BaselineAcoustic { get; set; }
        public string? CandidateAcoustic { get; set; }
        public string? BaselineE2e { get; set; }
        public string? CandidateE2e { get; set; }
        public string? BaselineEvidence { get; set; }
        public string? CandidateEvidence { get; set; }
        public string? Output { get; set; }
        public string? Csv { get; set; }
        public bool AcousticOnly { get; set; }
        public bool E2eOnly { get; set; }
        public bool Enforce { get; set; }
        public bool Help { get; set; }
    }

    private readonly record struct ComparisonValue(string Status, string Detail);

    private sealed record EvidenceSnapshot(
        string DatasetManifestHash,
        string Commands,
        string ExecutionBackends,
        string Environment,
        IReadOnlyDictionary<string, string> SharedModelHashes)
    {
        public static EvidenceSnapshot Load(string root, bool includeE2eCommand)
        {
            string fullRoot = Path.GetFullPath(root);
            if (!Directory.Exists(fullRoot))
                throw new DirectoryNotFoundException($"Comparison evidence directory not found: {fullRoot}");
            string manifestHash = ReadHash(fullRoot, "dataset-manifest.sha256.txt", "dataset-manifest.json");
            string commands = NormalizeCommandText(
                ReadRequired(fullRoot, "commands.txt"),
                fullRoot,
                includeE2eCommand);
            string backends = NormalizeSharedBackends(ReadRequired(fullRoot, "execution-backends.txt"));
            string environment = ReadEnvironment(fullRoot);
            Dictionary<string, string> models = ReadModelHashes(fullRoot);
            if (models.Count == 0)
                throw new InvalidDataException($"No shared model hashes found in {fullRoot}.");
            return new EvidenceSnapshot(manifestHash, commands, backends, environment, models);
        }

        private static string ReadEnvironment(string root)
        {
            string[] names = [
                "environment-variables.txt", "dotnet-info.txt", "os.txt", "cpu.txt", "nvidia-smi-q.txt"
            ];
            var parts = new List<string>();
            foreach (string name in names)
            {
                string path = Path.Combine(root, name);
                if (File.Exists(path))
                {
                    parts.Add(
                        $"[{name}]\n{NormalizeEnvironmentEvidence(name, File.ReadAllText(path))}");
                }
            }
            if (parts.Count == 0)
                throw new FileNotFoundException($"No runtime environment evidence found in {root}.");
            return string.Join("\n", parts);
        }

        internal static string NormalizeEnvironmentEvidence(string name, string value)
        {
            string normalized = NormalizeText(value);
            if (string.Equals(name, "nvidia-smi-q.txt", StringComparison.OrdinalIgnoreCase))
                return NormalizeNvidiaSmi(normalized);
            if (string.Equals(name, "dotnet-info.txt", StringComparison.OrdinalIgnoreCase))
                return NormalizeDotnetInfo(normalized);
            if (string.Equals(name, "os.txt", StringComparison.OrdinalIgnoreCase))
            {
                return string.Join(
                    "\n",
                    normalized.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                        .Where(line => !line.TrimStart().StartsWith(
                            "LastBootUpTime",
                            StringComparison.OrdinalIgnoreCase))
                        .Select(line => line.TrimEnd()));
            }
            return normalized;
        }

        private static string NormalizeDotnetInfo(string value)
        {
            string[] fieldPrefixes =
            [
                "Version:", "Commit:", "Workload version:", "MSBuild version:",
                "OS Name:", "OS Version:", "OS Platform:", "RID:", "Base Path:"
            ];
            string[] entryPrefixes =
            [
                "Host:", "Architecture:", ".NET SDKs installed:",
                ".NET runtimes installed:", "Other architectures found:",
                "Microsoft.", "x86 ", "registered at "
            ];

            string[] stableLines = value
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line => fieldPrefixes.Any(prefix =>
                                   line.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) ||
                               entryPrefixes.Any(prefix =>
                                   line.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) ||
                               Regex.IsMatch(line, @"^\d+\.\d+\.\d+\s+\["))
                .ToArray();
            if (stableLines.Length == 0)
                throw new InvalidDataException("No stable .NET runtime identity fields found in dotnet-info evidence.");
            return string.Join("\n", stableLines);
        }

        private static string NormalizeNvidiaSmi(string value)
        {
            string[] stableFields =
            [
                "Driver Version", "CUDA Version", "Attached GPUs", "Product Name",
                "Product Brand", "Product Architecture", "GPU UUID", "Serial Number",
                "Board ID", "GPU Part Number", "Module ID", "VBIOS Version", "Bus Id",
                "Device Id", "Sub System Id"
            ];
            string[] lines = value.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .Where(line => stableFields.Any(field =>
                    line.StartsWith(field + " ", StringComparison.OrdinalIgnoreCase) &&
                    line.Contains(':')))
                .ToArray();
            if (lines.Length == 0)
                throw new InvalidDataException("No stable GPU identity fields found in nvidia-smi evidence.");
            return string.Join("\n", lines);
        }

        private static Dictionary<string, string> ReadModelHashes(string root)
        {
            string path = Path.Combine(root, "model-files.sha256.csv");
            if (!File.Exists(path))
                throw new FileNotFoundException("Model hash evidence is required for comparison.", path);
            var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string line in File.ReadLines(path).Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                string[] fields = ParseCsv(line);
                if (fields.Length < 3)
                    throw new InvalidDataException($"Invalid model hash row: {line}");
                string modelPath = NormalizePath(fields[0]);
                string hash = fields[2].Trim();
                if (IsAlgorithmModel(modelPath))
                    continue;
                if (!result.TryAdd(modelPath, hash))
                    throw new InvalidDataException($"Duplicate model hash path: {modelPath}");
            }
            return result;
        }

        private static bool IsAlgorithmModel(string path)
        {
            string lower = path.ToLowerInvariant();
            return lower.Contains("voicebox.dtd.renderaware", StringComparison.Ordinal) ||
                   lower.Contains("render_aware_dtd", StringComparison.Ordinal) ||
                   lower.Contains("\\dtd\\", StringComparison.Ordinal) ||
                   lower.Contains("dfsmn", StringComparison.Ordinal) ||
                   lower.Contains("barge-in-sidechain", StringComparison.Ordinal);
        }

        private static string NormalizeSharedBackends(string value)
        {
            var lines = NormalizeText(value)
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line =>
                    !line.Contains("legacy barge-in", StringComparison.OrdinalIgnoreCase) &&
                    !line.Contains("render-aware dtd", StringComparison.OrdinalIgnoreCase) &&
                    !line.Contains("barge-in detector", StringComparison.OrdinalIgnoreCase))
                .ToArray();
            return string.Join("\n", lines);
        }

        private static string ReadHash(string root, string hashFile, string sourceFile)
        {
            string path = Path.Combine(root, hashFile);
            if (File.Exists(path))
            {
                Match match = Regex.Match(File.ReadAllText(path), "[0-9a-fA-F]{64}");
                if (!match.Success)
                    throw new InvalidDataException($"No SHA-256 hash in {path}.");
                return match.Value.ToUpperInvariant();
            }
            string source = Path.Combine(root, sourceFile);
            if (!File.Exists(source))
                throw new FileNotFoundException($"Dataset manifest evidence is required in {root}.", source);
            return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(source)));
        }

        private static string ReadRequired(string root, string name)
        {
            string path = Path.Combine(root, name);
            if (!File.Exists(path))
                throw new FileNotFoundException($"Required comparison evidence is missing: {path}", path);
            return File.ReadAllText(path);
        }

        private static string NormalizeText(string value)
            => value.Replace("\r\n", "\n", StringComparison.Ordinal).Trim();

        internal static string NormalizeCommandText(
            string value,
            string evidenceRoot,
            bool includeE2e)
        {
            string normalized = NormalizeText(value).Replace(evidenceRoot, "<evidence>", StringComparison.OrdinalIgnoreCase);
            normalized = normalized.Replace(evidenceRoot.Replace('\\', '/'), "<evidence>", StringComparison.OrdinalIgnoreCase);
            // Baseline evidence predates the canonical callback-256 rerun and includes
            // diagnostic/build-only command lines. Compare the actual acoustic and E2E
            // benchmark invocations, whose arguments define the experiment identity.
            normalized = string.Join(
                "\n",
                normalized.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Where(line => (includeE2e && line.Contains("--mode e2e", StringComparison.Ordinal)) ||
                                   (line.Contains("--dataset", StringComparison.Ordinal) &&
                                    line.Contains("--callback-samples 256", StringComparison.Ordinal)))
                    .OrderBy(line => line, StringComparer.Ordinal));
            normalized = Regex.Replace(
                normalized,
                "(?i)(--output\\s+|--csv\\s+|-o\\s+)([^\\s`]+)",
                "$1<output>");
            return normalized;
        }

        private static string NormalizePath(string value)
            => Path.GetFullPath(value.Trim().Trim('"')).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        private static string[] ParseCsv(string line)
        {
            var fields = new List<string>();
            var builder = new StringBuilder();
            bool quoted = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (quoted && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        builder.Append('"');
                        i++;
                    }
                    else
                    {
                        quoted = !quoted;
                    }
                }
                else if (c == ',' && !quoted)
                {
                    fields.Add(builder.ToString());
                    builder.Clear();
                }
                else
                {
                    builder.Append(c);
                }
            }
            fields.Add(builder.ToString());
            return fields.ToArray();
        }
    }
}
