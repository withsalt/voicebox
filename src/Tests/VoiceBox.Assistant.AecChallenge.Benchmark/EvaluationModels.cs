using VoiceBox.Assistant.Api.Configuration;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal enum EvaluationSuite
{
    Smoke,
    Standard,
    SyntheticTest,
    RealTest,
    Icassp2022,
    DeviceSample,
    PreparedSmoke,
    PreparedPhase1,
    All,
}

internal sealed class EvaluationOptions
{
    public const int TargetSampleRate = 16000;

    public string DatasetRoot { get; set; } = string.Empty;
    public EvaluationSuite Suite { get; set; } = EvaluationSuite.Smoke;
    public string? CaseId { get; set; }
    public int LimitPerStratum { get; set; }
    public int Seed { get; set; } = 42;
    public int[] CallbackSamples { get; set; } = [256];
    public string? OutputPath { get; set; }
    public bool EnableNoiseSuppression { get; set; }
    public bool Verbose { get; set; }
    public bool EnforceGates { get; set; }
    public bool SelfTest { get; set; }
    public int SkippedIncompletePairs { get; internal set; }

    public int PlaybackStartupGuardMs { get; set; }
    public int ActiveNearEndWindowMs { get; set; }
    public int MinDurationMs { get; set; }
    public float MinRms { get; set; }

    public double MinimumRecall { get; set; } = 0.90;
    public double MaximumFalseTriggersPerHour { get; set; } = 1.0;
    public double MaximumP95EligibleLatencyMs { get; set; } = 700;
    public int MinimumGatePositiveCount { get; set; } = 30;
    public int MinimumGateNegativeCount { get; set; } = 30;
    public int MinimumGateLatencyCount { get; set; } = 20;

    public EvaluationOptions()
    {
        var defaults = new VoiceAssistantBargeInOptions();
        PlaybackStartupGuardMs = defaults.PlaybackStartupGuardMs;
        ActiveNearEndWindowMs = defaults.ActiveNearEndWindowMs;
        MinDurationMs = defaults.MinDurationMs;
        MinRms = defaults.MinRms;
    }
}

internal sealed record EvaluationCase(
    string Id,
    string Corpus,
    string Scenario,
    bool ExpectedNearEnd,
    string MicrophonePath,
    string RenderPath,
    string? NearEndReferencePath = null,
    double NearEndScale = 1,
    double? SignalToEchoRatioDb = null,
    bool? FarEndNonlinear = null,
    bool? FarEndNoisy = null,
    bool? NearEndNoisy = null,
    string? DeviceClass = null,
    string? DeviceSession = null,
    string? ClusterId = null);

internal sealed record OracleActivity(
    bool Available,
    bool Eligible,
    long FirstSpeechSample,
    long EligibleOnsetSample,
    long LastSpeechSampleExclusive,
    double ActiveDurationMs,
    double ThresholdRms)
{
    public static OracleActivity Unavailable { get; } =
        new(false, false, -1, -1, -1, 0, 0);
}

internal sealed record CaseResult
{
    public string Id { get; init; } = string.Empty;
    public string Corpus { get; init; } = string.Empty;
    public string Scenario { get; init; } = string.Empty;
    public string? DeviceClass { get; init; }
    public string? DeviceSession { get; init; }
    public string ClusterId { get; init; } = string.Empty;
    public bool ExpectedNearEnd { get; init; }
    public int CallbackSamples { get; init; }
    public int SourceSampleRate { get; init; }
    public int EvaluationSampleRate { get; init; }
    public double DurationMs { get; init; }
    public double EvaluatedDurationMs { get; init; }
    public bool Resampled { get; init; }
    public bool Detected { get; init; }
    public double? DetectionTimeMs { get; init; }
    public string DecisionReason { get; init; } = string.Empty;
    public int DecisionFrameCount { get; init; }
    public int DecisionAcceptedFrameCount { get; init; }
    public int DecisionEchoLikeFrameCount { get; init; }
    public double DecisionAcceptedDurationMs { get; init; }
    public double DecisionScoreDurationMs { get; init; }
    public double DecisionPeakScoreDurationMs { get; init; }
    public float PeakNearEndProbability { get; init; }
    public float PeakSmoothedNearEndProbability { get; init; }
    public OracleActivity Oracle { get; init; } = OracleActivity.Unavailable;
    public double? EligibleLatencyMs { get; init; }
    public bool EarlyFalseTrigger { get; init; }
    public bool TruePositive { get; init; }
    public bool FalseNegative { get; init; }
    public bool FalsePositive { get; init; }
    public double? SignalToEchoRatioDb { get; init; }
    public bool? FarEndNonlinear { get; init; }
    public bool? FarEndNoisy { get; init; }
    public bool? NearEndNoisy { get; init; }
    public string? CoordinatorDiagnostic { get; init; }
    public string? Error { get; init; }
}

internal sealed record MetricSlice
{
    public string Name { get; init; } = string.Empty;
    public int Total { get; init; }
    public int Positive { get; init; }
    public int Negative { get; init; }
    public int Ungradeable { get; init; }
    public int TruePositive { get; init; }
    public int FalseNegative { get; init; }
    public int FalsePositive { get; init; }
    public int TrueNegative { get; init; }
    public int EarlyFalseTrigger { get; init; }
    public int ErrorCount { get; init; }
    public int ClusterCount { get; init; }
    public double? Recall { get; init; }
    public double? RecallLower95 { get; init; }
    public double? RecallUpper95 { get; init; }
    public double? Precision { get; init; }
    public double? PrecisionLower95 { get; init; }
    public double? PrecisionUpper95 { get; init; }
    public double? Specificity { get; init; }
    public double? SpecificityLower95 { get; init; }
    public double? SpecificityUpper95 { get; init; }
    public double NegativeExposureHours { get; init; }
    public double? FalseTriggersPerHour { get; init; }
    public double? FalseTriggersPerHourUpper95 { get; init; }
    public int OraclePositive { get; init; }
    public int LatencyCount { get; init; }
    public int LatencyCensored { get; init; }
    public double? LatencyP50Ms { get; init; }
    // Reports produced before latency P95 was serialized legitimately omit this
    // derived field. The strict comparator treats a missing value as censored.
    public double? LatencyP95Ms { get; init; }
}

internal enum GateStatus
{
    Passed,
    Failed,
    Inconclusive,
}

internal enum GitWorktreeState
{
    Clean,
    Dirty,
    Unavailable,
}

internal sealed record QualityGate(
    string Name,
    GateStatus Status,
    double? Actual,
    double Required,
    string Comparison,
    string Detail);

internal sealed record EvaluationReport
{
    public string SchemaVersion { get; init; } = string.Empty;
    public DateTimeOffset CreatedUtc { get; init; }
    public string DatasetRoot { get; init; } = string.Empty;
    public string Suite { get; init; } = string.Empty;
    public string Scope { get; init; } = string.Empty;
    public string RuntimeVersion { get; init; } = string.Empty;
    public string OperatingSystem { get; init; } = string.Empty;
    public string? GitCommit { get; init; }
    public GitWorktreeState GitWorktreeState { get; init; }
    public string? DatasetGitCommit { get; init; }
    public GitWorktreeState DatasetGitWorktreeState { get; init; }
    public ReportConfiguration Configuration { get; init; } = new();
    public double WallTimeSeconds { get; init; }
    public double AudioTimeSeconds { get; init; }
    public double RealTimeFactor { get; init; }
    public MetricSlice Overall { get; init; } = new();
    public IReadOnlyList<MetricSlice> Slices { get; init; } = [];
    public IReadOnlyList<QualityGate> Gates { get; init; } = [];
    public IReadOnlyList<CaseResult> Cases { get; init; } = [];
}

internal sealed record ReportConfiguration
{
    public int TargetSampleRate { get; init; }
    public int[] CallbackSamples { get; init; } = [];
    public int PlaybackStartupGuardMs { get; init; }
    public int ActiveNearEndWindowMs { get; init; }
    public int MinDurationMs { get; init; }
    public float MinRms { get; init; }
    public bool MainAecWebRtc { get; init; }
    public bool MainAecNoiseSuppression { get; init; }
    public string OracleDefinition { get; init; } = string.Empty;
    public double ConfidenceLevel { get; init; }
    public string RecallConfidenceMethod { get; init; } = string.Empty;
    public string FalseTriggerConfidenceMethod { get; init; } = string.Empty;
    public string LatencyCensoring { get; init; } = string.Empty;
    public int Seed { get; init; }
    public int LimitPerStratum { get; init; }
    public int SkippedIncompletePairs { get; init; }
}
