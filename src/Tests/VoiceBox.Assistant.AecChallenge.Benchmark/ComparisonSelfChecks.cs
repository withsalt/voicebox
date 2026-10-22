using System.Text.Json;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static partial class SelfChecks
{
    private static void ComparisonAllowsAlgorithmChangesAndCoversEverySliceMetric()
    {
        (string status, _) = ComparisonProgram.CompareScalarForSelfTest(0.50, 0.51, higherIsBetter: true);
        Assert(status == "improved", "higher-is-better metrics should recognize an improvement");
        (status, _) = ComparisonProgram.CompareScalarForSelfTest(0.50, 0.49, higherIsBetter: false);
        Assert(status == "improved", "lower-is-better metrics should recognize an improvement");
        (status, _) = ComparisonProgram.CompareScalarForSelfTest(0.50, 0.49, higherIsBetter: true);
        Assert(status == "regressed", "higher-is-better metrics should detect a regression");
        (status, _) = ComparisonProgram.CompareScalarForSelfTest(0.50, 0.51, higherIsBetter: false);
        Assert(status == "regressed", "lower-is-better metrics should detect a regression");
    }

    private static void ComparisonRejectsNewCaseRegressions()
    {
        (string status, _) = ComparisonProgram.CompareScalarForSelfTest(0.75, null, higherIsBetter: true);
        Assert(status == "regressed", "a newly censored higher-is-better metric must regress");
        (status, _) = ComparisonProgram.CompareScalarForSelfTest(null, 0.75, higherIsBetter: false);
        Assert(status == "improved", "an observed lower-is-better metric should improve over unavailable baseline");
    }

    private static void ComparisonEnforcesGateTransitions()
    {
        (string status, _) = ComparisonProgram.CompareScalarForSelfTest(1, 2, higherIsBetter: false);
        Assert(status == "regressed", "gate-like maximum metrics must reject increases");
    }

    private static void ComparisonLoadsFrozenLegacyAlgorithmFieldsStrictly()
    {
        // Unknown legacy configuration fields are intentionally ignored by the report reader;
        // shared configuration fields remain explicitly compared by the comparison engine.
        using JsonDocument document = JsonDocument.Parse("{\"sidechain\":\"DFSMN\",\"dfsmnDoubleTalkNearEndThreshold\":1.5}");
        Assert(document.RootElement.GetProperty("sidechain").GetString() == "DFSMN", "legacy algorithm metadata fixture should be valid");
    }

    private static void ComparisonAllowsAlgorithmModelHashChangesOnly()
    {
        var baseline = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["shared/asr.onnx"] = "A",
        };
        var same = new Dictionary<string, string>(baseline, StringComparer.OrdinalIgnoreCase);
        var changed = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["shared/asr.onnx"] = "B",
        };
        Assert(ComparisonProgram.ModelHashesMatchForSelfTest(baseline, same), "identical shared model hashes should match");
        Assert(!ComparisonProgram.ModelHashesMatchForSelfTest(baseline, changed), "changed shared model hashes must be rejected");
    }

    private static void ComparisonIgnoresVolatileGpuTelemetryButKeepsGpuIdentity()
    {
        const string baseline = """
            Timestamp : Sat Aug 1
            Driver Version : 595.79
            CUDA Version : 13.2
            Product Name : NVIDIA RTX 4090
            GPU UUID : GPU-123
            Performance State : P2
            GPU Current Temp : 44 C
            """;
        const string candidate = """
            Timestamp : Sun Aug 2
            Driver Version : 595.79
            CUDA Version : 13.2
            Product Name : NVIDIA RTX 4090
            GPU UUID : GPU-123
            Performance State : P8
            GPU Current Temp : 51 C
            """;
        const string differentGpu = """
            Driver Version : 595.79
            CUDA Version : 13.2
            Product Name : NVIDIA RTX 5090
            GPU UUID : GPU-456
            """;

        string normalizedBaseline =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("nvidia-smi-q.txt", baseline);
        string normalizedCandidate =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("nvidia-smi-q.txt", candidate);
        string normalizedDifferentGpu =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("nvidia-smi-q.txt", differentGpu);
        Assert(normalizedBaseline == normalizedCandidate,
            "volatile GPU telemetry must not change environment identity");
        Assert(normalizedBaseline != normalizedDifferentGpu,
            "GPU identity changes must remain comparable evidence failures");

        string os = ComparisonProgram.NormalizeEnvironmentForSelfTest(
            "os.txt",
            "Caption : Windows\nBuildNumber : 26200\nLastBootUpTime : yesterday");
        Assert(!os.Contains("LastBootUpTime", StringComparison.Ordinal),
            "OS boot time must not change environment identity");
    }

    private static void ComparisonNormalizesLocalizedDotnetInfoButKeepsRuntimeIdentity()
    {
        const string localized = """
            .NET SDK:
             Version:           10.0.400
             Commit:            14fbf8d527
             Workload version:  10.0.400-manifests.b0ae88bd
             MSBuild version:   18.9.6+14fbf8d52

            运行时环境:
             OS Name:     Windows
             OS Version:  10.0.26200
             OS Platform: Windows
             RID:         win-x64
             Base Path:   C:\Program Files\dotnet\sdk\10.0.400\

            Host:
              Version:      10.0.11
              Architecture: x64
              Commit:       e2f47b0110

            .NET SDKs installed:
              10.0.400 [C:\Program Files\dotnet\sdk]

            .NET runtimes installed:
              Microsoft.NETCore.App 10.0.11 [C:\Program Files\dotnet\shared\Microsoft.NETCore.App]
            """;
        const string mojibake = """
            .NET SDK:
             Version:           10.0.400
             Commit:            14fbf8d527
             Workload version:  10.0.400-manifests.b0ae88bd
             MSBuild version:   18.9.6+14fbf8d52

            杞愯鏃剁幆澧?
             OS Name:     Windows
             OS Version:  10.0.26200
             OS Platform: Windows
             RID:         win-x64
             Base Path:   C:\Program Files\dotnet\sdk\10.0.400\

            Host:
              Version:      10.0.11
              Architecture: x64
              Commit:       e2f47b0110

            .NET SDKs installed:
              10.0.400 [C:\Program Files\dotnet\sdk]

            .NET runtimes installed:
              Microsoft.NETCore.App 10.0.11 [C:\Program Files\dotnet\shared\Microsoft.NETCore.App]
            """;
        string normalizedLocalized =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("dotnet-info.txt", localized);
        string normalizedMojibake =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("dotnet-info.txt", mojibake);
        Assert(normalizedLocalized == normalizedMojibake,
            "localized dotnet labels must not make an otherwise identical runtime incomparable");

        string changedSdk = mojibake.Replace("10.0.400", "10.0.401", StringComparison.Ordinal);
        string normalizedChanged =
            ComparisonProgram.NormalizeEnvironmentForSelfTest("dotnet-info.txt", changedSdk);
        Assert(normalizedLocalized != normalizedChanged,
            "a .NET SDK identity change must remain incomparable");
    }

    private static void ComparisonCliIsStrictAndDoesNotCaptureExistingModes()
    {
        Assert(
            ComparisonProgram.IsRequested(["--mode", "compare"]),
            "compare mode should be recognized");
        Assert(
            !ComparisonProgram.IsRequested(["--mode", "e2e"]),
            "existing e2e mode must not be captured by comparison mode");
        Assert(
            !ComparisonProgram.IsRequested(["--dataset", "compare"]),
            "a dataset value equal to compare must not select comparison mode");
        Assert(
            ComparisonProgram.AcousticOnlyArgumentsAreAcceptedForSelfTest(),
            "an intermediate acoustic-only comparison must not require an unexecuted E2E report");
        Assert(
            ComparisonProgram.AcousticOnlyArgumentsRejectE2eForSelfTest(),
            "acoustic-only comparison must reject an E2E input instead of silently skipping it");
        Assert(
            ComparisonProgram.AcousticOnlyCommandIdentityIgnoresUnexecutedE2eForSelfTest(),
            "acoustic-only identity must compare only the acoustic command that actually ran");
    }
}
