using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class ReportWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    public static (string JsonPath, string CsvPath) Write(
        EvaluationReport report,
        string? requestedPath)
    {
        string jsonPath = ResolveOutputPath(requestedPath);
        string directory = Path.GetDirectoryName(jsonPath)
            ?? throw new InvalidOperationException($"Cannot resolve output directory for {jsonPath}.");
        Directory.CreateDirectory(directory);

        File.WriteAllText(
            jsonPath,
            JsonSerializer.Serialize(report, JsonOptions),
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        string csvPath = Path.ChangeExtension(jsonPath, ".cases.csv");
        using (var writer = new StreamWriter(csvPath, append: false, new UTF8Encoding(false)))
        {
            writer.WriteLine(
                "id,corpus,scenario,deviceClass,deviceSession,clusterId,expectedNearEnd,callbackSamples," +
                "sourceSampleRate,durationMs,evaluatedDurationMs,resampled,detected,detectionTimeMs," +
                "eligibleLatencyMs,earlyFalseTrigger,truePositive,falseNegative,falsePositive," +
                "decisionReason,acceptedDurationMs,scoreDurationMs,peakScoreDurationMs," +
                "peakNearEndProbability,peakSmoothedNearEndProbability,oracleEligible," +
                "oracleFirstSpeechSample,oracleEligibleOnsetSample,oracleActiveDurationMs," +
                "oracleThresholdRms,serDb,coordinatorDiagnostic,error");
            foreach (CaseResult c in report.Cases)
            {
                writer.WriteLine(string.Join(",",
                    Csv(c.Id),
                    Csv(c.Corpus),
                    Csv(c.Scenario),
                    Csv(c.DeviceClass ?? string.Empty),
                    Csv(c.DeviceSession ?? string.Empty),
                    Csv(c.ClusterId),
                    Bool(c.ExpectedNearEnd),
                    c.CallbackSamples.ToString(CultureInfo.InvariantCulture),
                    c.SourceSampleRate.ToString(CultureInfo.InvariantCulture),
                    Number(c.DurationMs),
                    Number(c.EvaluatedDurationMs),
                    Bool(c.Resampled),
                    Bool(c.Detected),
                    Number(c.DetectionTimeMs),
                    Number(c.EligibleLatencyMs),
                    Bool(c.EarlyFalseTrigger),
                    Bool(c.TruePositive),
                    Bool(c.FalseNegative),
                    Bool(c.FalsePositive),
                    Csv(c.DecisionReason),
                    Number(c.DecisionAcceptedDurationMs),
                    Number(c.DecisionScoreDurationMs),
                    Number(c.DecisionPeakScoreDurationMs),
                    Number(c.PeakNearEndProbability),
                    Number(c.PeakSmoothedNearEndProbability),
                    Bool(c.Oracle.Eligible),
                    c.Oracle.FirstSpeechSample.ToString(CultureInfo.InvariantCulture),
                    c.Oracle.EligibleOnsetSample.ToString(CultureInfo.InvariantCulture),
                    Number(c.Oracle.ActiveDurationMs),
                    Number(c.Oracle.ThresholdRms),
                    Number(c.SignalToEchoRatioDb),
                    Csv(c.CoordinatorDiagnostic ?? string.Empty),
                    Csv(c.Error ?? string.Empty)));
            }
        }

        return (jsonPath, csvPath);
    }

    private static string ResolveOutputPath(string? requestedPath)
    {
        if (!string.IsNullOrWhiteSpace(requestedPath))
        {
            string full = Path.GetFullPath(requestedPath);
            return string.Equals(Path.GetExtension(full), ".json", StringComparison.OrdinalIgnoreCase)
                ? full
                : full + ".json";
        }

        string fileName = $"barge-in-aec-challenge-{DateTime.Now:yyyyMMdd-HHmmss}.json";
        foreach (string start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            string? cursor = Path.GetFullPath(start);
            for (int i = 0; i < 10 && cursor is not null; i++)
            {
                string src = Path.Combine(cursor, "src");
                if (Directory.Exists(Path.Combine(src, "VoiceBox.Assistant")))
                    return Path.Combine(src, "TestResults", fileName);
                if (Directory.Exists(Path.Combine(cursor, "VoiceBox.Assistant")))
                    return Path.Combine(cursor, "TestResults", fileName);
                cursor = Path.GetDirectoryName(cursor);
            }
        }

        return Path.Combine(Environment.CurrentDirectory, fileName);
    }

    private static string Csv(string value)
        => $"\"{value.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";

    private static string Bool(bool value) => value ? "true" : "false";

    private static string Number(double value)
        => value.ToString("0.########", CultureInfo.InvariantCulture);

    private static string Number(float value)
        => value.ToString("0.########", CultureInfo.InvariantCulture);

    private static string Number(double? value)
        => value?.ToString("0.########", CultureInfo.InvariantCulture) ?? string.Empty;
}
