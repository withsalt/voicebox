using System.Text.Json;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class GoldenTraceContractTests
{
    [Fact]
    public void ScenarioTraceSchema_RequiresStableBusinessFieldsWithoutClrTypeNames()
    {
        using JsonDocument schema = GoldenTraceStore.Read<JsonDocument>("Schemas/assistant-scenario-trace.schema.json");
        JsonElement properties = schema.RootElement.GetProperty("properties");

        Assert.Contains("schema", properties.EnumerateObject().Select(static value => value.Name));
        Assert.Contains("catalogVersion", properties.EnumerateObject().Select(static value => value.Name));
        Assert.Contains("oracleKind", properties.EnumerateObject().Select(static value => value.Name));
        Assert.Contains("publicationSeq", properties.GetProperty("publicEvents").GetProperty("items").GetProperty("properties").EnumerateObject().Select(static value => value.Name));
        Assert.DoesNotContain("assemblyQualifiedName", schema.RootElement.GetRawText(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("clrType", schema.RootElement.GetRawText(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ScheduleManifestSchema_RequiresNamedControlledVariants()
    {
        using JsonDocument schema = GoldenTraceStore.Read<JsonDocument>("Schemas/schedule-manifest.schema.json");
        JsonElement variant = schema.RootElement
            .GetProperty("properties")
            .GetProperty("variants")
            .GetProperty("items");

        string[] required = variant.GetProperty("required").EnumerateArray()
            .Select(static value => value.GetString()!)
            .ToArray();

        // A declared oracle kind plus a per-variant artifact hash keeps a named
        // schedule family from drifting silently.
        Assert.Equal(["scheduleId", "traceFile", "happensBefore", "status", "sha256"], required);
        Assert.Contains(
            "oracleKind",
            schema.RootElement.GetProperty("required").EnumerateArray()
                .Select(static value => value.GetString()!));
        Assert.False(variant.GetProperty("additionalProperties").GetBoolean());
    }

    [Theory]
    [InlineData("Schedules/speech-end-vs-asr-start/manifest.json")]
    [InlineData("Schedules/response-complete-vs-barge-in/manifest.json")]
    public void ScheduleManifests_DeclareAnOracleKindAndMatchEveryVariantArtifactHash(string manifestPath)
    {
        ScheduleManifest manifest = GoldenTraceStore.Read<ScheduleManifest>(manifestPath);
        Assert.Equal("AllowedTraceSet", manifest.OracleKind);
        Assert.NotEmpty(manifest.Variants);

        string directory = Path.GetDirectoryName(manifestPath)!.Replace('\\', '/');
        foreach (ScheduleVariant variant in manifest.Variants)
        {
            string tracePath = $"{directory}/{variant.TraceFile}";
            string actual = GoldenTraceStore.ComputeSha256(tracePath);
            Assert.Equal(variant.Sha256, actual);
        }
    }

    [Fact]
    public void Normalizer_SortsTraceCollectionsWithoutChangingBusinessPayload()
    {
        var trace = new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            "normalizer-contract",
            "ExactTrace",
            [new ScenarioInput(2, "b", "payload-b"), new ScenarioInput(1, "a", "payload-a")],
            [new ScenarioEvent(2, "b"), new ScenarioEvent(1, "a")],
            [new ScenarioProviderCall(2, "b", "payload-b"), new ScenarioProviderCall(1, "a", "payload-a")],
            new ScenarioFinalState("Stopped", "Disabled"));

        AssistantScenarioTrace normalized = AssistantScenarioTraceNormalizer.Normalize(trace);

        Assert.Equal([1, 2], normalized.Inputs.Select(static value => value.Ordinal));
        Assert.Equal([1, 2], normalized.PublicEvents.Select(static value => value.PublicationSeq));
        Assert.Equal([1, 2], normalized.ProviderCalls.Select(static value => value.Ordinal));
        Assert.Equal("payload-a", normalized.Inputs[0].Detail);
    }

    [Theory]
    [InlineData("Schedules/speech-end-vs-asr-start/manifest.json", "speech-end-first.json", "CR-003")]
    [InlineData("Schedules/speech-end-vs-asr-start/manifest.json", "asr-ready-first.json", "CR-003")]
    [InlineData("Schedules/response-complete-vs-barge-in/manifest.json", "response-generated-first.json", "CR-006")]
    public void ScheduleVariantTraces_AreManifestedWithAnExactOracle(
        string manifestPath,
        string traceFile,
        string scenarioId)
    {
        ScheduleManifest manifest = GoldenTraceStore.Read<ScheduleManifest>(manifestPath);
        ScheduleVariant variant = Assert.Single(
            manifest.Variants,
            value => string.Equals(value.TraceFile, traceFile, StringComparison.Ordinal));
        Assert.Equal(traceFile, variant.TraceFile);
        string tracePath = Path.Combine(Path.GetDirectoryName(manifestPath)!, traceFile)
            .Replace('\\', '/');
        AssistantScenarioTrace trace = GoldenTraceStore.Read<AssistantScenarioTrace>(tracePath);
        Assert.Equal(scenarioId, trace.ScenarioId);
        Assert.Equal("ExactTrace", trace.OracleKind);
    }

    [Theory]
    [InlineData(
        "Schedules/speech-end-vs-asr-start/manifest.json",
        "runtime-speech-end-first.json",
        "runtime-speech-end-first")]
    [InlineData(
        "Schedules/speech-end-vs-asr-start/manifest.json",
        "runtime-asr-ready-first.json",
        "runtime-asr-ready-first")]
    [InlineData(
        "Schedules/response-complete-vs-barge-in/manifest.json",
        "runtime-barge-in-before-response-generated.json",
        "runtime-barge-in-before-response-generated")]
    public void RuntimeScheduleTraces_AreManifestedAndSelfIdentifying(
        string manifestPath,
        string traceFile,
        string scheduleId)
    {
        ScheduleManifest manifest = GoldenTraceStore.Read<ScheduleManifest>(manifestPath);
        ScheduleVariant variant = Assert.Single(
            manifest.Variants,
            value => string.Equals(value.TraceFile, traceFile, StringComparison.Ordinal));
        Assert.Equal(scheduleId, variant.ScheduleId);
        Assert.Equal("frozen", variant.Status);

        string tracePath = Path.Combine(Path.GetDirectoryName(manifestPath)!, traceFile)
            .Replace('\\', '/');
        AssistantScenarioTrace trace = GoldenTraceStore.Read<AssistantScenarioTrace>(tracePath);
        Assert.Equal(scheduleId, trace.ScheduleId);
        Assert.Equal("ExactTrace", trace.OracleKind);
    }

    [Theory]
    [InlineData("barge-in-timeout.json", "CR-013")]
    [InlineData("speaker-pending-match.json", "CR-005")]
    [InlineData("stop-during-response.json", "CR-009")]
    public void ScenarioTraces_DeclareTheirIdAndExactOracle(
        string fileName,
        string scenarioId)
    {
        AssistantScenarioTrace trace = GoldenTraceStore.Read<AssistantScenarioTrace>(fileName);
        Assert.Equal(scenarioId, trace.ScenarioId);
        Assert.Equal("ExactTrace", trace.OracleKind);
    }
}
