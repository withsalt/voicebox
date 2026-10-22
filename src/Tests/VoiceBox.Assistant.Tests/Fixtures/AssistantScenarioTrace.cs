using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.Assistant.Tests;

internal sealed record AssistantScenarioTrace(
    string Schema,
    string CatalogVersion,
    string ScenarioId,
    string OracleKind,
    IReadOnlyList<ScenarioInput> Inputs,
    IReadOnlyList<ScenarioEvent> PublicEvents,
    IReadOnlyList<ScenarioProviderCall> ProviderCalls,
    ScenarioFinalState Final)
{
    public string? ScheduleId { get; init; }
}

internal sealed record ScenarioInput(int Ordinal, string Type, string Detail);

internal sealed record ScenarioEvent(
    int PublicationSeq,
    string Type,
    string? From = null,
    string? To = null,
    string? Text = null,
    bool? IsThinking = null);

internal sealed record ScenarioProviderCall(int Ordinal, string Type, string Detail);

internal sealed record ScenarioFinalState(string RuntimeState, string WakeMode);

internal sealed record KernelSemanticProjection(
    string Schema,
    string CatalogVersion,
    string ScenarioId,
    string OracleKind,
    IReadOnlyList<KernelSemanticStep> Steps);

internal sealed record KernelSemanticStep(
    int Ordinal,
    string Signal,
    string PreviousState,
    string NextState,
    IReadOnlyList<KernelSemanticEffect> Effects);

internal sealed record KernelSemanticEffect(
    int EffectIndex,
    string Type,
    string Detail);

internal sealed record ScheduleManifest(
    string Schema,
    string CatalogVersion,
    string ScenarioId,
    string OracleKind,
    IReadOnlyList<ScheduleVariant> Variants);

internal sealed record ScheduleVariant(
    string ScheduleId,
    string TraceFile,
    string HappensBefore,
    string Status,
    string Sha256);

internal static class AssistantScenarioTraceNormalizer
{
    public static AssistantScenarioTrace Normalize(AssistantScenarioTrace trace)
    {
        ArgumentNullException.ThrowIfNull(trace);

        return trace with
        {
            Inputs = trace.Inputs.OrderBy(static value => value.Ordinal).ToArray(),
            PublicEvents = trace.PublicEvents.OrderBy(static value => value.PublicationSeq).ToArray(),
            ProviderCalls = trace.ProviderCalls.OrderBy(static value => value.Ordinal).ToArray(),
        };
    }
}

internal static class GoldenTraceStore
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
    };

    public static T Read<T>(string relativePath)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "GoldenTraces", relativePath);
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(json, SerializerOptions)
            ?? throw new InvalidOperationException($"Golden trace was empty: {path}");
    }

    public static string Serialize<T>(T value)
        => JsonSerializer.Serialize(value, SerializerOptions);

    /// <summary>
    /// Hashes a golden artifact for schedule-manifest verification. Line endings are normalized so
    /// a checkout with CRLF cannot invalidate a recorded variant hash.
    /// </summary>
    public static string ComputeSha256(string relativePath)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "GoldenTraces", relativePath);
        string normalized = File.ReadAllText(path).Replace("\r\n", "\n", StringComparison.Ordinal);
        return Convert.ToHexString(
            System.Security.Cryptography.SHA256.HashData(
                System.Text.Encoding.UTF8.GetBytes(normalized)))
            .ToLowerInvariant();
    }
}
