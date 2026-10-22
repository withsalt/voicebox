using System.Collections.Concurrent;
using System.Text.Json;
using Json.Schema;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Integrations.Mcp;

/// <summary>
/// Checks tool arguments against the advertised JSON Schema and returns validation details to the
/// model. Compiled schemas are cached by tool ID.
/// </summary>
internal sealed class ToolArgumentValidator
{
    private const int MaxReportedViolations = 8;

    private readonly ConcurrentDictionary<AssistantToolId, JsonSchema?> _schemas = new();

    /// <summary>Returns null when validation passes or the schema is unusable; otherwise returns JSON errors.</summary>
    public string? Validate(AssistantToolDefinition definition, OwnedJsonValue arguments)
    {
        JsonSchema? schema = _schemas.GetOrAdd(definition.Id, _ => TryCompile(definition));
        if (schema is null)
            return null;

        EvaluationResults results = schema.Evaluate(
            arguments.ToJsonElement(),
            new EvaluationOptions
            {
                OutputFormat = OutputFormat.List,
                RequireFormatValidation = false,
            });
        if (results.IsValid)
            return null;

        return CreateViolationJson(results);
    }

    private static JsonSchema? TryCompile(AssistantToolDefinition definition)
    {
        try
        {
            return JsonSchema.FromText(definition.InputSchema.CanonicalJson);
        }
        catch (JsonException)
        {
            return null;
        }
        catch (ArgumentException)
        {
            return null;
        }
    }

    private static string CreateViolationJson(EvaluationResults results)
    {
        var violations = new List<Violation>();
        Collect(results, violations);
        if (violations.Count == 0)
            violations.Add(new Violation("", "The arguments do not satisfy the tool input schema."));

        return JsonSerializer.Serialize(new ValidationFailure(
            "invalid_arguments",
            "The arguments do not satisfy the tool input schema. Correct them and call the tool again.",
            violations.Take(MaxReportedViolations).ToArray()));
    }

    private static void Collect(EvaluationResults results, List<Violation> violations)
    {
        if (violations.Count >= MaxReportedViolations)
            return;

        if (!results.IsValid && results.Errors is { Count: > 0 } errors)
        {
            foreach ((string keyword, string message) in errors)
            {
                if (violations.Count >= MaxReportedViolations)
                    return;
                violations.Add(new Violation(
                    results.InstanceLocation.ToString(),
                    $"{keyword}: {message}"));
            }
        }

        if (results.Details is not { Count: > 0 } details)
            return;

        foreach (EvaluationResults nested in details)
        {
            if (nested.IsValid)
                continue;
            Collect(nested, violations);
        }
    }

    private sealed record ValidationFailure(
        string Error,
        string Message,
        IReadOnlyList<Violation> Violations);

    private sealed record Violation(string Path, string Reason);
}
