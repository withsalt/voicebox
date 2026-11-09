using VoiceBox.Assistant.Conversation.Internal.Execution;
using System.Collections.Concurrent;
using System.Text.Json;
using Json.Schema;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>
/// Checks the argument object and optionally its advertised JSON Schema, returning validation
/// details to the model. Compiled schemas are cached by tool ID.
/// </summary>
internal sealed class ToolArgumentValidator(bool validateSchema = true)
{
    private const int MaxReportedViolations = 8;

    private readonly ConcurrentDictionary<ConversationToolId, JsonSchema?> _schemas = new();

    /// <summary>Returns JSON errors for invalid arguments. An unusable schema is skipped.</summary>
    public string? Validate(ToolDefinition definition, OwnedJsonValue arguments)
    {
        if (arguments.ToJsonElement().ValueKind != JsonValueKind.Object)
        {
            return JsonSerializer.Serialize(new ValidationFailure(
                "invalid_arguments",
                "Tool arguments must be a JSON object. Correct them and call the tool again.",
                [new Violation("", "Expected a JSON object.")]));
        }
        if (!validateSchema)
            return null;

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

    private static JsonSchema? TryCompile(ToolDefinition definition)
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

    private sealed record ValidationFailure(string Error, string Message, IReadOnlyList<Violation> Violations);

    private sealed record Violation(string Path, string Reason);
}
