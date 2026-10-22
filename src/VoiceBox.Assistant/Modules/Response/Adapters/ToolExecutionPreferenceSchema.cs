using System.Text.Json;
using System.Text.Json.Nodes;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

internal static class ToolExecutionPreferenceSchema
{
    internal const string PropertyName = "__voicebox_execution";

    public static string AddTo(string schemaJson, AssistantToolExecutionPolicy policy)
    {
        if (policy != AssistantToolExecutionPolicy.ModelChoice)
            return schemaJson;

        JsonObject schema = JsonNode.Parse(schemaJson)?.AsObject()
            ?? throw new ArgumentException("Tool input schema must be a JSON object.", nameof(schemaJson));
        JsonObject properties = schema["properties"] as JsonObject ?? new JsonObject();
        if (properties.ContainsKey(PropertyName))
        {
            throw new InvalidOperationException(
                $"Tool input schema reserves property '{PropertyName}'.");
        }

        properties[PropertyName] = new JsonObject
        {
            ["type"] = "string",
            ["enum"] = new JsonArray("foreground", "background"),
            ["description"] = "Choose background for a task that may take noticeable time; otherwise choose foreground.",
        };
        schema["properties"] = properties;
        JsonArray required = schema["required"] as JsonArray ?? new JsonArray();
        required.Add(PropertyName);
        schema["required"] = required;
        return schema.ToJsonString(new JsonSerializerOptions { WriteIndented = false });
    }

    public static (OwnedJsonValue Arguments, AssistantToolExecutionPreference Preference) Extract(
        OwnedJsonValue arguments,
        AssistantToolExecutionPolicy policy)
    {
        AssistantToolExecutionPreference preference = policy switch
        {
            AssistantToolExecutionPolicy.ForegroundOnly => AssistantToolExecutionPreference.Foreground,
            AssistantToolExecutionPolicy.BackgroundOnly => AssistantToolExecutionPreference.Background,
            AssistantToolExecutionPolicy.ModelChoice => ReadPreference(arguments),
            _ => throw new ArgumentOutOfRangeException(nameof(policy), policy, null),
        };

        JsonObject value = JsonNode.Parse(arguments.CanonicalJson)?.AsObject()
            ?? throw new InvalidOperationException("Tool arguments must be a JSON object.");
        // This is provider orchestration metadata, never part of the real tool input. Strip it
        // even when a model sends it to a host-forced foreground/background tool.
        value.Remove(PropertyName);
        return (OwnedJsonValue.Parse(value.ToJsonString()), preference);
    }

    private static AssistantToolExecutionPreference ReadPreference(OwnedJsonValue arguments)
    {
        JsonElement value = arguments.ToJsonElement();
        if (value.ValueKind != JsonValueKind.Object ||
            !value.TryGetProperty(PropertyName, out JsonElement property) ||
            property.ValueKind != JsonValueKind.String)
        {
            return AssistantToolExecutionPreference.Foreground;
        }

        return property.GetString() switch
        {
            "background" => AssistantToolExecutionPreference.Background,
            "foreground" => AssistantToolExecutionPreference.Foreground,
            _ => AssistantToolExecutionPreference.Foreground,
        };
    }
}
