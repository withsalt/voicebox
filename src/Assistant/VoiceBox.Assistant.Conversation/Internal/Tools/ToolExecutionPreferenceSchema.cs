using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>
/// Adds the model-visible execution preference to a tool schema and extracts it from the model's
/// arguments before the tool sees them.
/// </summary>
internal static class ToolExecutionPreferenceSchema
{
    private const string PropertyName = "__execution";

    public static string AddTo(string inputSchemaJson, ConversationToolExecutionPolicy policy)
    {
        using JsonDocument document = JsonDocument.Parse(inputSchemaJson);
        return ModelJson.Write(writer =>
        {
            writer.WriteStartObject();
            bool wroteProperties = false;
            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                if (property.NameEquals("properties"))
                {
                    writer.WritePropertyName("properties");
                    writer.WriteStartObject();
                    foreach (JsonProperty inner in property.Value.EnumerateObject())
                        inner.WriteTo(writer);
                    WritePreferenceProperty(writer, policy);
                    writer.WriteEndObject();
                    wroteProperties = true;
                }
                else
                {
                    property.WriteTo(writer);
                }
            }

            if (!wroteProperties)
            {
                writer.WritePropertyName("properties");
                writer.WriteStartObject();
                WritePreferenceProperty(writer, policy);
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
        });
    }

    private static void WritePreferenceProperty(Utf8JsonWriter writer, ConversationToolExecutionPolicy policy)
    {
        writer.WritePropertyName(PropertyName);
        writer.WriteStartObject();
        writer.WritePropertyName("type");
        writer.WriteStringValue("string");
        writer.WritePropertyName("enum");
        writer.WriteStartArray();
        switch (policy)
        {
            case ConversationToolExecutionPolicy.ForegroundOnly:
                writer.WriteStringValue("foreground");
                break;
            case ConversationToolExecutionPolicy.BackgroundOnly:
                writer.WriteStringValue("background");
                break;
            case ConversationToolExecutionPolicy.ModelChoice:
                writer.WriteStringValue("foreground");
                writer.WriteStringValue("background");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(policy), policy, null);
        }
        writer.WriteEndArray();
        writer.WritePropertyName("description");
        writer.WriteStringValue("Choose whether to run this tool in the foreground (the reply waits) or in the background.");
        writer.WriteEndObject();
    }

    /// <summary>
    /// Removes the preference from the model's arguments. Arguments that are not a JSON object are returned
    /// unchanged; argument validation reports them to the model.
    /// </summary>
    public static (JsonElement Arguments, ConversationToolExecutionPreference Preference) Extract(
        JsonElement arguments,
        ConversationToolExecutionPolicy policy)
    {
        if (arguments.ValueKind != JsonValueKind.Object || !arguments.TryGetProperty(PropertyName, out JsonElement preferenceElement))
            return (arguments, DefaultPreference(policy));

        string strippedJson = ModelJson.Write(writer =>
        {
            writer.WriteStartObject();
            foreach (JsonProperty property in arguments.EnumerateObject())
            {
                if (!property.NameEquals(PropertyName))
                    property.WriteTo(writer);
            }
            writer.WriteEndObject();
        });
        using JsonDocument stripped = JsonDocument.Parse(strippedJson);

        ConversationToolExecutionPreference preference = preferenceElement.ValueKind == JsonValueKind.String &&
            string.Equals(preferenceElement.GetString(), "background", StringComparison.Ordinal)
                ? ConversationToolExecutionPreference.Background
                : ConversationToolExecutionPreference.Foreground;
        return (stripped.RootElement.Clone(), Coerce(preference, policy));
    }

    private static ConversationToolExecutionPreference DefaultPreference(ConversationToolExecutionPolicy policy) => policy switch
    {
        ConversationToolExecutionPolicy.BackgroundOnly => ConversationToolExecutionPreference.Background,
        _ => ConversationToolExecutionPreference.Foreground,
    };

    private static ConversationToolExecutionPreference Coerce(
        ConversationToolExecutionPreference preference,
        ConversationToolExecutionPolicy policy) => policy switch
    {
        ConversationToolExecutionPolicy.ForegroundOnly => ConversationToolExecutionPreference.Foreground,
        ConversationToolExecutionPolicy.BackgroundOnly => ConversationToolExecutionPreference.Background,
        _ => preference,
    };
}
