using System.Text;
using System.Text.Json;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal sealed class OwnedJsonValue
{
    private readonly JsonElement _value;

    public OwnedJsonValue(JsonElement value)
    {
        _value = value.Clone();
        CanonicalJson = WriteCanonicalJson(_value);
        CanonicalHash = ComputeStableHash(CanonicalJson);
    }

    public string CanonicalJson { get; }

    public string CanonicalHash { get; }

    public JsonElement ToJsonElement() => _value.Clone();

    public static OwnedJsonValue Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument document = JsonDocument.Parse(json);
        return new OwnedJsonValue(document.RootElement);
    }

    public override string ToString() => CanonicalJson;

    private static string WriteCanonicalJson(JsonElement value)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
            WriteCanonicalValue(writer, value);
        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static void WriteCanonicalValue(Utf8JsonWriter writer, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (JsonProperty property in value.EnumerateObject()
                    .OrderBy(static property => property.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteCanonicalValue(writer, property.Value);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (JsonElement item in value.EnumerateArray())
                    WriteCanonicalValue(writer, item);
                writer.WriteEndArray();
                break;
            case JsonValueKind.String:
                writer.WriteStringValue(value.GetString());
                break;
            case JsonValueKind.Number:
                writer.WriteRawValue(value.GetRawText(), skipInputValidation: false);
                break;
            case JsonValueKind.True:
                writer.WriteBooleanValue(true);
                break;
            case JsonValueKind.False:
                writer.WriteBooleanValue(false);
                break;
            case JsonValueKind.Null:
                writer.WriteNullValue();
                break;
            default:
                throw new ArgumentException($"Unsupported JSON value kind '{value.ValueKind}'.", nameof(value));
        }
    }

    internal static string ComputeStableHash(string value)
    {
        const ulong offsetBasis = 14695981039346656037UL;
        const ulong prime = 1099511628211UL;
        ulong hash = offsetBasis;
        foreach (byte valueByte in Encoding.UTF8.GetBytes(value))
        {
            hash ^= valueByte;
            hash *= prime;
        }
        return hash.ToString("x16", System.Globalization.CultureInfo.InvariantCulture);
    }
}
