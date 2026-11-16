using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Unicode;

namespace VoiceBox.Assistant.Conversation.Internal;

/// <summary>
/// JSON settings for everything this assembly sends to the model. Non-ASCII text such as CJK stays literal
/// instead of being inflated into \uXXXX escapes. HTML-sensitive ASCII such as '&lt;' stays escaped exactly as
/// with the default encoder, so tool text is no closer to spelling an angle-bracket chat-template token.
/// </summary>
internal static class ModelJson
{
    private static readonly JavaScriptEncoder Encoder = JavaScriptEncoder.Create(UnicodeRanges.All);
    private static readonly JsonWriterOptions WriterOptions = new() { Encoder = Encoder };

    public static JsonSerializerOptions SerializerOptions { get; } = new() { Encoder = Encoder };

    public static string Write(Action<Utf8JsonWriter> write)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer, WriterOptions))
            write(writer);
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    /// <summary>
    /// Property-sorted text without insignificant whitespace. Equal JSON values always produce equal text, so the
    /// text can identify a value and keeps provider transcripts byte-stable.
    /// </summary>
    public static string Canonicalize(JsonElement value) => Write(writer => WriteCanonical(writer, value));

    /// <summary>Canonical text of a JSON document given as text.</summary>
    public static string Canonicalize(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        return Canonicalize(document.RootElement);
    }

    /// <summary>Stable 64-bit FNV-1a digest for correlation and name shortening. It is not collision resistant.</summary>
    public static string StableHash(string value)
    {
        const ulong offsetBasis = 14695981039346656037UL;
        const ulong prime = 1099511628211UL;
        ulong hash = offsetBasis;
        foreach (byte valueByte in Encoding.UTF8.GetBytes(value))
        {
            hash ^= valueByte;
            hash *= prime;
        }
        return hash.ToString("x16", CultureInfo.InvariantCulture);
    }

    private static void WriteCanonical(Utf8JsonWriter writer, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (JsonProperty property in value.EnumerateObject().OrderBy(static property => property.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteCanonical(writer, property.Value);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (JsonElement item in value.EnumerateArray())
                    WriteCanonical(writer, item);
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
}
