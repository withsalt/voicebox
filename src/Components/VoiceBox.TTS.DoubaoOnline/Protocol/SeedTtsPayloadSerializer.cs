using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// 请求/响应 payload 的 JSON 序列化配置集中处。payload 很小、频率不高，使用默认反射序列化即可。
    /// </summary>
    internal static class SeedTtsPayloadSerializer
    {
        private static readonly JsonSerializerOptions Options = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        public static string Serialize(StartSessionPayload payload)
            => JsonSerializer.Serialize(payload, Options);

        public static string Serialize(TaskRequestPayload payload)
            => JsonSerializer.Serialize(payload, Options);

        public static string SerializeAdditions(AdditionsConfig additions)
            => JsonSerializer.Serialize(additions, Options);

        public static ResponsePayload? DeserializeResponse(string json)
            => JsonSerializer.Deserialize<ResponsePayload>(json, Options);

        public static ErrorPayload? TryDeserializeError(string json)
        {
            try { return JsonSerializer.Deserialize<ErrorPayload>(json, Options); }
            catch (JsonException) { return null; }
        }
    }
}
