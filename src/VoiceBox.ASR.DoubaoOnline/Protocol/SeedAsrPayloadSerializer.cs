using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// 请求/响应 payload 的 JSON 序列化配置集中处。使用 source-unfriendly 的默认反射序列化即可，
    /// payload 很小、频率不高，无需 source generator。
    /// </summary>
    internal static class SeedAsrPayloadSerializer
    {
        private static readonly JsonSerializerOptions Options = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true,
        };

        public static string Serialize(RequestPayload payload)
            => JsonSerializer.Serialize(payload, Options);

        public static ResponsePayload? DeserializeResponse(string json)
            => JsonSerializer.Deserialize<ResponsePayload>(json, Options);

        public static ErrorPayload? TryDeserializeError(string json)
        {
            try { return JsonSerializer.Deserialize<ErrorPayload>(json, Options); }
            catch (JsonException) { return null; }
        }
    }
}
