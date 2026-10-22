using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Options;

namespace VoiceBox.TTS.DoubaoOnline.SpeakerCatalog
{
    /// <summary>
    /// 大模型音色列表 (ListSpeakers) 客户端。独立于 WebSocket 合成服务：管理类 OpenAPI 使用
    /// 火山引擎签名 V4 (AK/SK)，用于枚举某模型版本 (seed-tts-1.0 / seed-tts-2.0) 下的可用音色。
    /// </summary>
    public sealed class VolcSpeakerCatalogClient : IDisposable
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            PropertyNameCaseInsensitive = true,
        };

        private readonly VolcSpeakerCatalogOptions _options;
        private readonly HttpClient _http;
        private readonly bool _ownsHttp;

        public VolcSpeakerCatalogClient(IOptions<VolcSpeakerCatalogOptions> options)
            : this(options?.Value ?? throw new ArgumentNullException(nameof(options)), null) { }

        public VolcSpeakerCatalogClient(VolcSpeakerCatalogOptions options, HttpClient? http = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrWhiteSpace(_options.AccessKeyId) || string.IsNullOrWhiteSpace(_options.SecretAccessKey))
                throw new ArgumentException("VolcSpeakerCatalogOptions requires AccessKeyId and SecretAccessKey.");

            _ownsHttp = http is null;
            _http = http ?? new HttpClient();
            if (_ownsHttp && _options.TimeoutSec > 0)
                _http.Timeout = TimeSpan.FromSeconds(_options.TimeoutSec);
        }

        /// <summary>
        /// 拉取指定模型版本的音色列表。
        /// </summary>
        /// <param name="resourceIds">模型版本，例如 <c>["seed-tts-2.0"]</c>。默认 seed-tts-2.0。</param>
        /// <param name="voiceTypes">可选，按音色 ID 过滤。</param>
        /// <param name="page">页码 (从 1 开始)。</param>
        /// <param name="limit">页大小。</param>
        public async Task<VolcSpeakerListResult> ListSpeakersAsync(
            IReadOnlyList<string>? resourceIds = null,
            IReadOnlyList<string>? voiceTypes = null,
            int page = 1,
            int limit = 100,
            CancellationToken ct = default)
        {
            var body = new ListSpeakersRequest
            {
                ResourceIDs = resourceIds is { Count: > 0 } ? resourceIds : new[] { SeedTtsResourceIds.SeedTts20 },
                VoiceTypes = voiceTypes is { Count: > 0 } ? voiceTypes : null,
                Page = page,
                Limit = limit,
            };
            byte[] payload = JsonSerializer.SerializeToUtf8Bytes(body, JsonOptions);

            var query = new List<KeyValuePair<string, string>>
            {
                new("Action", "ListSpeakers"),
                new("Version", _options.Version),
            };

            var signReq = new VolcSignatureV4.Request(
                Method: "POST",
                Host: _options.Host,
                Path: "/",
                Query: query,
                Body: payload,
                AccessKeyId: _options.AccessKeyId,
                SecretAccessKey: _options.SecretAccessKey,
                Region: _options.Region,
                Service: _options.Service,
                UtcNow: DateTime.UtcNow);
            var authHeaders = VolcSignatureV4.Sign(signReq);

            string url = $"https://{_options.Host}/?Action=ListSpeakers&Version={Uri.EscapeDataString(_options.Version)}";
            using var request = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new ByteArrayContent(payload),
            };
            request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json")
            {
                CharSet = "UTF-8",
            };
            foreach (var kv in authHeaders)
                request.Headers.TryAddWithoutValidation(kv.Key, kv.Value);

            using var response = await _http.SendAsync(request, ct).ConfigureAwait(false);
            string json = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
                throw new DoubaoOnlineTTSException(
                    $"ListSpeakers failed: HTTP {(int)response.StatusCode}. Body: {Truncate(json, 512)}");

            ListSpeakersResponse? parsed;
            try { parsed = JsonSerializer.Deserialize<ListSpeakersResponse>(json, JsonOptions); }
            catch (JsonException ex) { throw new DoubaoOnlineTTSException("Failed to parse ListSpeakers response.", ex); }

            if (parsed?.Result is null)
                throw new DoubaoOnlineTTSException($"ListSpeakers returned no result. Body: {Truncate(json, 512)}");

            return new VolcSpeakerListResult
            {
                Total = parsed.Result.Total,
                Speakers = parsed.Result.Speakers ?? new List<VolcSpeaker>(),
            };
        }

        private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "…";

        public void Dispose()
        {
            if (_ownsHttp) _http.Dispose();
        }

        // ---- 请求/响应 DTO ----

        private sealed class ListSpeakersRequest
        {
            [JsonPropertyName("ResourceIDs")]
            public IReadOnlyList<string>? ResourceIDs { get; set; }

            [JsonPropertyName("VoiceTypes")]
            public IReadOnlyList<string>? VoiceTypes { get; set; }

            [JsonPropertyName("Page")]
            public int Page { get; set; }

            [JsonPropertyName("Limit")]
            public int Limit { get; set; }
        }

        private sealed class ListSpeakersResponse
        {
            [JsonPropertyName("Result")]
            public ResultBody? Result { get; set; }
        }

        private sealed class ResultBody
        {
            [JsonPropertyName("Total")]
            public int Total { get; set; }

            [JsonPropertyName("Speakers")]
            public List<VolcSpeaker>? Speakers { get; set; }
        }
    }
}
