using VoiceBox.TTS.DoubaoOnline.Protocol;

namespace VoiceBox.TTS.DoubaoOnline.Session
{
    /// <summary>
    /// 把 <see cref="DoubaoOnlineTTSOptions"/> 翻译成握手所需的 URI、鉴权请求头，以及
    /// StartSession / TaskRequest 的 JSON payload。集中于此便于单测与复用。
    /// </summary>
    internal static class SeedTtsRequestBuilder
    {
        public static Uri BuildUri(DoubaoOnlineTTSOptions options)
        {
            string scheme = options.UseTls ? "wss" : "ws";
            string path = string.IsNullOrWhiteSpace(options.Path) ? "/api/v3/tts/bidirection" : options.Path;
            return new Uri($"{scheme}://{options.Host}{path}");
        }

        /// <summary>
        /// 构造握手请求头，使用新版控制台的 <c>X-Api-Key</c> 鉴权。
        /// </summary>
        public static Dictionary<string, string> BuildHeaders(DoubaoOnlineTTSOptions options, string requestId, string connectId)
        {
            var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["X-Api-Resource-Id"] = options.ResourceId,
                ["X-Api-Request-Id"] = requestId,
                ["X-Api-Connect-Id"] = connectId,
            };

            if (!string.IsNullOrWhiteSpace(options.ApiKey))
                headers["X-Api-Key"] = options.ApiKey!;

            if (options.ReturnUsage)
                headers["X-Control-Require-Usage-Tokens-Return"] = "*";

            if (options.ExtraHeaders is { Count: > 0 })
            {
                foreach (var kv in options.ExtraHeaders)
                    headers[kv.Key] = kv.Value;
            }

            return headers;
        }

        /// <summary>构造 StartSession 的 JSON payload。<paramref name="voice"/> 为解析后的音色 ID。</summary>
        public static string BuildStartSessionJson(DoubaoOnlineTTSOptions options, string voice, string uid)
        {
            var additions = new AdditionsConfig
            {
                DisableMarkdownFilter = options.DisableMarkdownFilter ? true : null,
                DisableEmojiFilter = options.DisableEmojiFilter ? true : null,
                ExplicitLanguage = string.IsNullOrWhiteSpace(options.ExplicitLanguage) ? null : options.ExplicitLanguage,
                ExplicitDialect = string.IsNullOrWhiteSpace(options.ExplicitDialect) ? null : options.ExplicitDialect,
            };

            var payload = new StartSessionPayload
            {
                User = string.IsNullOrWhiteSpace(uid) ? null : new UserConfig { Uid = uid },
                ReqParams = new ReqParams
                {
                    Speaker = voice,
                    Model = string.IsNullOrWhiteSpace(options.Model) ? null : options.Model,
                    AudioParams = new AudioParams
                    {
                        Format = "pcm",
                        SampleRate = options.SampleRate,
                        SpeechRate = options.SpeechRate,
                        LoudnessRate = options.LoudnessRate,
                        EnableSubtitle = options.EnableSubtitle ? true : null,
                    },
                    Additions = additions.HasAny ? SeedTtsPayloadSerializer.SerializeAdditions(additions) : null,
                    PostProcess = options.Pitch != 0 ? new PostProcess { Pitch = options.Pitch } : null,
                    ContextTexts = options.ContextTexts is { Count: > 0 } ? options.ContextTexts : null,
                    SectionId = string.IsNullOrWhiteSpace(options.SectionId) ? null : options.SectionId,
                },
            };

            return SeedTtsPayloadSerializer.Serialize(payload);
        }

        /// <summary>构造 TaskRequest 的 JSON payload (仅携带待合成文本)。</summary>
        public static string BuildTaskRequestJson(string text)
        {
            var payload = new TaskRequestPayload
            {
                ReqParams = new ReqParams { Text = text },
            };
            return SeedTtsPayloadSerializer.Serialize(payload);
        }

        /// <summary>连接级 / 结束会话事件的空 payload。</summary>
        public const string EmptyJson = "{}";
    }
}
