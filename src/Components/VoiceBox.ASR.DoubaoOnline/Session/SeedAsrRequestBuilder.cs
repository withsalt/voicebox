using VoiceBox.ASR.DoubaoOnline.Protocol;

namespace VoiceBox.ASR.DoubaoOnline.Session
{
    /// <summary>
    /// 把 <see cref="DoubaoOnlineASROptions"/> 翻译成握手所需的 URI、鉴权请求头，
    /// 以及 full client request 的 payload。集中于此便于单测与复用。
    /// </summary>
    internal static class SeedAsrRequestBuilder
    {
        public static Uri BuildUri(DoubaoOnlineASROptions options)
        {
            string scheme = options.UseTls ? "wss" : "ws";
            string path = options.StreamMode.ToRequestPath();
            return new Uri($"{scheme}://{options.Host}{path}");
        }

        /// <summary>
        /// 构造握手请求头，使用新版控制台的 <c>X-Api-Key</c> 鉴权。
        /// </summary>
        public static Dictionary<string, string> BuildHeaders(DoubaoOnlineASROptions options, string requestId, string connectId)
        {
            var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["X-Api-Resource-Id"] = options.ResourceId,
                ["X-Api-Request-Id"] = requestId,
                ["X-Api-Connect-Id"] = connectId,
            };

            if (!string.IsNullOrWhiteSpace(options.ApiKey))
                headers["X-Api-Key"] = options.ApiKey!;

            if (options.ExtraHeaders is { Count: > 0 })
            {
                foreach (var kv in options.ExtraHeaders)
                    headers[kv.Key] = kv.Value;
            }

            return headers;
        }

        /// <summary>把识别相关配置映射到 full client request 的 JSON payload 字符串。</summary>
        public static string BuildRequestJson(DoubaoOnlineASROptions options)
        {
            var payload = new RequestPayload
            {
                User = string.IsNullOrWhiteSpace(options.Uid) ? null : new UserConfig { Uid = options.Uid },
                Audio = new AudioConfig
                {
                    Format = "pcm",
                    Codec = "raw",
                    Rate = options.SampleRate,
                    Bits = 16,
                    Channel = options.Channels,
                },
                Request = new RequestConfig
                {
                    ModelName = options.ModelName,
                    EnableItn = options.EnableItn,
                    EnablePunc = options.EnablePunc,
                    EnableDdc = options.EnableDdc,
                    ShowUtterances = options.ShowUtterances,
                    ResultType = string.IsNullOrWhiteSpace(options.ResultType) ? null : options.ResultType,
                    EndWindowSize = options.EndWindowSize > 0 ? options.EndWindowSize : null,
                    Corpus = string.IsNullOrWhiteSpace(options.HotWords)
                        ? null
                        : new CorpusConfig { Context = options.HotWords },
                },
            };

            return SeedAsrPayloadSerializer.Serialize(payload);
        }
    }
}
