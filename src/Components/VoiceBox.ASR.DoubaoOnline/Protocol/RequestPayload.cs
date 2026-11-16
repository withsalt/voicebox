using System.Text.Json.Serialization;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// full client request 的 payload。序列化为 JSON 后 (可选) Gzip 压缩发送。
    /// 未设置 (null) 的字段不参与序列化，交由服务端使用默认值。
    /// </summary>
    internal sealed class RequestPayload
    {
        [JsonPropertyName("user")]
        public UserConfig? User { get; set; }

        [JsonPropertyName("audio")]
        public AudioConfig Audio { get; set; } = new();

        [JsonPropertyName("request")]
        public RequestConfig Request { get; set; } = new();
    }

    internal sealed class UserConfig
    {
        [JsonPropertyName("uid")]
        public string? Uid { get; set; }
    }

    internal sealed class AudioConfig
    {
        [JsonPropertyName("format")]
        public string Format { get; set; } = "pcm";

        [JsonPropertyName("codec")]
        public string Codec { get; set; } = "raw";

        [JsonPropertyName("rate")]
        public int Rate { get; set; } = 16000;

        [JsonPropertyName("bits")]
        public int Bits { get; set; } = 16;

        [JsonPropertyName("channel")]
        public int Channel { get; set; } = 1;
    }

    internal sealed class RequestConfig
    {
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = "bigmodel";

        [JsonPropertyName("enable_itn")]
        public bool? EnableItn { get; set; }

        [JsonPropertyName("enable_punc")]
        public bool? EnablePunc { get; set; }

        [JsonPropertyName("enable_ddc")]
        public bool? EnableDdc { get; set; }

        [JsonPropertyName("show_utterances")]
        public bool? ShowUtterances { get; set; }

        [JsonPropertyName("result_type")]
        public string? ResultType { get; set; }

        [JsonPropertyName("end_window_size")]
        public int? EndWindowSize { get; set; }

        [JsonPropertyName("corpus")]
        public CorpusConfig? Corpus { get; set; }
    }

    internal sealed class CorpusConfig
    {
        /// <summary>热词直传，JSON string，形如 <c>{"hotwords":[{"word":"热词1"}]}</c>。</summary>
        [JsonPropertyName("context")]
        public string? Context { get; set; }
    }
}
