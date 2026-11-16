using System.Text.Json.Serialization;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// StartSession 事件的 JSON payload。<c>event</c> 与 <c>session_id</c> 位于二进制帧字段，不在 JSON 中。
    /// 未设置 (null) 的字段不参与序列化，交由服务端使用默认值。
    /// </summary>
    internal sealed class StartSessionPayload
    {
        [JsonPropertyName("user")]
        public UserConfig? User { get; set; }

        [JsonPropertyName("namespace")]
        public string Namespace { get; set; } = "BidirectionalTTS";

        [JsonPropertyName("req_params")]
        public ReqParams ReqParams { get; set; } = new();
    }

    /// <summary>TaskRequest 事件的 JSON payload：仅携带待合成文本 (置于 req_params.text)。</summary>
    internal sealed class TaskRequestPayload
    {
        [JsonPropertyName("namespace")]
        public string Namespace { get; set; } = "BidirectionalTTS";

        [JsonPropertyName("req_params")]
        public ReqParams ReqParams { get; set; } = new();
    }

    internal sealed class UserConfig
    {
        [JsonPropertyName("uid")]
        public string? Uid { get; set; }
    }

    internal sealed class ReqParams
    {
        /// <summary>音色 ID。</summary>
        [JsonPropertyName("speaker")]
        public string? Speaker { get; set; }

        /// <summary>具体模型版本 (复刻音色时使用)。</summary>
        [JsonPropertyName("model")]
        public string? Model { get; set; }

        /// <summary>待合成文本 (仅 TaskRequest 使用)。</summary>
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("audio_params")]
        public AudioParams? AudioParams { get; set; }

        /// <summary>文本处理开关，接口要求为 JSON 编码后的字符串 (不是嵌套对象)。</summary>
        [JsonPropertyName("additions")]
        public string? Additions { get; set; }

        [JsonPropertyName("post_process")]
        public PostProcess? PostProcess { get; set; }

        /// <summary>语音指令数组。</summary>
        [JsonPropertyName("context_texts")]
        public IReadOnlyList<string>? ContextTexts { get; set; }

        /// <summary>多轮会话 ID。</summary>
        [JsonPropertyName("section_id")]
        public string? SectionId { get; set; }
    }

    internal sealed class AudioParams
    {
        [JsonPropertyName("format")]
        public string Format { get; set; } = "pcm";

        [JsonPropertyName("sample_rate")]
        public int SampleRate { get; set; } = 24000;

        [JsonPropertyName("speech_rate")]
        public int SpeechRate { get; set; }

        [JsonPropertyName("loudness_rate")]
        public int LoudnessRate { get; set; }

        [JsonPropertyName("enable_subtitle")]
        public bool? EnableSubtitle { get; set; }
    }

    internal sealed class PostProcess
    {
        [JsonPropertyName("pitch")]
        public int Pitch { get; set; }
    }

    /// <summary>
    /// <see cref="ReqParams.Additions"/> 内部字段。序列化为 JSON 字符串后放入 additions。
    /// 全字段可空：仅当有非默认值时才写入，避免发送冗余项。
    /// </summary>
    internal sealed class AdditionsConfig
    {
        [JsonPropertyName("disable_markdown_filter")]
        public bool? DisableMarkdownFilter { get; set; }

        [JsonPropertyName("disable_emoji_filter")]
        public bool? DisableEmojiFilter { get; set; }

        [JsonPropertyName("explicit_language")]
        public string? ExplicitLanguage { get; set; }

        [JsonPropertyName("explicit_dialect")]
        public string? ExplicitDialect { get; set; }

        /// <summary>是否有任一字段被设置 (决定是否需要序列化并发送 additions)。</summary>
        [JsonIgnore]
        public bool HasAny =>
            DisableMarkdownFilter is not null ||
            DisableEmojiFilter is not null ||
            !string.IsNullOrEmpty(ExplicitLanguage) ||
            !string.IsNullOrEmpty(ExplicitDialect);
    }
}
