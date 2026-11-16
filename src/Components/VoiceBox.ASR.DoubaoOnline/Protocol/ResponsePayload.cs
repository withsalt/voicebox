using System.Text.Json.Serialization;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// full server response 的 payload JSON 结构。字段按需填充，未开启的能力对应字段可能缺省。
    /// </summary>
    internal sealed class ResponsePayload
    {
        [JsonPropertyName("result")]
        public ResultBody? Result { get; set; }

        [JsonPropertyName("audio_info")]
        public AudioInfo? AudioInfo { get; set; }
    }

    internal sealed class ResultBody
    {
        /// <summary>整个音频到目前为止的识别结果文本。</summary>
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        /// <summary>分句信息，仅在 show_utterances=true 时下发。</summary>
        [JsonPropertyName("utterances")]
        public List<Utterance>? Utterances { get; set; }
    }

    internal sealed class Utterance
    {
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        /// <summary>是否为已确定分句 (VAD 判停或二遍识别确认)。</summary>
        [JsonPropertyName("definite")]
        public bool Definite { get; set; }

        [JsonPropertyName("start_time")]
        public int StartTime { get; set; }

        [JsonPropertyName("end_time")]
        public int EndTime { get; set; }
    }

    internal sealed class AudioInfo
    {
        /// <summary>已处理音频时长 (毫秒)。</summary>
        [JsonPropertyName("duration")]
        public int Duration { get; set; }
    }

    /// <summary>错误帧的 JSON payload (若可解析)。</summary>
    internal sealed class ErrorPayload
    {
        [JsonPropertyName("error")]
        public string? Error { get; set; }

        [JsonPropertyName("message")]
        public string? Message { get; set; }
    }
}
