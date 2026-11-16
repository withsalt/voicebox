using System.Text.Json.Serialization;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// full server response 的 JSON payload。不同事件填充不同字段：句起/句终携带 <see cref="Text"/>，
    /// 字幕事件携带 <see cref="Words"/>，计费信息携带 <see cref="Usage"/>。
    /// </summary>
    internal sealed class ResponsePayload
    {
        /// <summary>合成音频对应文本 (句起/句终事件)。</summary>
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        /// <summary>字级别时间戳 (开启字幕时)。</summary>
        [JsonPropertyName("words")]
        public List<WordTimestamp>? Words { get; set; }

        /// <summary>音素级时间戳 (开启字幕时)。</summary>
        [JsonPropertyName("phonemes")]
        public List<WordTimestamp>? Phonemes { get; set; }

        /// <summary>资源消耗统计 (请求头开启计费返回时)。</summary>
        [JsonPropertyName("usage")]
        public UsageInfo? Usage { get; set; }
    }

    internal sealed class WordTimestamp
    {
        [JsonPropertyName("word")]
        public string? Word { get; set; }

        [JsonPropertyName("startTime")]
        public double StartTime { get; set; }

        [JsonPropertyName("endTime")]
        public double EndTime { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }
    }

    internal sealed class UsageInfo
    {
        /// <summary>本次请求计费的文本字数 (含标点)。</summary>
        [JsonPropertyName("text_words")]
        public int TextWords { get; set; }
    }

    /// <summary>错误帧 / 失败事件的 JSON payload (若可解析)。</summary>
    internal sealed class ErrorPayload
    {
        [JsonPropertyName("error")]
        public string? Error { get; set; }

        [JsonPropertyName("message")]
        public string? Message { get; set; }
    }
}
