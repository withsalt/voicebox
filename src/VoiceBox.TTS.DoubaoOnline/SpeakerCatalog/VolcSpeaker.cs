using System.Text.Json.Serialization;

namespace VoiceBox.TTS.DoubaoOnline.SpeakerCatalog
{
    /// <summary>大模型音色列表 (ListSpeakers) 返回的单个音色。</summary>
    public sealed class VolcSpeaker
    {
        /// <summary>在 TTS 接口里使用的音色 ID。</summary>
        [JsonPropertyName("VoiceType")]
        public string VoiceType { get; set; } = string.Empty;

        /// <summary>音色名。</summary>
        [JsonPropertyName("Name")]
        public string? Name { get; set; }

        /// <summary>音色头像 URL。</summary>
        [JsonPropertyName("Avatar")]
        public string? Avatar { get; set; }

        /// <summary>性别。</summary>
        [JsonPropertyName("Gender")]
        public string? Gender { get; set; }

        /// <summary>年龄段。</summary>
        [JsonPropertyName("Age")]
        public string? Age { get; set; }

        /// <summary>普通标签。</summary>
        [JsonPropertyName("NormalLabels")]
        public List<string>? NormalLabels { get; set; }

        /// <summary>特殊标签。</summary>
        [JsonPropertyName("SpecialLabels")]
        public List<string>? SpecialLabels { get; set; }

        /// <summary>试听链接。</summary>
        [JsonPropertyName("TrialURL")]
        public string? TrialUrl { get; set; }

        /// <summary>语言信息。</summary>
        [JsonPropertyName("Languages")]
        public List<VolcSpeakerLanguage>? Languages { get; set; }

        /// <summary>情感信息。</summary>
        [JsonPropertyName("Emotions")]
        public List<VolcSpeakerEmotion>? Emotions { get; set; }

        /// <summary>描述。</summary>
        [JsonPropertyName("Description")]
        public string? Description { get; set; }

        /// <summary>音色对应的模型版本 (seed-tts-2.0 等)。</summary>
        [JsonPropertyName("ResourceID")]
        public string? ResourceId { get; set; }
    }

    public sealed class VolcSpeakerLanguage
    {
        [JsonPropertyName("Language")]
        public string? Language { get; set; }

        [JsonPropertyName("Text")]
        public string? Text { get; set; }

        [JsonPropertyName("Flag")]
        public string? Flag { get; set; }
    }

    public sealed class VolcSpeakerEmotion
    {
        [JsonPropertyName("Icon")]
        public string? Icon { get; set; }

        [JsonPropertyName("Label")]
        public string? Label { get; set; }

        [JsonPropertyName("Value")]
        public string? Value { get; set; }
    }

    /// <summary>ListSpeakers 返回的分页结果。</summary>
    public sealed class VolcSpeakerListResult
    {
        /// <summary>音色总个数。</summary>
        public int Total { get; init; }

        /// <summary>当前页音色列表。</summary>
        public IReadOnlyList<VolcSpeaker> Speakers { get; init; } = Array.Empty<VolcSpeaker>();
    }
}
