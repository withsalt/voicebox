namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// Per-call inference options. Null sampling values use the service's configured defaults.
    /// </summary>
    public sealed class LlmCompleteOptions
    {
        /// <summary>
        /// Maximum number of <b>new</b> tokens to generate for this call.
        /// Null uses the service's configured default.
        /// </summary>
        public int? MaxTokens { get; set; }

        /// <summary>
        /// Whether to sample tokens. Null uses the service's default; false selects greedy decoding.
        /// </summary>
        public bool? DoSample { get; set; }

        public float? Temperature { get; set; }

        public int? TopK { get; set; }

        public float? TopP { get; set; }

        public float? RepetitionPenalty { get; set; }

        /// <summary>
        /// 是否启用思考 (chain-of-thought) 模式。仅在模型 chat template 支持时生效。
        /// </summary>
        public bool EnableThinking { get; set; }
    }
}
