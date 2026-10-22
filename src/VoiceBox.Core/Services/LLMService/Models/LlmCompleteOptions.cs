namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 单次推理的可覆盖参数。所有字段为 null 时使用 <c>genai_config.json</c> 内 search 段的默认值。
    /// </summary>
    public sealed class LlmCompleteOptions
    {
        /// <summary>
        /// 本次推理最多 <b>新生成</b> 多少 token (语义同 transformers 的 <c>max_new_tokens</c>)。
        /// 服务内部会自动把它换算为 ORT-GenAI 需要的 <c>max_length = promptTokens + MaxTokens</c>。
        /// null 表示沿用服务配置的默认值 (如 <c>OnnxGenAILLMOptions.MaxTokens</c>)。
        /// </summary>
        public int? MaxTokens { get; set; }

        /// <summary>
        /// 是否启用采样。null 表示沿用 manifest；false 表示贪心解码。
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
