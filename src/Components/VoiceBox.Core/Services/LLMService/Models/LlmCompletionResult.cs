namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 一次完整推理的结果。
    /// </summary>
    public sealed class LlmCompletionResult
    {
        /// <summary>
        /// 模型最终输出的可见文本 (已过滤掉 special token 与思考块)。
        /// </summary>
        public required string Text { get; init; }

        /// <summary>
        /// 思考模式启用时, 这里包含全部思考过程文本; 否则为空字符串。
        /// </summary>
        public string Thinking { get; init; } = string.Empty;

        /// <summary>
        /// 本次生成所用 token 数量 (包含提示 token + 新生成 token)。
        /// </summary>
        public int TotalTokens { get; init; }

        /// <summary>
        /// 提示 (prompt) 部分的 token 数量 (含多模态输入展开出的 audio/image soft token); 未知时为 0。
        /// </summary>
        public int PromptTokens { get; init; }

        /// <summary>
        /// 本次新生成的 token 数量 (语义同 transformers 的 completion tokens); 未知时为 0。
        /// </summary>
        public int CompletionTokens { get; init; }

        /// <summary>
        /// 本次推理墙钟耗时 (秒)。
        /// </summary>
        public double ElapsedSeconds { get; init; }
    }
}
