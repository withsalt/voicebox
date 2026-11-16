namespace VoiceBox.LLM.OpenAI
{
    /// <summary>
    /// OpenAI reasoning 模型的推理强度, 映射到 Chat Completions 的 <c>reasoning_effort</c>。
    /// </summary>
    public enum OpenAIReasoningEffortLevel
    {
        /// <summary>
        /// 请求模型不使用 reasoning。仅部分较新的官方 OpenAI reasoning 模型支持; 火山方舟关闭思考请优先使用
        /// <see cref="ThinkingMode.Disabled"/> 或 <see cref="Minimal"/>。
        /// </summary>
        None = 0,

        /// <summary>
        /// 最小 reasoning 强度。适合追求低延迟、低成本的简单对话。
        /// </summary>
        Minimal,

        /// <summary>
        /// 低 reasoning 强度。
        /// </summary>
        Low,

        /// <summary>
        /// 中等 reasoning 强度。
        /// </summary>
        Medium,

        /// <summary>
        /// 高 reasoning 强度。
        /// </summary>
        High,

        /// <summary>
        /// 最高 reasoning 强度。仅部分 OpenAI-compatible 模型支持, 如火山方舟的部分 DeepSeek 模型。
        /// </summary>
        Max,
    }
}
