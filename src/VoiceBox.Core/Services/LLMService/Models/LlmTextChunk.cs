namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 流式输出的一段增量文本。
    /// </summary>
    public sealed class LlmTextChunk
    {
        public required string Text { get; init; }

        public required LlmChunkKind Kind { get; init; }

        /// <summary>
        /// 本块是否为流的最后一块。仅供消费方做收尾, 不依赖此值判断停机条件。
        /// </summary>
        public bool IsFinal { get; init; }
    }
}
