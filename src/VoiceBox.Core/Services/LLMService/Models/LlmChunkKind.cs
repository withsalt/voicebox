namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 流式输出分类。
    /// </summary>
    public enum LlmChunkKind
    {
        /// <summary>对外可见的正常文本输出。</summary>
        Text = 0,

        /// <summary>思考 / 推理过程片段, 通常用于侧栏或 stderr 显示, 不进入最终回答。</summary>
        Thinking = 1,
    }
}
