namespace VoiceBox.Core.Services.LLMService
{
    /// <summary>
    /// 支持创建有状态多轮对话的 LLM 服务。实现方通常能在会话内复用 KV-cache 等跨轮上下文,
    /// 从而降低长对话每轮的重复计算 (例如本地 ONNX 推理)。
    ///
    /// 远端 / 无状态实现 (如 OpenAI-compatible 服务) 可不实现本接口; 调用方应通过类型判断优雅回退到
    /// <see cref="IStreamLLMService.StreamAsync"/>。
    /// </summary>
    public interface IConversationalLLMService : IStreamLLMService
    {
        /// <summary>
        /// 创建一段新的有状态对话。返回的 <see cref="ILLMConversation"/> 由调用方拥有, 必须 Dispose。
        /// 每段会话独立持有自己的跨轮上下文, 互不影响。
        /// </summary>
        ILLMConversation CreateConversation();
    }
}
