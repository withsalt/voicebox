using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Core.Services.LLMService
{
    /// <summary>
    /// 一段有状态的多轮对话。相对无状态的 <see cref="IStreamLLMService"/>, 会话可在内部保留
    /// KV-cache 等跨轮上下文, 使连续对话中重复的前缀无需每轮重新计算。
    ///
    /// 调用方式与无状态接口一致: 每轮仍传入<b>完整</b>的消息列表 (含 system 与既有历史);
    /// 会话内部自行对账并复用可复用的部分 —— 因此裁剪历史、回滚未回答的 user 消息等操作,
    /// 只需照常改变下一轮传入的消息列表即可, 无需额外通知会话。
    ///
    /// 会话<b>非线程安全</b>: 同一实例上的调用必须串行 (一轮结束再起下一轮)。
    /// 用完必须 <see cref="IDisposable.Dispose"/> 以释放持有的 native 资源。
    /// </summary>
    public interface ILLMConversation : IDisposable
    {
        /// <summary>
        /// 按 token 流式返回增量文本, 语义同 <see cref="IStreamLLMService.StreamAsync"/>。
        /// </summary>
        IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default);

        /// <summary>
        /// 一次性收集完整回复后返回, 语义同 <see cref="ILLMService.CompleteAsync"/>。
        /// </summary>
        Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default);
    }
}
