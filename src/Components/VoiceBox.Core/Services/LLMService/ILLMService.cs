using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Core.Services.LLMService
{
    /// <summary>
    /// 大语言模型推理服务。支持多模态消息输入 (文本 + 音频 + 图像), 输出纯文本。
    /// </summary>
    public interface ILLMService : IDisposable
    {
        /// <summary>
        /// 同步收集完整回复后一次性返回。内部仍然按 token 流式生成。
        /// </summary>
        Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default);
    }
}
