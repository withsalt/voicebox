using System.Collections.Generic;
using System.Threading;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Core.Services.LLMService
{
    /// <summary>
    /// 支持流式输出的 LLM 推理服务。
    /// </summary>
    public interface IStreamLLMService : ILLMService
    {
        /// <summary>
        /// 按 token 流式返回增量文本。文本块和思考块通过 <see cref="LlmTextChunk.Kind"/> 区分。
        /// </summary>
        IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default);
    }
}
