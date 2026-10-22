using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Core.Services.LLMService;

/// <summary>
/// 支持 provider 原生结构化工具调用的 LLM 服务。实现方必须产生 typed tool-call 事件，
/// 不得通过解析已经对用户可见的普通文本模拟工具调用。
/// </summary>
public interface IToolCallingStreamLLMService : IStreamLLMService
{
    IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
        LlmToolTurnRequest request,
        CancellationToken cancellationToken = default);
}
