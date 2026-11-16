using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Core.Services.LLMService;

/// <summary>
/// 支持 provider 原生结构化工具调用的 LLM 服务。实现方必须产生 typed tool-call 事件，
/// 不得通过解析已经对用户可见的普通文本模拟工具调用。
/// </summary>
/// <remarks>
/// 如果实现方会在同一轮既输出可见文字又发起工具调用, 调用方会在后续请求的历史中回传一条同时包含
/// <see cref="TextContent"/> 与 <see cref="ToolCallContent"/> 的 assistant 消息 (文字在前, 调用在后),
/// 实现方必须接受这种消息。每轮只输出其中一种的实现方不会收到它。
/// </remarks>
public interface IToolCallingStreamLLMService : IStreamLLMService
{
    IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(LlmToolTurnRequest request, CancellationToken cancellationToken = default);
}
