using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Internal;
using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 基础 LLM 会话: 有限内存历史、流式文本、思考输出、取消与资源释放。
/// </summary>
/// <remarks>
/// 构造完成即可使用, 不需要启动步骤。借用调用方的 <see cref="IStreamLLMService"/>; 会话创建并
/// 拥有 provider conversation。历史只在完全生成后提交, 取消或失败清理暂存状态。
/// </remarks>
public sealed class LlmConversationSession : IConversationSession
{
    private readonly ConversationSessionCore _core;

    internal LlmConversationSession(
        string conversationId,
        LlmConversationOptions options,
        IReadOnlyList<IConversationContextProvider> contextProviders,
        IStreamLLMService service,
        ILogger logger)
    {
        var history = new ConversationHistoryStore(options.SystemPrompt, options.MaxHistoryTurns);
        _core = new ConversationSessionCore(conversationId, history, options, contextProviders, new TextOnlyTurnExecutor(service), logger);
    }

    public string ConversationId => _core.ConversationId;

    public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
        => _core.StreamAsync(request, cancellationToken);

    public ValueTask DisposeAsync() => _core.DisposeAsync();
}
