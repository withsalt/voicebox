namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 在构建每轮模型请求前提供附加上下文 (记忆、检索结果等)。返回空字符串表示没有附加上下文。
/// </summary>
/// <remarks>
/// 每轮请求只调用一次; 同一轮流式工具续推复用本轮背景, 不重复检索。默认检索失败使本轮失败
/// 且不提交; 希望降级为空上下文的实现应自行处理可预期故障。
/// </remarks>
public interface IConversationContextProvider
{
    ValueTask<string> GetContextAsync(ConversationContext context, CancellationToken cancellationToken = default);
}
