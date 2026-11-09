namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 一个业务会话实例。会话负责理解输入、组织上下文、生成回答并管理自己的对话状态。
/// </summary>
/// <remarks>
/// <para>
/// 同一会话实例上的请求按串行语义处理; 不同实例互不影响。实现必须支持取消, 并在提前结束枚举
/// 或释放时清理未提交的暂存状态。
/// </para>
/// <para>
/// 一次正常请求输出零个或多个 <see cref="ConversationTextDelta"/> /
/// <see cref="ConversationThinkingDelta"/>, 随后恰好一个 <see cref="ConversationCompleted"/>,
/// 之后结束。故障通过异步枚举异常传播, 取消使用 <see cref="OperationCanceledException"/>。
/// </para>
/// </remarks>
public interface IConversationSession : IAsyncDisposable
{
    /// <summary>
    /// 业务会话身份。实例生命周期内稳定、非空。它表示业务会话, 不表示麦克风开关或一次唤醒。
    /// </summary>
    string ConversationId { get; }

    /// <summary>
    /// 执行一轮请求并返回输出流。返回流按单次消费设计, 不保证可重复枚举。
    /// </summary>
    IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default);
}
