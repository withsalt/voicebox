namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 会话的可选后台输出能力。只有能够产生请求结束之后结果的会话实现该接口。
/// </summary>
/// <remarks>
/// Start 创建本次运行的后台通道; Stop 取消阻塞的读写并完成通道。重新 Start 创建新通道,
/// 旧消费者不能读到新一轮输出。消息按会话提交顺序发布, 只保证一次运行内单次、有序消费。
/// </remarks>
public interface IConversationBackgroundOutput : IConversationLifecycle
{
    /// <summary>
    /// 读取本次运行的后台输出。消费者必须使用 Start/Stop 之外的运行取消令牌结束阻塞读取。
    /// </summary>
    IAsyncEnumerable<ConversationBackgroundMessage> ReadBackgroundMessagesAsync(CancellationToken cancellationToken = default);
}
