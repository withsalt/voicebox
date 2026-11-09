namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 请求结束之后由会话产生的后台输出, 例如后台工具任务的结果。
/// </summary>
/// <param name="MessageId">标识一次已生成的后台消息。</param>
/// <param name="ConversationId">产生该消息的业务会话身份。</param>
/// <param name="OriginatingRequestId">发起该后台工作的请求关联 ID。</param>
/// <param name="Text">可展示、可播报的完整后台文字; 不包含原始工具结果。</param>
public sealed record ConversationBackgroundMessage(
    Guid MessageId,
    string ConversationId,
    Guid OriginatingRequestId,
    string Text)
{
    /// <summary>消息 ID, 不得为空。</summary>
    public Guid MessageId { get; } = MessageId == Guid.Empty
        ? throw new ArgumentException("MessageId must not be empty.", nameof(MessageId))
        : MessageId;

    /// <summary>会话身份, 不得为空。</summary>
    public string ConversationId { get; } = string.IsNullOrWhiteSpace(ConversationId)
        ? throw new ArgumentException("ConversationId must not be empty.", nameof(ConversationId))
        : ConversationId;

    /// <summary>发起请求 ID, 不得为空。</summary>
    public Guid OriginatingRequestId { get; } = OriginatingRequestId == Guid.Empty
        ? throw new ArgumentException("OriginatingRequestId must not be empty.", nameof(OriginatingRequestId))
        : OriginatingRequestId;

    /// <summary>后台文字, 不得为 null。</summary>
    public string Text { get; } = Text ?? throw new ArgumentNullException(nameof(Text));
}
