namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 一轮会话输入，携带本轮用户文字、可选语音快照和调用方生成的关联 ID。
/// </summary>
/// <param name="RequestId">调用方为本次请求生成的关联 ID, 在同一会话内唯一。</param>
/// <param name="Text">本轮识别或输入的文字。</param>
/// <param name="Audio">与本轮 Text 对应的用户语音；纯文字输入或无音频时为 null。</param>
public sealed record ConversationRequest(Guid RequestId, string Text, ConversationAudio? Audio = null)
{
    /// <summary>请求 ID, 用于关联后台结果等异步输出; 不承诺网络重试幂等。</summary>
    public Guid RequestId { get; } = RequestId == Guid.Empty
        ? throw new ArgumentException("RequestId must not be empty.", nameof(RequestId))
        : RequestId;

    /// <summary>本轮用户文字, 不得为 null。</summary>
    public string Text { get; } = Text ?? throw new ArgumentNullException(nameof(Text));
}
