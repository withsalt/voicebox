namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 本轮回答已完成。一次正常请求恰好产生一个该事件, 并作为流的最后一个事件。
/// </summary>
/// <param name="Text">
/// 本轮全部可见文字增量的顺序拼接, 不含思考和工具参数。空回复或仅思考时为空字符串。
/// 该文字不能再送入 TTS。
/// </param>
public sealed record ConversationCompleted(string Text) : ConversationEvent
{
    /// <summary>本轮完整可见回答文字, 不得为 null。</summary>
    public string Text { get; } = Text ?? throw new ArgumentNullException(nameof(Text));
}
