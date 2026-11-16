namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 思考过程增量。它不属于回答文字, 不进入 TTS, 也不进入正式历史。
/// </summary>
public sealed record ConversationThinkingDelta(string Text) : ConversationEvent
{
    /// <summary>思考文字, 不得为 null。</summary>
    public string Text { get; } = Text ?? throw new ArgumentNullException(nameof(Text));
}
