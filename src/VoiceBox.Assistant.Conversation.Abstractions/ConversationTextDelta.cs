namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 对外可见的回答文字增量。
/// </summary>
/// <param name="Text">增量文字。</param>
/// <param name="EndsTextSegment">
/// 表示一段可独立展示的文字已经结束。普通 token 输出为 false; 完整提示片段可以设为 true,
/// 消费方据此在继续拉取上游前把该片段送入自己的展示或语音管线。
/// </param>
public sealed record ConversationTextDelta(
    string Text,
    bool EndsTextSegment = false) : ConversationEvent
{
    /// <summary>增量文字, 不得为 null。</summary>
    public string Text { get; } = Text ?? throw new ArgumentNullException(nameof(Text));
}
