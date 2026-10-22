namespace VoiceBox.Assistant.Api.Events;

/// <summary>
/// 一轮回复完整播完 (LLM 流终止且 TTS 播放完成) 时触发。
/// </summary>
public sealed class AssistantReplyEventArgs : EventArgs
{
    public required string Text { get; init; }
    public required TimeSpan Elapsed { get; init; }
}
