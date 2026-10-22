namespace VoiceBox.Assistant.Api.Events;

/// <summary>
/// AI 回复文本已完整生成时触发; 此时 TTS 可能仍在合成或播放。
/// </summary>
public sealed class AssistantResponseGeneratedEventArgs : EventArgs
{
    public required string Text { get; init; }
}
