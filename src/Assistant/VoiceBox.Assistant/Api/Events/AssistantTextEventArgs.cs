namespace VoiceBox.Assistant.Api.Events;

/// <summary>
/// LLM 流式输出过程中的增量文本。<see cref="IsThinking"/> 区分可见回答与 thinking。
/// thinking 只供 UI 展示, 不会被 VoiceAssistant 送入 TTS。
/// </summary>
public sealed class AssistantTextEventArgs : EventArgs
{
    public required string Delta { get; init; }

    /// <summary>
    /// true 表示本增量是模型思考过程; false 表示正式回答文本。
    /// </summary>
    public required bool IsThinking { get; init; }
}
