namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具允许的执行位置。</summary>
public enum ConversationToolExecutionPolicy
{
    ForegroundOnly,
    BackgroundOnly,
    ModelChoice,
}
