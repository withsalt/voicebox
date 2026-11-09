namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具副作用风险等级。</summary>
public enum ConversationToolRisk
{
    ReadOnly,
    Sensitive,
    SideEffecting,
    Destructive,
    Privileged,
}
