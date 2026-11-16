namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具调用的规范化结果分类。</summary>
public enum ConversationToolResultKind
{
    Accepted,
    Success,
    Denied,
    Cancelled,
    TimedOut,
    ToolError,
    ProtocolError,
    InputRequired,
    UnsupportedCapability,
    UnknownOutcome,
}
