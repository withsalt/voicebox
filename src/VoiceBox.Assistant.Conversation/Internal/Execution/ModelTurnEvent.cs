using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

internal abstract record ModelTurnEvent;

internal sealed record ThinkingDelta(string Text) : ModelTurnEvent;

internal sealed record VisibleTextDelta(string Text) : ModelTurnEvent;

internal sealed record ModelTurnCompleted : ModelTurnEvent;

internal sealed record ModelTurnToolCall(
    string CallId,
    ConversationToolId ToolId,
    OwnedJsonValue Arguments) : ModelTurnEvent;
