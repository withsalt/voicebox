using VoiceBox.Assistant.Conversation.Internal.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>One event of a single model round, already bound to the tool catalog.</summary>
internal abstract record ModelTurnEvent;

internal sealed record ThinkingDelta(string Text) : ModelTurnEvent;

internal sealed record VisibleTextDelta(string Text) : ModelTurnEvent;

internal sealed record ModelTurnToolCall(ToolCall Call) : ModelTurnEvent;
