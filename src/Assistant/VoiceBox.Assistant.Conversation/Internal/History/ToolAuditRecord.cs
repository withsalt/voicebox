using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.History;

internal sealed record ToolAuditRecord(long TurnId, string CallId, ConversationToolId ToolId, ConversationToolResultKind ResultKind);
