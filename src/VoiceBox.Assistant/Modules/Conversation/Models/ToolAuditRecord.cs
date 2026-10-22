using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.Conversation.Models;

internal sealed record ToolAuditRecord(
        long TurnId,
        string CallId,
        AssistantToolId ToolId,
        AssistantToolResultKind ResultKind);
