namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal sealed record ToolCallRequested(
    string CallId,
    AssistantToolId ToolId,
    OwnedJsonValue Arguments) : ModelTurnEvent;
