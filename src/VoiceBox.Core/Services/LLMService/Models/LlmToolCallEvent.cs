namespace VoiceBox.Core.Services.LLMService.Models;

public sealed record LlmToolCallEvent(
    string CallId,
    string ToolName,
    string ArgumentsJson) : LlmTurnEvent;
