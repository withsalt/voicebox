namespace VoiceBox.Core.Services.LLMService.Models;

public sealed record LlmTurnCompletedEvent(
    int PromptTokens = 0,
    int CompletionTokens = 0,
    int TotalTokens = 0) : LlmTurnEvent;
