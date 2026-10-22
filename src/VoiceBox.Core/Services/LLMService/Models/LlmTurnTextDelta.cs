namespace VoiceBox.Core.Services.LLMService.Models;

public sealed record LlmTurnTextDelta(
    string Text,
    LlmChunkKind Kind = LlmChunkKind.Text) : LlmTurnEvent;
