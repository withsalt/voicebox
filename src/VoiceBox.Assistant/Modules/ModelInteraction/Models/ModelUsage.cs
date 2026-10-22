namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal readonly record struct ModelUsage(
    int PromptTokens = 0,
    int CompletionTokens = 0,
    int TotalTokens = 0);
