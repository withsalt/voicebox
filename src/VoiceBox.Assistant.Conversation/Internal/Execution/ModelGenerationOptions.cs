using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

internal readonly record struct ModelGenerationOptions(
    int? MaxTokens = null,
    bool EnableThinking = false)
{
    public LlmCompleteOptions ToProviderOptions()
        => new()
        {
            MaxTokens = MaxTokens,
            EnableThinking = EnableThinking,
        };
}
