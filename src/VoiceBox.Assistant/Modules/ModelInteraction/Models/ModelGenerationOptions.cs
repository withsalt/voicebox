using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal readonly record struct ModelGenerationOptions(
    int? MaxTokens = null,
    bool? DoSample = null,
    float? Temperature = null,
    int? TopK = null,
    float? TopP = null,
    float? RepetitionPenalty = null,
    bool EnableThinking = false)
{
    public static ModelGenerationOptions From(LlmCompleteOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new ModelGenerationOptions(
            options.MaxTokens,
            options.DoSample,
            options.Temperature,
            options.TopK,
            options.TopP,
            options.RepetitionPenalty,
            options.EnableThinking);
    }

    public LlmCompleteOptions ToProviderOptions()
        => new()
        {
            MaxTokens = MaxTokens,
            DoSample = DoSample,
            Temperature = Temperature,
            TopK = TopK,
            TopP = TopP,
            RepetitionPenalty = RepetitionPenalty,
            EnableThinking = EnableThinking,
        };
}
