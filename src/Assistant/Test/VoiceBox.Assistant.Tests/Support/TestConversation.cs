using VoiceBox.Assistant.Conversation;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Tests;

/// <summary>Creates default conversation sessions for Assistant integration tests.</summary>
internal static class TestConversation
{
    public static IConversationSession Create(
        IStreamLLMService llm,
        bool enableThinking = false,
        Action<LlmConversationOptions>? configure = null)
    {
        return new ConversationBuilder(llm)
            .UseOptions(options =>
            {
                options.EnableThinking = enableThinking;
                configure?.Invoke(options);
            })
            .Build();
    }

    public static IConversationSession CreateWithTools(
        IStreamLLMService llm,
        bool enableThinking,
        Action<ConversationToolsBuilder> configure,
        Action<LlmConversationOptions>? configureOptions = null)
    {
        return new ConversationBuilder(llm)
            .UseOptions(options =>
            {
                options.EnableThinking = enableThinking;
                configureOptions?.Invoke(options);
            })
            .UseTools(configure)
            .Build();
    }
}
