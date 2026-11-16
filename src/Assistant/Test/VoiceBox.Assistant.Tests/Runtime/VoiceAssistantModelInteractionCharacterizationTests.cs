using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task TextOnlyLlmContract_PreservesPromptOptionsThinkingVisibleAndFinalOrder()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Llm.EmitThinkingWhenEnabled = true;
        services.Llm.EmitSeparateFinalChunk = true;
        using var assistant = services.CreateAssistant(enableThinking: true);
        var events = new List<(string Text, bool IsThinking)>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.AssistantText += (_, value) => events.Add((value.Delta, value.IsThinking));
        assistant.AssistantReplied += (_, value) => replied.TrySetResult(value.Text);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));

        ChatMessage[] prompt = Assert.Single(services.Llm.Prompts);
        LlmCompleteOptions options = Assert.Single(services.Llm.StreamOptions);
        Assert.Collection(
            prompt,
            message => Assert.Equal(ChatRole.System, message.Role),
            message =>
            {
                Assert.Equal(ChatRole.User, message.Role);
                Assert.Equal("你好", Assert.IsType<TextContent>(Assert.Single(message.Content)).Text);
            });
        Assert.True(options.EnableThinking);
        Assert.Equal([("先分析一下。", true), ("好的。", false)], events);
        Assert.Equal(["好的。"], services.Tts.SynthesizedTexts);
    }
}
