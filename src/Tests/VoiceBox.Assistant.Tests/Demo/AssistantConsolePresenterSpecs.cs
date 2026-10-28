using VoiceAssistantDemo.ConsoleUI;
using VoiceAssistantDemo.Presentation;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task DemoPresenterDisplaysThinkingAndReplyFromPublicAssistantEvents()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Llm.EmitThinkingWhenEnabled = true;
        services.Llm.EmitSeparateFinalChunk = true;
        using var output = new StringWriter();
        using var display = new ConsoleDisplay(output, interactive: false);
        using var assistant = services.CreateAssistant(enableThinking: true);
        using var presenter = new AssistantConsolePresenter(assistant, display, "测试唤醒词");
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, _) => replied.TrySetResult();

        await assistant.StartAsync();
        presenter.ShowReady();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        string text = output.ToString().Replace("\r\n", "\n");
        Assert.Contains("可以直接说话, 按 Q 退出", text);
        Assert.Contains("  我：你好", text);
        Assert.Contains("  [思考] 先分析一下。\n  助手：好的。\n\n", text);
        Assert.Equal(1, text.Split("  助手：", StringSplitOptions.None).Length - 1);
        Assert.DoesNotContain('\x1b', text);
    }

    [Fact]
    public async Task DemoPresenterStopsDisplayingEventsAfterDisposal()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var output = new StringWriter();
        using var display = new ConsoleDisplay(output, interactive: false);
        using var assistant = services.CreateAssistant();
        var presenter = new AssistantConsolePresenter(assistant, display, "测试唤醒词");
        presenter.Dispose();
        presenter.Dispose();
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, _) => replied.TrySetResult();

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        Assert.Empty(output.ToString());
    }
}
