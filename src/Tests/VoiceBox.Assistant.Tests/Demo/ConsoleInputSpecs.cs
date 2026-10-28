using VoiceAssistantDemo.ConsoleUI;
using VoiceAssistantDemo.Presentation;
using VoiceBox.Assistant.Modules.Speaker.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests.Demo;

public sealed class ConsoleInputSpecs
{
    [Fact]
    public async Task RedirectedCommandsIgnoreBlankLinesAndAcceptLowercaseKeys()
    {
        using var reader = new StringReader("\n  s  \nr\n");
        var input = new ConsoleInput(reader, interactive: false);

        Assert.Equal(ConsoleKey.S, await input.ReadKeyAsync());
        Assert.Equal(ConsoleKey.R, await input.ReadKeyAsync());
        Assert.Equal(ConsoleKey.Q, await input.ReadKeyAsync());
    }

    [Fact]
    public async Task EnrollmentCanBeCancelledWhileWaitingForAStartCommand()
    {
        using var output = new StringWriter();
        using var display = new ConsoleDisplay(output, interactive: false);
        using var reader = new CancellableReader();
        var input = new ConsoleInput(reader, interactive: false);
        var interaction = new SpeakerEnrollmentConsoleInteraction(display, input, "测试唤醒词");
        using var cancellation = new CancellationTokenSource();

        Task waiting = interaction.WaitForSegmentStartAsync(
            new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.CleanConfirmation },
            cancellation.Token);
        Assert.Contains("测试唤醒词", output.ToString());
        Assert.False(waiting.IsCompleted);
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        display.WriteLines("取消后仍能显示");
        Assert.Contains("取消后仍能显示", output.ToString());
    }

    [Fact]
    public async Task EnrollmentExitsWhenRedirectedInputEnds()
    {
        using var output = new StringWriter();
        using var display = new ConsoleDisplay(output, interactive: false);
        using var reader = new StringReader(string.Empty);
        var interaction = new SpeakerEnrollmentConsoleInteraction(
            display, new ConsoleInput(reader, interactive: false), "测试唤醒词");

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => interaction.WaitForSegmentStartAsync(
            new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.CleanEnrollment },
            CancellationToken.None));
    }

    private sealed class CancellableReader : TextReader
    {
        public override async ValueTask<string?> ReadLineAsync(CancellationToken cancellationToken)
        {
            await Task.Delay(Timeout.Infinite, cancellationToken);
            return null;
        }
    }
}
