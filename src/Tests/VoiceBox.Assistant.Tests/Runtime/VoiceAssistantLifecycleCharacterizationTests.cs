using VoiceBox.Assistant.Api.State;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task LifecycleContract_RepeatStartIsRejectedAndStopBeforeStartHasNoAudioSideEffect()
    {
        var beforeStartServices = TestServices.Create();
        using (var beforeStart = beforeStartServices.CreateAssistant())
        {
            await beforeStart.StopAsync();
            Assert.Equal(0, beforeStartServices.Audio.StartCount);
            Assert.Equal(0, beforeStartServices.Audio.StopCount);
        }

        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        await assistant.StartAsync();

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() => assistant.StartAsync());

        Assert.Equal("VoiceAssistant 已启动, 不可重复 Start。", exception.Message);
        Assert.Equal(1, services.Audio.StartCount);
        await assistant.StopAsync();

        InvalidOperationException afterStop = await Assert.ThrowsAsync<InvalidOperationException>(
            () => assistant.StartAsync());

        Assert.Equal("VoiceAssistant 已启动, 不可重复 Start。", afterStop.Message);
        Assert.Equal(1, services.Audio.StartCount);
        Assert.Equal(1, services.Audio.StopCount);
    }

    [Fact]
    public async Task LifecycleContract_StartFailureDetachesAndEndsStopped()
    {
        var services = TestServices.Create();
        services.Audio.StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant();

        Task start = assistant.StartAsync();
        await services.Audio.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        services.Audio.StartBlock.TrySetException(new InvalidOperationException("start failure"));

        await Assert.ThrowsAsync<InvalidOperationException>(() => start);

        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.Equal(0, services.Audio.StopCount);
        services.Vad.EmitSpeechStarted();
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task LifecycleContract_StopProviderFailureIsContainedAndTerminalStateIsPublished()
    {
        var services = TestServices.Create();
        services.Audio.StopException = new InvalidOperationException("stop failure");
        using var assistant = services.CreateAssistant();
        var stopped = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.StateChanged += (_, value) =>
        {
            if (value.To == AssistantState.Stopped)
                stopped.TrySetResult();
        };

        await assistant.StartAsync();
        await assistant.StopAsync();
        await stopped.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.Equal(0, services.Audio.StopCount);
    }
}
