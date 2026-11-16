using VoiceBox.Assistant.Api.State;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task OptionalWakeupStartFailure_ConcurrentStop_ConvergesToStoppedWithoutStartingAudio()
    {
        var services = TestServices.Create();
        services.Wakeup.StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Wakeup.StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Wakeup.StartException = new InvalidOperationException("wakeup start failure");
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        Task start = Task.Run(async () => await assistant.StartAsync());
        await services.Wakeup.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Task stop = assistant.StopAsync();
        Assert.False(stop.IsCompleted);

        services.Wakeup.StartBlock.TrySetResult();
        await Assert.ThrowsAsync<InvalidOperationException>(() => start);
        await stop;

        // Speaker evidence and the wakeup gate start before the audio frontend; a wakeup start
        // failure therefore never starts the frontend, and the already started speaker module is
        // rolled back.
        Assert.Equal(0, services.Audio.StartCount);
        Assert.Equal(1, services.Wakeup.StartCount);
        Assert.False(services.OwnerSpeaker.Last!.Started);
        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.False(services.Audio.IsRecording);
    }
}
