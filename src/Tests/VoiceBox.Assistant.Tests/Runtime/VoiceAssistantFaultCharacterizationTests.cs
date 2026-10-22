using VoiceBox.Assistant.Api.State;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task RenderAwareDetectorFault_PublishesFaultBeforeStoppedWithoutFallback()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(enableBargeIn: true);
        var observed = new List<string>();
        var stopped = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.Faulted += (_, value) => observed.Add($"Faulted:{value.Component}");
        assistant.StateChanged += (_, value) =>
        {
            observed.Add($"StateChanged:{value.From}->{value.To}");
            if (value.To == AssistantState.Stopped)
                stopped.TrySetResult();
        };

        await assistant.StartAsync();
        services.PlaybackDetector.PushException = new InvalidOperationException("detector fault");
        services.Audio.EmitFrame(
            new float[320],
            new float[320],
            new float[320],
            new float[320]);

        await stopped.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(
            ["Faulted:render-aware-dtd", "StateChanged:Idle->Stopped"],
            observed);
        Assert.Equal(1, services.PlaybackDetector.PushCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.Equal(1, services.Audio.StopCount);
    }
}
