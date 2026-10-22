using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task AcousticCandidate_DucksEarlyAndRestoresOriginalGainAfterFalseAlarm()
    {
        var services = TestServices.Create();
        using var assistant = await StartDuckingResponseAsync(services);
        services.Audio.SetPlaybackGain(0.6f);
        services.Audio.PlaybackGainChanges.Clear();

        EmitNearEndFrames(services, count: 2, renderRms: 0.08, captureRms: 0.09);

        Assert.Equal(0.15f, services.Audio.PlaybackGain);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Equal(AssistantState.Speaking, assistant.State);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));

        EmitSilenceFrames(services, count: 9, renderRms: 0.08);
        Assert.Equal(0.15f, services.Audio.PlaybackGain);
        EmitSilenceFrames(services, count: 1, renderRms: 0.08);

        Assert.Equal(new[] { 0.15f, 0.6f }, services.Audio.PlaybackGainChanges);
        Assert.Equal(AssistantState.Speaking, assistant.State);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Single(services.Llm.Prompts);

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        Assert.Equal(2, services.Audio.PlaybackGainChanges.Count);
    }

    [Fact]
    public async Task AcousticCandidate_ConfirmedAfterDuckingStopsPlaybackBeforeAsrResult()
    {
        var services = TestServices.Create();
        using var assistant = await StartDuckingResponseAsync(services);
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        EmitNearEndFrames(services, count: 3, renderRms: 0.08, captureRms: 0.09);
        Assert.Equal(new[] { 0.15f }, services.Audio.PlaybackGainChanges);
        Assert.Equal(0, services.Audio.ClearQueueCount);

        EmitNearEndFrames(services, count: 7, renderRms: 0.08, captureRms: 0.09);
        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await WaitUntilAsync(() => services.Audio.PlaybackGain == 1f);

        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(new[] { 0.15f, 1f }, services.Audio.PlaybackGainChanges);
        Assert.Single(services.Llm.Prompts);
        Assert.Equal(AssistantState.Listening, assistant.State);
    }

    [Fact]
    public async Task AcousticCandidate_ShutdownRestoresGainWithoutFurtherAudioFrames()
    {
        var services = TestServices.Create();
        using var assistant = await StartDuckingResponseAsync(services);
        EmitNearEndFrames(services, count: 2, renderRms: 0.08, captureRms: 0.09);
        Assert.Equal(0.15f, services.Audio.PlaybackGain);

        await assistant.StopAsync();

        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task AcousticCandidate_DuckingCanBeDisabled()
    {
        var services = TestServices.Create();
        using var assistant = await StartDuckingResponseAsync(services, duckGain: 1f);
        EmitNearEndFrames(services, count: 2, renderRms: 0.08, captureRms: 0.09);
        EmitSilenceFrames(services, count: 15, renderRms: 0.08);

        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Equal(AssistantState.Speaking, assistant.State);
    }

    private static async Task<VoiceAssistant> StartDuckingResponseAsync(TestServices services, float duckGain = 0.15f)
    {
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.PlaybackDuckGain = duckGain;
                options.BargeIn.PlaybackDuckReleaseMs = 200;
                options.BargeIn.MinDurationMs = 180;
            });
        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        return assistant;
    }
}
