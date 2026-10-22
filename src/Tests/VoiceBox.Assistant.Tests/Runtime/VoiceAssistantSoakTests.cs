using System.Collections;
using System.Reflection;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task StartupCancellation_AfterAudioStartsRollsBackRunningModules()
    {
        var services = TestServices.Create();
        using var cancellation = new CancellationTokenSource();
        services.Audio.OnStarted = cancellation.Cancel;
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => assistant.StartAsync(cancellation.Token));

        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.False(services.Audio.IsRecording);
        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(services.Wakeup.StartCount, services.Wakeup.StopCount);
        Assert.False(services.OwnerSpeaker.Last!.Started);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SplitAudio_FailedBuildDoesNotRetainFrontendSubscriptions(bool validationFailure)
    {
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        for (int i = 0; i < 100; i++)
        {
            var builder = VoiceAssistant.CreateBuilder().UseAudio(record, playback);
            if (validationFailure)
            {
                var services = TestServices.Create();
                builder.UseOptions(new() { MaxHistoryTurns = 0 })
                    .UseTurnBoundary(services.Vad)
                    .UseSpeechRecognition(services.Asr)
                    .UseResponseGeneration(services.Llm)
                    .UseSpeechSynthesis(services.Tts);
                Assert.Throws<ArgumentException>(() => builder.Build());
            }
            else
            {
                Assert.Throws<AssistantCompositionException>(() => builder.Build());
            }
        }

        var subscription = (Delegate?)typeof(FakeDuplexService)
            .GetField("OnDataAvailable", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(record);
        Assert.Null(subscription);
        Assert.Equal(0, record.StartCount);
        Assert.Equal(0, playback.StartCount);
    }

    [Fact]
    public void SplitAudio_SteadyStateCaptureDoesNotAllocatePerFrame()
    {
        var record = new FakeDuplexService();
        using var frontend = new SplitAssistantAudioFrontend(record, new FakeDuplexService());
        int samples = 0;
        frontend.OnProcessedCaptureFrame += (in ProcessedCaptureFrame frame) => samples += frame.SampleCount;
        var audio = new float[160];
        for (int i = 0; i < 1_000; i++)
            record.EmitRecordData(audio);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
            record.EmitRecordData(audio);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(11_000 * 160, samples);
        Assert.InRange(allocated, 0, 1_024);
    }

    [Fact]
    public void SplitAudio_ReentrantCaptureKeepsEachBorrowedFrameIntact()
    {
        var record = new FakeDuplexService();
        using var frontend = new SplitAssistantAudioFrontend(record, new FakeDuplexService());
        int callbacks = 0;
        frontend.OnProcessedCaptureFrame += (in ProcessedCaptureFrame frame) =>
        {
            callbacks++;
            if (callbacks == 1)
            {
                Assert.Equal(new[] { 0.1f, 0.2f }, frame.Audio.ToArray());
                record.EmitRecordData([0.7f]);
                Assert.Equal(new[] { 0.1f, 0.2f }, frame.Audio.ToArray());
            }
            else
            {
                Assert.Equal(new[] { 0.7f }, frame.Audio.ToArray());
                Assert.Equal(2, frame.SourceRange.StartSample);
            }
        };
        record.EmitRecordData([0.1f, 0.2f]);
        Assert.Equal(2, callbacks);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CompletedTurns_LateVadAudioDoesNotAccumulateCaptureRecords(bool lateEndpoint)
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        var runtime = RuntimeGraphProbe.RequireField<AssistantRuntime>(assistant, "_runtime");
        int replies = 0;
        assistant.AssistantReplied += (_, _) => Interlocked.Increment(ref replies);
        await assistant.StartAsync();

        for (int round = 1; round <= 30; round++)
        {
            services.Asr.CompleteStart($"soak-{round}");
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.1f, 0.2f]);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
            await WaitUntilAsync(() => Volatile.Read(ref replies) == round &&
                runtime.ActiveTurnContextCount == 0);
            Assert.Equal(AssistantState.Idle, assistant.State);

            // A delayed VAD callback has no active utterance after the completed turn.
            if (lateEndpoint)
                services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
            else
                services.Vad.EmitSpeaking([0.3f]);
            await assistant.EnqueueSpeechAsync(string.Empty);
        }

        await assistant.StopAsync();
        var capture = (IDictionary)RuntimeGraphProbe.RequireField(assistant, "_captureByTurn");
        Assert.Empty(capture);
        Assert.Equal(30, services.Asr.FinishCount);
        Assert.Equal(30, services.Llm.Prompts.Count);
    }
}
