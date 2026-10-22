using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Adapters;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task SingleTalk_ConfirmationDisabled_CompletesNormally()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.Equal(0, services.Speaker.FeedCount);
        Assert.Null(services.OwnerSpeaker.Last);
    }

    [Fact]
    public async Task SingleTalk_KwsEnabledButDisabled_KeepsPlainUtterancePath()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant(enableWakeWord: true);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();
        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.NotNull(services.OwnerSpeaker.Last);
        Assert.False(services.OwnerSpeaker.Last!.Policy.Get(SpeakerVerificationPurpose.SingleTalk).Enabled);
    }

    [Fact]
    public async Task SingleTalk_EnabledOwnerMatch_CompletesSemanticTurn()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(services, profilePath);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);
            SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));

            Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Equal(AssistantState.Idle, assistant.State);
            Assert.True(services.Asr.FedAudio.ContainsKey("session-1"));
            Assert.Single(services.Llm.Prompts);
        }
        finally
        {
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_EnabledListeningGate_CarriesPendingSpeakerOnConfirm()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(
            services,
            profilePath,
            options =>
            {
                options.Listening.MinDurationMs = 200;
                options.Listening.MinRms = 0.02f;
            });
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);
            SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));

            Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Equal(AssistantState.Idle, assistant.State);
            Assert.Single(services.Llm.Prompts);
        }
        finally
        {
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_InferenceTimeoutWhileUtteranceIsOpen_DoesNotRejectEarly()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(services, profilePath);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);

            services.Speaker.FeedEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Speaker.FeedBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

            float[] first = FakeSpeakerService.OwnerAudio(9600); // 600ms: first complete speaker window
            float[] second = FakeSpeakerService.OwnerAudio(6400); // keep the VAD utterance open
            services.Vad.PredictAction = () =>
            {
                services.Vad.EmitSpeechStarted();
                services.Vad.EmitSpeaking(first);
            };
            services.Audio.Emit(first);
            await services.Speaker.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

            // The realtime timeout elapses while the speaker is still talking. The next chunk must
            // keep the SingleTalk candidate pending so the endpoint can finish the verification.
            await Task.Delay(350);
            services.Vad.PredictAction = () => services.Vad.EmitSpeaking(second);
            services.Audio.Emit(second);
            await Task.Delay(50);

            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(1000));
            services.Speaker.FeedBlock.TrySetResult();

            Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Single(services.Llm.Prompts);
        }
        finally
        {
            services.Speaker.FeedBlock?.TrySetResult();
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_EnabledMismatch_CompletesWithoutSemanticTurn()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(services, profilePath);
        int userSpoke = 0;
        assistant.UserSpoke += (_, _) => userSpoke++;

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(1);
            SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));
            await WaitUntilAsync(() =>
                assistant.State == AssistantState.Idle && services.Asr.FinishCount >= 1);

            Assert.Equal(0, userSpoke);
            Assert.Empty(services.Llm.Prompts);
            Assert.Equal(AssistantWakeMode.ContinuousConversation, assistant.WakeMode);
        }
        finally
        {
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_EnabledUnlocatableVadAudio_IsUnavailableAndDropsTheCommand()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(services, profilePath);
        int userSpoke = 0;
        assistant.UserSpoke += (_, _) => userSpoke++;

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);

            float[] processed = FakeSpeakerService.OwnerAudio(16000);
            float[] unlocatable = ConstantSamples(0.91f, processed.Length);
            services.Vad.PredictAction = () =>
            {
                services.Vad.EmitSpeechStarted();
                services.Vad.EmitSpeaking(unlocatable);
            };
            services.Audio.Emit(processed);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(1000));
            await WaitUntilAsync(() =>
                assistant.State == AssistantState.Idle && services.Asr.FinishCount >= 1);

            Assert.Equal(0, userSpoke);
            Assert.Empty(services.Llm.Prompts);
        }
        finally
        {
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_Enabled_MapsVadChunkOntoProcessedSourceRange()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        // Build() without the lifecycle-test DTD seam so SingleTalk retains its standalone
        // processed-speech locator instead of the barge-in reservoir.
        using var assistant = VoiceAssistant.CreateBuilder()
            .UseOptions(new VoiceAssistantOptions
            {
                MinSentenceChars = 1,
                PlaybackPrerollMs = 0,
                PlaybackTailBufferMs = 0,
                Listening = { MinDurationMs = 0, MinRms = 0 },
                Wakeup = { Enable = true, PreRollMs = 2500 },
                Speaker = { ProfilePath = profilePath, Confirmation = { Enable = true } },
            })
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .Build();

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);

            float[] utterance = FakeSpeakerService.OwnerAudio(4800);
            services.Audio.Emit(utterance);

            var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
                .GetField("_runtime", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(assistant)!;
            var speakerGate = RuntimeGraphProbe.RequireField<VadSpeakerGateCoordinator>(runtime, "_speakerGate");
            var timeline = RuntimeGraphProbe.RequireField<ISpeechTimelineLocator>(speakerGate, "_timeline");
            Assert.IsType<ProcessedSpeechTimelineLocator>(timeline);
            Assert.True(timeline.TryLocate(
                utterance,
                out long streamEpoch,
                out long startSample,
                out long endSample));
            Assert.Equal(1, streamEpoch);
            Assert.Equal(utterance.Length, endSample - startSample);
        }
        finally
        {
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_AsrFinishesBeforeSpeakerMatch_ParksThenReleasesTheResult()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(services, profilePath);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);
            services.Speaker.FeedBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));
            await WaitUntilAsync(() => services.Asr.FinishCount >= 1);
            Assert.Empty(services.Llm.Prompts);

            services.Speaker.FeedBlock.TrySetResult();
            Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Single(services.Llm.Prompts);
        }
        finally
        {
            services.Speaker.FeedBlock?.TrySetResult();
            TryDelete(profilePath);
        }
    }

    [Fact]
    public async Task SingleTalk_PendingTurn_DoesNotEndContinuousConversationOnIdleTimeout()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        using var assistant = CreateSingleTalkAssistant(
            services,
            profilePath,
            configure: options => options.Speaker.Confirmation.InferenceTimeoutMs = 2_000,
            continuousConversationIdleTimeout: TimeSpan.FromMilliseconds(40));
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        try
        {
            await assistant.StartAsync();
            await EnterOwnerConversationAsync(services, assistant);
            services.Speaker.FeedBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));
            await WaitUntilAsync(() =>
                assistant.State is AssistantState.Listening or AssistantState.Recognizing);

            await PumpSilentFramesUntilAsync(services, static () => false, maxFrames: 12);
            Assert.Equal(AssistantWakeMode.ContinuousConversation, assistant.WakeMode);
            Assert.True(
                assistant.State is AssistantState.Listening or AssistantState.Recognizing,
                $"Expected Listening or Recognizing, got {assistant.State}.");

            services.Speaker.FeedBlock.TrySetResult();
            Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        }
        finally
        {
            services.Speaker.FeedBlock?.TrySetResult();
            TryDelete(profilePath);
        }
    }

    private static VoiceAssistant CreateSingleTalkAssistant(
        TestServices services,
        string profilePath,
        Action<VoiceAssistantOptions>? configure = null,
        TimeSpan? continuousConversationIdleTimeout = null)
        => services.CreateAssistant(
            enableWakeWord: true,
            useRealSpeaker: true,
            continuousConversationIdleTimeout: continuousConversationIdleTimeout,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.Speaker.Confirmation.Enable = true;
                configure?.Invoke(options);
            });

    private static async Task EnterOwnerConversationAsync(TestServices services, VoiceAssistant assistant)
    {
        services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(0);
        services.Audio.Emit(FakeSpeakerService.OwnerAudio(16000));
        services.Wakeup.EmitWakeup(startTimeMs: 200, durationMs: 600);
        for (int i = 0; i < 100 && assistant.WakeMode != AssistantWakeMode.ContinuousConversation; i++)
        {
            services.Audio.Emit(FakeSpeakerService.OwnerAudio(1600));
            await Task.Delay(10);
        }

        Assert.Equal(AssistantWakeMode.ContinuousConversation, assistant.WakeMode);
    }

    private static void SpeakLocatedUtterance(TestServices services, float[] utterance)
    {
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(utterance);
        };
        services.Audio.Emit(utterance);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(utterance.Length * 1000.0 / 16000.0));
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); } catch { }
    }
}
