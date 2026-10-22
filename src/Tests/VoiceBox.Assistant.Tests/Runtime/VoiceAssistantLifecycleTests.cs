using Microsoft.Extensions.Logging.Abstractions;
using System.Reflection;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.TurnContext;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.SpeakerService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;
using Xunit;

#pragma warning disable CS0067

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{

    [Fact]
    public async Task OwnerSpeakerGate_RejectsOtherSpeakerThenAllowsOwnerWakeup()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        // Real speaker module on the identity tap: the KWS-enabled composition is built from the
        // owner-speaker-profile-v2 written for this frontend fingerprint.
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options => options.Speaker.ProfilePath = profilePath);
        int acceptedWakeups = 0;
        assistant.WakeupDetected += (_, _) => acceptedWakeups++;

        try
        {
            await assistant.StartAsync();
            // Impostor: an embedding orthogonal to the owner axis scores ~0 on every window.
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(1);
            services.Audio.Emit(FakeSpeakerService.OwnerAudio(16000));
            int startsBeforeMismatch = services.Wakeup.StartCount;
            services.Wakeup.EmitWakeup(startTimeMs: 200, durationMs: 600);
            for (int i = 0; i < 100 && services.Wakeup.StartCount == startsBeforeMismatch; i++)
            {
                services.Audio.Emit(FakeSpeakerService.OwnerAudio(1600));
                await Task.Delay(10);
            }
            Assert.True(services.Wakeup.StartCount > startsBeforeMismatch);

            Assert.Equal(AssistantWakeMode.WaitingForWakeWord, assistant.WakeMode);
            Assert.Equal(0, acceptedWakeups);
            Assert.Equal(1, services.Wakeup.StartCount - startsBeforeMismatch);

            // Owner: the same keyword geometry with the owner embedding authorizes on the first
            // scored window (high confidence), without waiting for a 2-of-3 majority.
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(0);
            int feedsBeforeOwner = services.Speaker.FeedCount;
            services.Audio.Emit(FakeSpeakerService.OwnerAudio(16000));
            services.Wakeup.EmitWakeup(startTimeMs: 200, durationMs: 600);
            for (int i = 0; i < 100 && assistant.WakeMode != AssistantWakeMode.ContinuousConversation; i++)
            {
                services.Audio.Emit(FakeSpeakerService.OwnerAudio(1600));
                await Task.Delay(10);
            }
            Assert.Equal(AssistantWakeMode.ContinuousConversation, assistant.WakeMode);

            Assert.Equal(1, acceptedWakeups);
            Assert.True(services.Speaker.FeedCount > feedsBeforeOwner);
            await assistant.StopAsync();
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    [Fact]
    public async Task OwnerSpeakerGate_SilentKeywordCandidate_IsDeniedForInsufficientEvidence()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options => options.Speaker.ProfilePath = profilePath);
        int acceptedWakeups = 0;
        assistant.WakeupDetected += (_, _) => acceptedWakeups++;

        try
        {
            await assistant.StartAsync();
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(0);
            // Silence carries no identity evidence: even the owner embedding cannot be produced,
            // and the keyword alone never opens the session.
            services.Audio.Emit(new float[16000]);
            int startsBefore = services.Wakeup.StartCount;
            services.Wakeup.EmitWakeup(startTimeMs: 200, durationMs: 600);
            for (int i = 0; i < 100 && services.Wakeup.StartCount == startsBefore; i++)
            {
                services.Audio.Emit(new float[1600]);
                await Task.Delay(10);
            }

            Assert.True(services.Wakeup.StartCount > startsBefore);
            Assert.Equal(AssistantWakeMode.WaitingForWakeWord, assistant.WakeMode);
            Assert.Equal(0, acceptedWakeups);
            Assert.Equal(0, services.Speaker.FeedCount);
            await assistant.StopAsync();
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    [Fact]
    public async Task SpeechEndBeforeAsrSessionReady_CompletesTurnAndReturnsIdle()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        Assert.Equal(AssistantState.Listening, assistant.State);

        services.Asr.CompleteStart("session-1");

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.Equal(1, services.Asr.FinishCount);
        Assert.Single(services.Asr.FedAudio["session-1"]);
        var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        await WaitUntilAsync(() => runtime.ActiveTurnContextCount == 0);
    }

    [Fact]
    public async Task DisposeWithoutExplicitStop_StopsAudioServicesAndDetachesVadEvents()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();

        await assistant.StartAsync();
        assistant.Dispose();

        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);

        services.Vad.EmitSpeechStarted();

        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task DisposeAfterExplicitStop_DoesNotStopAudioTwice()
    {
        var services = TestServices.Create();
        VoiceAssistant assistant = services.CreateAssistant();

        await assistant.StartAsync();
        await assistant.StopAsync();
        assistant.Dispose();

        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task ControlPlane_RepeatedTurnStartAndDispose_DoesNotLeavePendingWork()
    {
        for (int iteration = 0; iteration < 10; iteration++)
        {
            var services = TestServices.Create();
            services.Asr.CompleteStart("session-1");
            var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Tts.BlockAfterFirstChunk = firstTtsBlock;
            using var assistant = services.CreateAssistant(enableBargeIn: true);

            await assistant.StartAsync();
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

            try
            {
                await WaitUntilAsync(() =>
                    assistant.State == AssistantState.Speaking &&
                    services.Audio.AcceptedSamples > 0);
            }
            catch (OperationCanceledException exception)
            {
                throw new Xunit.Sdk.XunitException(
                    $"iteration={iteration}; state={assistant.State}; acceptedSamples={services.Audio.AcceptedSamples}; " +
                    $"asrFinishCount={services.Asr.FinishCount}; {exception.Message}");
            }

            firstTtsBlock.TrySetResult();
            await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        }
    }

    [Fact]
    public async Task ConcurrentStopAsync_StopsAudioOnlyOnce()
    {
        var services = TestServices.Create();
        services.Audio.CompleteStopAsynchronously = true;
        using var assistant = services.CreateAssistant();
        await assistant.StartAsync();

        Task firstStop = assistant.StopAsync();
        Task secondStop = assistant.StopAsync();
        await Task.WhenAll(firstStop, secondStop).WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task StopAsync_ConcurrentWithDispose_WaitsForPublishedStopTask()
    {
        var services = TestServices.Create();
        services.Audio.StopEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StopBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        VoiceAssistant assistant = services.CreateAssistant();
        await assistant.StartAsync();

        object runtime = typeof(VoiceAssistant)
            .GetField("_runtime", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        object lifecycleLock = RuntimeGraphProbe.RequireField(runtime, "_lifecycleLock");
        object runScope = RuntimeGraphProbe.RequireField(runtime, "_runScope");
        var disposedField = runScope.GetType()
            .GetField("_disposed", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;

        Task disposeTask;
        Task stopTask;
        Monitor.Enter(lifecycleLock);
        try
        {
            using var disposeStarted = new ManualResetEventSlim();
            disposeTask = Task.Run(() =>
            {
                disposeStarted.Set();
                assistant.Dispose();
            });
            Assert.True(disposeStarted.Wait(TimeSpan.FromSeconds(2)));

            // The previous implementation set _disposed before waiting for this lock. Give it a
            // deterministic opportunity to enter that state before StopAsync re-enters the lock.
            SpinWait.SpinUntil(
                () => (int)disposedField.GetValue(runScope)! != 0,
                TimeSpan.FromMilliseconds(200));
            stopTask = assistant.StopAsync();
        }
        finally
        {
            Monitor.Exit(lifecycleLock);
        }

        await services.Audio.StopEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.False(stopTask.IsCompleted);

        services.Audio.StopBlock.TrySetResult();
        await Task.WhenAll(stopTask, disposeTask).WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task StopDuringStart_DoesNotReturnToIdleOrStartWakeupService()
    {
        var services = TestServices.Create();
        services.Audio.StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        Task startTask = assistant.StartAsync();
        await services.Audio.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Task stopTask = assistant.StopAsync();
        await stopTask.WaitAsync(TimeSpan.FromSeconds(2));
        services.Audio.StartBlock.TrySetResult();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => startTask.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.False(services.Audio.IsRecording);
        // Speaker and wakeup gates start before the audio frontend so no identity frame is lost;
        // an aborted frontend start therefore rolls them back instead of leaving KWS listening.
        Assert.Equal(1, services.Wakeup.StartCount);
        Assert.Equal(services.Wakeup.StartCount, services.Wakeup.StopCount);
        Assert.False(services.Wakeup.IsListening);
        Assert.False(services.OwnerSpeaker.Last!.Started);
    }

    [Fact]
    public async Task Dispose_DoesNotDependOnCurrentSynchronizationContext()
    {
        var services = TestServices.Create();
        services.Audio.CompleteStopAsynchronously = true;

        Task disposeTask = Task.Run(() =>
        {
            SynchronizationContext? previous = SynchronizationContext.Current;
            SynchronizationContext.SetSynchronizationContext(new NonPumpingSynchronizationContext());
            try
            {
                VoiceAssistant assistant = services.CreateAssistant();
                assistant.StartAsync().GetAwaiter().GetResult();
                assistant.Dispose();
            }
            finally
            {
                SynchronizationContext.SetSynchronizationContext(previous);
            }
        });

        await disposeTask.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(1, services.Audio.StopCount);
    }

    [Fact]
    public async Task PublicEvents_UseVoiceAssistantFacadeAsSender()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant();
        var stateSender = new TaskCompletionSource<object?>(TaskCreationOptions.RunContinuationsAsynchronously);
        var userSender = new TaskCompletionSource<object?>(TaskCreationOptions.RunContinuationsAsynchronously);
        var textSender = new TaskCompletionSource<object?>(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.StateChanged += (sender, _) =>
        {
            if (!stateSender.Task.IsCompleted)
                stateSender.TrySetResult(sender);
        };
        assistant.UserSpoke += (sender, _) => userSender.TrySetResult(sender);
        assistant.AssistantText += (sender, e) =>
        {
            if (!e.IsThinking)
                textSender.TrySetResult(sender);
        };

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        Assert.Same(assistant, await stateSender.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Same(assistant, await userSender.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Same(assistant, await textSender.Task.WaitAsync(TimeSpan.FromSeconds(2)));
    }

    [Fact]
    public async Task EventSubscriberException_DoesNotPreventLaterSubscribersFromReceivingEvent()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        var laterSubscriberCalled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.StateChanged += (_, _) => throw new InvalidOperationException("subscriber failed");
        assistant.StateChanged += (_, _) => laterSubscriberCalled.TrySetResult();

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();

        await laterSubscriberCalled.Task.WaitAsync(TimeSpan.FromSeconds(2));
    }

    [Fact]
    public async Task StopWhileListeningWithActiveAsrSession_CleansAsrSession()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant();

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        await WaitUntilAsync(() => services.Asr.FedAudio.ContainsKey("session-1"));

        await assistant.StopAsync();

        Assert.Equal(1, services.Asr.FinishCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task PlaybackQueueBackpressure_RetriesInsteadOfDroppingAcceptedAudio()
    {
        var services = TestServices.Create();
        services.Audio.FailuresBeforeSuccess = 2;
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.True(services.Audio.EnqueueCalls >= 3);
        Assert.Equal(services.Tts.EmittedSamples, services.Audio.AcceptedSamples);
    }

    [Fact]
    public async Task TtsSpeedOption_IsPassedToStreamTts()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant(ttsSpeed: 1.25f);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.NotNull(services.Tts.LastStreamOptions);
        Assert.Equal(1.25f, services.Tts.LastStreamOptions!.Speed);
    }

    [Fact]
    public async Task EnableThinking_RaisesThinkingTextButDoesNotSendItToTts()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Llm.EmitThinkingWhenEnabled = true;
        using var assistant = services.CreateAssistant(enableThinking: true);
        var textEvents = new List<(bool IsThinking, string Delta)>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.AssistantText += (_, e) =>
        {
            lock (textEvents)
                textEvents.Add((e.IsThinking, e.Delta));
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Contains(textEvents, e => e.IsThinking && e.Delta == "先分析一下。");
        Assert.Contains(textEvents, e => !e.IsThinking && e.Delta == "好的。");
        Assert.Single(services.Tts.SynthesizedTexts);
        Assert.Equal("好的。", services.Tts.SynthesizedTexts[0]);
    }

    [Fact]
    public async Task BargeInDuringThinking_CompletesWithoutRetainingDecisionContext()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        var thinkingBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Llm.EmitThinkingWhenEnabled = true;
        services.Llm.BlockAfterThinkingChunk = thinkingBlock;
        using VoiceAssistant assistant = services.CreateAssistant(enableBargeIn: true, enableThinking: true);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Thinking);

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking(BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);

        thinkingBlock.SetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
    }

    [Fact]
    public async Task WakeWordEnabled_BeforeWakeup_FeedsWakeServiceOnly()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f, 0.2f, 0.3f]);

        Assert.Single(services.Wakeup.FedAudio);
        Assert.Equal([0.1f, 0.2f, 0.3f], services.Wakeup.FedAudio[0]);
        Assert.Empty(services.Vad.PredictedAudio);
    }

    [Fact]
    public async Task WakeWordEnabled_WakeupReplaysPreRollIntoVad()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            wakeupPreRollMs: 1000,
            wakeupReplayLeadInMs: 0);

        await assistant.StartAsync();

        services.Audio.Emit([1f, 2f, 3f, 4f]);
        services.Wakeup.EmitWakeup(startTimeMs: 0, durationMs: 0);
        services.Audio.Emit([5f, 6f]);

        Assert.Equal(1, services.Vad.ResetCount);
        Assert.Single(services.Vad.PredictedAudio);
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], services.Vad.PredictedAudio[0]);
        Assert.Equal(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task WakeWordEnabled_ReplayStartsAtWakeWordStartWhenTimingIsAvailable()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            wakeupPreRollMs: 1000,
            wakeupReplayLeadInMs: 0);

        await assistant.StartAsync();

        services.Audio.Emit(Enumerable.Range(0, 3200).Select(i => (float)i).ToArray());
        services.Wakeup.EmitWakeup(startTimeMs: 0, durationMs: 100);
        services.Audio.Emit([99f]);

        Assert.Single(services.Vad.PredictedAudio);
        Assert.Equal(3201, services.Vad.PredictedAudio[0].Length);
        Assert.Equal(0f, services.Vad.PredictedAudio[0][0]);
    }

    [Fact]
    public async Task WakeWordEnabled_AfterWakeup_ContinuousConversationFeedsWakeServiceSideChannel()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Audio.Emit([0.3f]);

        Assert.Equal(2, services.Wakeup.FedAudio.Count);
        Assert.Equal([0.3f], services.Wakeup.FedAudio[1]);
        Assert.Equal(2, services.Vad.PredictedAudio.Count);
        Assert.Equal([0.3f], services.Vad.PredictedAudio[1]);
    }

    [Fact]
    public async Task WakeWordEnabled_WakeupThenSpeech_CompletesNormalTurn()
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

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-1"));
    }

    [Fact]
    public async Task WakeWordEnabled_WhenContinuousConversationDisabled_ReturnsToWakeWordAfterTurn()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            configure: options => options.Conversation.ContinuousConversationEnabled = false);
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
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, assistant.WakeMode);

        int vadPredictionCount = services.Vad.PredictedAudio.Count;
        services.Audio.Emit([0.9f]);

        Assert.Equal(vadPredictionCount, services.Vad.PredictedAudio.Count);
        Assert.Equal([0.9f], services.Wakeup.FedAudio[^1]);
    }

    [Fact]
    public async Task WakeWordEnabled_EndConversationKeyword_ReturnsToWakeWordWithoutCallingLlm()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-1", "好的，再 见！");
        using var assistant = services.CreateAssistant(enableWakeWord: true);

        int userSpokeCount = 0;
        assistant.UserSpoke += (_, _) => userSpokeCount++;

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));

        await WaitUntilAsync(() =>
            assistant.WakeMode == AssistantWakeMode.WaitingForWakeWord &&
            services.Asr.FinishCount == 1);

        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.Equal(1, userSpokeCount);
        Assert.Empty(services.Llm.Prompts);
        Assert.Empty(services.Tts.SynthesizedTexts);
    }

    [Fact]
    public async Task EndConversationKeyword_IsIgnoredWithoutWakeWordGate()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-1", "再见");
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Single(services.Llm.Prompts);
        Assert.Equal(AssistantWakeMode.Disabled, assistant.WakeMode);
    }

    [Fact]
    public async Task WakeWordEnabled_BargeInDisabled_WakeWordInterruptsSpeakingResponse()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "再回答一个问题");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            enableBargeIn: false,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Tts.BlockAfterFirstChunk = null;
        EmitCorrelatedPlaybackInput(services, [0.7f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.8f]);

        await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);
        Assert.Equal(AssistantState.Idle, assistant.State);
        // ClearQueue is an earlier Kernel effect; VAD replay follows the completed commit.
        await WaitUntilAsync(() => services.Vad.PredictedAudio.Any(chunk => chunk.SequenceEqual([0.7f, 0.8f])));
        Assert.Contains(services.Vad.PredictedAudio, chunk => chunk.SequenceEqual([0.7f, 0.8f]));
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.7f, 0.8f]);
        services.Vad.EmitSpeaking([0.9f, 1.0f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(300));

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Contains(services.Asr.FedAudio["session-2"], chunk => chunk.SequenceEqual([0.7f, 0.8f]));
    }

    [Fact]
    public async Task WakeWordEnabled_BargeInEnabled_WakeWordStillInterruptsSpeakingResponse()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "再回答一个问题");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Tts.BlockAfterFirstChunk = null;
        services.Audio.ClearQueueEntered = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.ClearQueueBlock = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        EmitCorrelatedPlaybackInput(services, [0.7f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.8f]);

        try
        {
            // The Kernel issues the owner-scoped StopPlayback/ClearPlaybackQueue effects while it
            // commits the wake-word interruption; the module-side replay into the VAD follows the
            // commit, so at this boundary the reply is stopped but not yet replayed.
            await services.Audio.ClearQueueEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.True(services.Audio.ClearQueueCount >= 1);
            Assert.Equal(AssistantState.Idle, assistant.State);
            Assert.DoesNotContain(services.Vad.PredictedAudio, chunk => chunk.SequenceEqual([0.7f, 0.8f]));

            // Playback clear is the externally visible wakeup-cancellation boundary. Submit the
            // fresh turn while ClearQueue is still on the control-loop stack; handler-tail cleanup
            // must not erase these newly admitted facts.
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.7f, 0.8f]);
            services.Vad.EmitSpeaking([0.9f, 1.0f]);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(300));
        }
        finally
        {
            services.Audio.ClearQueueBlock.TrySetResult();
        }

        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.Contains(services.Vad.PredictedAudio, chunk => chunk.SequenceEqual([0.7f, 0.8f]));
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Contains(services.Asr.FedAudio["session-2"], chunk => chunk.SequenceEqual([0.7f, 0.8f]));
        Assert.Equal(1, services.Audio.ClearQueueCount);
    }

    [Fact]
    public async Task WakeWordEnabled_BargeInEnabled_WakeWordOnlyStartsNewAsrTurn()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", string.Empty);
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Tts.BlockAfterFirstChunk = null;
        EmitCorrelatedPlaybackInput(services, [0.7f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.8f]);

        await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);
        Assert.Equal(AssistantState.Idle, assistant.State);
        await WaitUntilAsync(() => services.Vad.PredictedAudio.Any(chunk => chunk.SequenceEqual([0.7f, 0.8f])));
        Assert.Contains(services.Vad.PredictedAudio, chunk => chunk.SequenceEqual([0.7f, 0.8f]));

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.7f, 0.8f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        Assert.Single(services.Llm.Prompts);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Contains(services.Asr.FedAudio["session-2"], chunk => chunk.SequenceEqual([0.7f, 0.8f]));
    }

    [Fact]
    public async Task WakeWordEnabled_WakeWordInterruptsBargeInValidationAndStartsFreshTurn()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.CompleteStart("session-3");
        services.Asr.SetResult("session-3", string.Empty);
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableWakeWord: true,
            enableBargeIn: true,
            configure: options =>
            {
                // This case is the keyword interrupting in-flight barge-in ASR, not owner
                // confirmation: leaving Enable on lets AlwaysOwner match the barge-in and
                // already be Listening before the second wakeup arrives.
                options.Speaker.Confirmation.Enable = false;
                options.BargeIn.PlaybackStartupGuardMs = 0;
            });

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.4f, 0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(200));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        await WaitUntilAsync(() => services.Asr.FedAudio.ContainsKey("session-2"));

        EmitCorrelatedPlaybackInput(services, [0.7f]);
        services.Wakeup.EmitWakeup(startTimeMs: 500, durationMs: 0);
        services.Audio.Emit([0.8f]);

        await WaitUntilAsync(() => services.Audio.ClearQueueCount >= 1);
        Assert.True(services.Audio.ClearQueueCount >= 1);
        Assert.Equal(AssistantState.Idle, assistant.State);
        await WaitUntilAsync(() => services.Vad.PredictedAudio.Any(chunk => chunk.SequenceEqual([0.7f, 0.8f])));
        Assert.Contains(services.Vad.PredictedAudio, chunk => chunk.SequenceEqual([0.7f, 0.8f]));

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.7f, 0.8f]);
        services.Vad.EmitSpeaking([0.9f, 1.0f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        await WaitUntilAsync(() => services.Asr.FinishCount >= 3);
        await WaitUntilAsync(() => services.Asr.FedAudio.ContainsKey("session-3"));
        Assert.True(services.Asr.FedAudio.ContainsKey("session-3"));
        Assert.Contains(services.Asr.FedAudio["session-3"], chunk => chunk.SequenceEqual([0.7f, 0.8f]));
    }

    [Fact]
    public async Task WakeWordEnabled_RaisesWakeupEventWithoutSynthesizingSpeech()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(enableWakeWord: true);
        var detected = new TaskCompletionSource<WakeupEventArgs>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.WakeupDetected += (_, e) => detected.TrySetResult(e.Wakeup);

        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);

        WakeupEventArgs wakeup = await detected.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("小爱同学", wakeup.Keyword);
        Assert.Empty(services.Tts.SynthesizedTexts);
        Assert.Equal(0, services.Audio.AcceptedSamples);
    }

    [Fact]
    public async Task ListeningGate_LowRmsSpeech_DoesNotStartTurnOrAsrSession()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(configure: options =>
        {
            options.Listening.MinDurationMs = 200;
            options.Listening.MinRms = 0.02f;
        });
        var stateChanges = new List<AssistantState>();
        assistant.StateChanged += (_, e) => { lock (stateChanges) stateChanges.Add(e.To); };

        await assistant.StartAsync();

        // 模拟 TTS 回声残差: 时长够 (500ms@16k) 但 rms=0.014 低于门槛。
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking(ConstantSamples(0.014f, 8000));
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        await Task.Delay(50);

        Assert.Equal(AssistantState.Idle, assistant.State);
        lock (stateChanges)
            Assert.DoesNotContain(AssistantState.Listening, stateChanges);
        Assert.Empty(services.Asr.FedAudio);
        Assert.Equal(0, services.Asr.FinishCount);
    }

    [Fact]
    public async Task ListeningGate_RealSpeech_CompletesTurnWithCandidateAudio()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant(configure: options =>
        {
            options.Listening.MinDurationMs = 200;
            options.Listening.MinRms = 0.02f;
        });
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        // 正常人声: 600ms@16k, rms=0.1 → 候选确认 → 正常完成一轮。
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking(ConstantSamples(0.1f, 9600));
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(600));

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(AssistantState.Idle, assistant.State);
        // 候选期音频不丢: 全部补喂给了 ASR。
        Assert.True(services.Asr.FedAudio.ContainsKey("session-1"));
        Assert.Equal(9600, services.Asr.FedAudio["session-1"].Sum(c => c.Length));
    }

    [Fact]
    public async Task BargeInDuringSpeaking_WithValidAsr_CancelsCurrentResponseAndStartsNewAsrSession()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-2", "停一下");
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var validationEvents = new List<BargeInValidationKind>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Tts.BlockAfterFirstChunk = null;
        services.Asr.CompleteStart("session-2");
        Assert.True(services.Audio.IsPlaying);
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        Assert.True(services.PlaybackDetector.PushCount > 0);
        Assert.True(services.PlaybackDetector.LastDecisionEnabled);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        // KWS disabled: the acoustic confirmation is the whole authorization. Playback stops at
        // once (Heard then Accepted), before any ASR text exists, and is never ducked.
        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Contains(BargeInValidationKind.Accepted);
        });
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(AssistantState.Listening, assistant.State);
        Assert.Single(services.Llm.Prompts);

        validationFinish.SetResult();
        // Accepted is the ordered Kernel publication barrier. The validated RecognitionCompleted
        // signal and its background GenerateResponse effect follow it, so wait on the provider
        // observation rather than assuming both independent consumers have run back-to-back.
        await WaitUntilAsync(() => services.Llm.Prompts.Count == 2);
        Assert.Equal(2, services.Llm.Prompts.Count);
        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.True(services.Asr.FinishCount >= 2);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Contains(
            services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk),
            sample => Math.Abs(sample - 0.4f) < 0.0001f);
        lock (validationEvents)
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents);
        Assert.Equal(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task PlaybackBargeIn_WithEmptyAsr_KeepsPlaybackStoppedWithoutStartingNewTurn()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-2", "");
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var validationEvents = new List<BargeInValidationKind>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
        };
        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Contains(BargeInValidationKind.Accepted);
        });
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Equal(AssistantState.Listening, assistant.State);

        // The empty whole-utterance result decides only that no response follows. The reply that
        // was legitimately stopped is never resumed.
        validationFinish.SetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Equal(2, services.Asr.FinishCount);
        Assert.Single(services.Llm.Prompts);
        lock (validationEvents)
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents);
        Assert.DoesNotContain(BargeInValidationKind.Rejected, validationEvents);
    }

    [Fact]
    public async Task PlaybackBargeIn_WithOwnTtsAsrText_KeepsPlaybackStoppedWithoutStartingNewTurn()
    {
        var services = TestServices.Create();
        services.Llm.ResponseText =
            "在一座被山围着的小镇上有个叫阿明的少年。后来他走出了小镇。";
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-2", "在一座被山围着的小镇上");
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var accepted = new TaskCompletionSource<BargeInValidationEventArgs>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var validationEvents = new List<BargeInValidationKind>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult(e);
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        BargeInValidationEventArgs acceptedEvent = await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("audio-confirmed", acceptedEvent.Reason);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Single(services.Llm.Prompts);

        // Echo of our own TTS text: no semantic turn, but the physical stop already happened and
        // is not undone.
        validationFinish.SetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        Assert.Single(services.Llm.Prompts);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        lock (validationEvents)
            Assert.DoesNotContain(BargeInValidationKind.Rejected, validationEvents);
    }

    [Fact]
    public async Task PlaybackBargeIn_AsrStartFailure_KeepsPlaybackStoppedAndReturnsToIdle()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var validationEvents = new List<BargeInValidationKind>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);

        // The stop was issued by the acoustic interrupt before ASR was involved; a failed session
        // start cannot bring the reply back, it only ends the new turn without a response.
        services.Asr.FailStart(new InvalidOperationException("start failed"));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Single(services.Llm.Prompts);
        lock (validationEvents)
            Assert.DoesNotContain(BargeInValidationKind.Rejected, validationEvents);
    }

    [Fact]
    public async Task PlaybackBargeIn_FillerTextAfterOwnerStop_ReturnsToWakeWordWithoutResponse()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "嗯");
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        // KWS enabled: the always-owner speaker module authorizes the candidate; the stop is the
        // speaker Match, not the acoustic confirmation.
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            continuousConversationIdleTimeout: TimeSpan.Zero,
            configure: options =>
            {
                options.Conversation.ContinuousConversationEnabled = false;
                options.BargeIn.PlaybackStartupGuardMs = 0;
            });
        var accepted = new TaskCompletionSource<BargeInValidationEventArgs>(TaskCreationOptions.RunContinuationsAsynchronously);
        var validationEvents = new List<BargeInValidationKind>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult(e);
        };

        await assistant.StartAsync();
        services.Audio.Emit(FakeSpeakerService.OwnerAudio(8000));
        services.Wakeup.EmitWakeup();
        services.Audio.Emit(FakeSpeakerService.OwnerAudio(1600));
        await WaitUntilAsync(() => assistant.WakeMode == AssistantWakeMode.ContinuousConversation);

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        BargeInValidationEventArgs acceptedEvent = await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("speaker-match", acceptedEvent.Reason);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Equal(AssistantState.Listening, assistant.State);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, assistant.WakeMode);

        // A filler utterance ends the new turn without a response; with continuous conversation
        // disabled the assistant then returns to the wake word.
        validationFinish.SetResult();
        await WaitUntilAsync(() => assistant.WakeMode == AssistantWakeMode.WaitingForWakeWord);

        Assert.Equal(AssistantState.Idle, assistant.State);
        Assert.Single(services.Llm.Prompts);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        lock (validationEvents)
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents);
    }

    [Fact]
    public async Task PlaybackBargeIn_AcousticInterrupt_StopsBeforeAsrAndUsesDelayedValidResult()
    {
        var services = TestServices.Create();
        var timeProvider = new ManualTimerTimeProvider();
        services.Asr.CompleteStart("session-1");
        var asrFinish = services.Asr.BlockFinish("session-2");
        services.Asr.SetResult("session-2", "停一下");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0,
            timeProvider: timeProvider);
        var accepted = new TaskCompletionSource<BargeInValidationEventArgs>(TaskCreationOptions.RunContinuationsAsynchronously);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult(e);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);
        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Tts.BlockAfterFirstChunk = null;
        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        // No timer is involved: the stop is decided by the acoustic confirmation itself, and the
        // still-outstanding ASR result cannot delay it.
        BargeInValidationEventArgs acceptedEvent =
            await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal("audio-confirmed", acceptedEvent.Reason);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(AssistantState.Listening, assistant.State);
        Assert.False(asrFinish.Task.IsCompleted);

        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        asrFinish.SetResult();
        await WaitUntilAsync(() => services.Tts.EmittedSamples >= 8);
        for (int i = 0; i < 8 && !replied.Task.IsCompleted; i++)
        {
            timeProvider.Advance(TimeSpan.FromMilliseconds(50));
            await Task.Yield();
        }
        Assert.Equal("好的。", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(2, services.Llm.Prompts.Count);
        Assert.Contains(
            services.Llm.Prompts[^1],
            message => message.Role == ChatRole.User && MessageText(message) == "停一下");
        Assert.Equal(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task PlaybackResidualDuringSpeaking_DoesNotInterruptCurrentResponse()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.MinRms = 0.1f;
            });

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, PlaybackResidualSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(1000));
        await Task.Delay(50);

        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Equal(AssistantState.Speaking, assistant.State);
    }

    [Fact]
    public async Task PlaybackStartupGuard_SuppressesVadButRetainsRecognitionPrefix()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 120;
                options.BargeIn.PlaybackCandidateSuppressionMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 200;
            });

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        await Task.Delay(20);

        float[] guardedPrefix = ConstantSamples(0.37f, 960);
        services.Audio.Emit(guardedPrefix);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking(guardedPrefix);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(60));
        await Task.Delay(20);

        Assert.Empty(services.Vad.PredictedAudio);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Equal(AssistantState.Speaking, assistant.State);
        Assert.False(services.PlaybackDetector.LastDecisionEnabled);
        Assert.True(services.PlaybackDetector.LastEchoLearningEnabled);

        var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        // With barge-in installed the near-end reservoir is the VAD gate's speech timeline locator.
        var nearEndTimeline = RuntimeGraphProbe.RequireField<ISpeechTimelineLocator>(
            runtime,
            "_timeline");
        Assert.IsType<ActiveNearEndCoordinator>(nearEndTimeline);
        Assert.True(nearEndTimeline.TryLocate(
            guardedPrefix,
            out long streamEpoch,
            out long startSample,
            out long endSample));
        Assert.Equal(1, streamEpoch);
        Assert.Equal(guardedPrefix.Length, endSample - startSample);
    }

    [Fact]
    public async Task AdaptiveBargeInDecider_RejectsLearnedPlaybackEcho()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        // 模拟高外放/无 AEC 设备: 播放参考活跃, 处理后采集残留稳定在 0.10。
        for (int i = 0; i < 30; i++)
            services.Audio.EmitMetrics(renderRms: 0.08, captureOutputRms: 0.10, captureProcessed: false);

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking(ConstantSamples(0.10f, 16000));
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(1000));
        await Task.Delay(50);

        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.Equal(AssistantState.Speaking, assistant.State);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_DuringSpeaking_StartsValidationWithoutVadSpeechStarted()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09, emitMainAudio: true);
        // Identity confirmation is disabled: early near-end evidence ducks the reply, then the
        // acoustic Interrupt stops it before the endpoint.
        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Any(e => e.Kind == BargeInValidationKind.Accepted);
        });
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Contains(0.15f, services.Audio.PlaybackGainChanges);
        Assert.Equal(AssistantState.Listening, assistant.State);
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.True(services.Asr.FedAudio["session-2"].Sum(c => c.Length) >= 160 * 16);
        lock (validationEvents)
        {
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents.Select(e => e.Kind));
            Assert.All(validationEvents, e => Assert.Equal(BargeInTriggerSource.NearEndWindow, e.Source));
            Assert.True(validationEvents[0].AudioFrameCount > 0);
            Assert.True(validationEvents[0].AudioAcceptedDurationMs > 0);
        }
        Assert.Equal(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task ActiveNearEnd_DoesNotCreatePlaybackValidationBeforeFirstRenderFrame()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        var firstRender = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockBeforeFirstChunk = firstRender;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking);

        int predictedBefore = services.Vad.PredictedAudio.Count;
        EmitNearEndFrames(services, count: 16, renderRms: 0, captureRms: 0.09, emitMainAudio: true);
        await Task.Delay(50);

        lock (validationEvents)
            Assert.Empty(validationEvents);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
        Assert.True(services.Vad.PredictedAudio.Count > predictedBefore);

        firstRender.TrySetResult();
        services.Tts.BlockBeforeFirstChunk = null;
    }

    [Fact]
    public async Task ActiveNearEndCapture_EndsOnConsecutiveLatestSilenceDespiteOlderWindowSpikes()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));
        int finishCount = services.Asr.FinishCount;

        // 模拟播放刚停止后的安静采集：滑窗中仍留有旧语音，而且每 160ms 边界
        // 恰有一个孤立尖峰。旧逻辑只看 AcceptedDurationMs>0，会永远把结束静音清零；
        // 新逻辑按最新帧连续静音，在前三个静音帧后就应收尾。
        for (int cycle = 0; cycle < 4; cycle++)
        {
            EmitNearEndFrames(services, count: 1, renderRms: 0, captureRms: 0.09, emitMainAudio: true);
            EmitSilenceFrames(services, count: 7, renderRms: 0);
        }

        await WaitUntilAsync(() => services.Asr.FinishCount == finishCount + 1);
        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.True(fed.Length < 16_000, $"Expected capture shorter than 1s, got {fed.Length / 16.0:F0}ms.");
    }

    [Fact]
    public async Task RenderAwareDtd_EndpointVadPreventsMidUtteranceCutoff()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "不好听，换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        int finishCount = services.Asr.FinishCount;
        services.Vad.EmitSpeechStarted();
        EmitSilenceFrames(services, count: 20, renderRms: 0.08);
        await Task.Delay(50);
        Assert.Equal(finishCount, services.Asr.FinishCount);

        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(600));
        EmitSilenceFrames(services, count: 8, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount == finishCount + 1);
    }

    [Fact]
    public async Task ActiveNearEndEndpointSignal_WaitsForDtdSilenceBeforeFinishingCapture()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        AssistantRuntime runtime = GetPrivateField<AssistantRuntime>(assistant, "_runtime");
        INearEndEndpointPort activeNearEnd = GetPrivateField<INearEndEndpointPort>(runtime, "_nearEnd");
        await WaitUntilAsync(() => activeNearEnd.GetProcessingState().HasActiveCapture);

        int finishCount = services.Asr.FinishCount;
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        // The matching SpeechAudioAvailable signal can reach the control loop after the VAD
        // endpoint. It must not replay that already-observed audio into the active data plane.
        RuntimeGraphProbe.InvokeMethod(
            runtime,
            "ProcessSpeaking",
            ConstantSamples(0.12f, 320),
            1L);

        // A playback endpoint is an input to the near-end DTD silence window. It must not be
        // replayed later as a state-machine speech end for a turn confirmed after this callback.
        await Task.Delay(50);
        Assert.Equal(finishCount, services.Asr.FinishCount);

        EmitSilenceFrames(services, count: 8, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount == finishCount + 1);
    }

    [Fact]
    public async Task RenderAwarePlayback_MainVadStillObservesPrimaryAecAudio()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
            });

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        int predictionsBefore = services.Vad.PredictedAudio.Count;
        EmitPlaybackEchoFrames(services, count: 4, renderRms: 0.08, echoRms: 0.02);

        Assert.Equal(predictionsBefore + 4, services.Vad.PredictedAudio.Count);
    }

    [Fact]
    public async Task RenderAwareDtd_MainVadTimelinePreservesPrefixAndActiveEndpoint()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "不好听，换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        float[] detectorOnlyPrefix = ConstantSamples(0.19f, 320 * 4);
        float[] detectorRender = AlternatingSamples(0.08, detectorOnlyPrefix.Length);
        float[] detectorEcho = Scale(detectorRender, 0.02 / 0.08);
        services.Audio.EmitFrame(
            detectorRender,
            detectorEcho,
            detectorEcho,
            detectorEcho,
            processedCapture: detectorOnlyPrefix);

        float[] vadPrefix = ConstantSamples(0.23f, 320 * 12);
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(vadPrefix);
        };
        float[] render = AlternatingSamples(0.08, vadPrefix.Length);
        float[] echo = Scale(render, 0.02 / 0.08);
        services.Audio.EmitFrame(render, echo, echo, echo, processedCapture: vadPrefix);

        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        int finishCount = services.Asr.FinishCount;
        EmitSilenceFrames(services, count: 20, renderRms: 0.08);
        await Task.Delay(50);
        Assert.Equal(finishCount, services.Asr.FinishCount);

        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(700));
        EmitSilenceFrames(services, count: 8, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount == finishCount + 1);

        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.True(fed.Length >= vadPrefix.Length);
        Assert.All(fed.AsSpan(0, vadPrefix.Length).ToArray(), sample => Assert.Equal(0.23f, sample));
        Assert.Equal(vadPrefix.Length, fed.Count(sample => Math.Abs(sample - 0.23f) < 0.0001f));
        Assert.DoesNotContain(fed, sample => Math.Abs(sample - 0.19f) < 0.0001f);
    }

    [Fact]
    public async Task RenderAwareDtd_StaleMainVadTimelineUsesDecisionHistoryWithoutOldAudio()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.PlaybackMaxCandidateMs = 700;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        float[] staleVadAudio = ConstantSamples(0.23f, 320 * 4);
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(staleVadAudio);
        };
        float[] render = AlternatingSamples(0.08, staleVadAudio.Length);
        float[] echo = Scale(render, 0.02 / 0.08);
        services.Audio.EmitFrame(render, echo, echo, echo, processedCapture: staleVadAudio);

        // 800ms exceeds the configured cross-device candidate lifetime. The old VAD segment
        // remains logically active to prove that age validation, not a speech-end event, rejects it.
        EmitPlaybackEchoFrames(services, count: 40, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        EmitSilenceFrames(services, count: 8, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);

        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.DoesNotContain(fed, sample => Math.Abs(sample - 0.23f) < 0.0001f);
    }

    [Fact]
    public async Task RenderAwareDtd_UnlocatableMainVadAudioUsesDecisionTimeline()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 60;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        float[] mainAudio = ConstantSamples(0.23f, 320 * 4);
        float[] unlocatableVadAudio = ConstantSamples(0.91f, mainAudio.Length);
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(unlocatableVadAudio);
        };
        float[] render = AlternatingSamples(0.08, mainAudio.Length);
        float[] echo = Scale(render, 0.02 / 0.08);
        services.Audio.EmitFrame(render, echo, echo, echo, processedCapture: mainAudio);

        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        EmitSilenceFrames(services, count: 8, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);

        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.DoesNotContain(fed, sample => Math.Abs(sample - 0.91f) < 0.0001f);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_DoesNotFeedMainChunkAlreadyIncludedInPreRollTwice()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        services.Vad.PredictAction = () =>
            EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09);
        float[] candidate = ConstantSamples(0.17f, 160 * 16);
        services.Audio.Emit(candidate);
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);

        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.Equal(candidate.Length, fed.Count(sample => Math.Abs(sample - 0.17f) < 0.0001f));
    }

    [Fact]
    public async Task ActiveNearEndCapture_VadSpeakingDoesNotBypassRecognitionTimeline()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        AssistantRuntime runtime = GetPrivateField<AssistantRuntime>(assistant, "_runtime");
        INearEndEndpointPort activeNearEnd = GetPrivateField<INearEndEndpointPort>(runtime, "_nearEnd");
        await WaitUntilAsync(() =>
            activeNearEnd.GetProcessingState().HasActiveCapture &&
            services.Asr.FedAudio.ContainsKey("session-2"));

        int fedBefore = services.Asr.FedAudio["session-2"].Sum(static chunk => chunk.Length);
        float[] vadOnlyChunk = ConstantSamples(0.91f, 4480);
        services.Vad.EmitSpeaking(vadOnlyChunk);
        await Task.Delay(50);

        int fedAfter = services.Asr.FedAudio["session-2"].Sum(static chunk => chunk.Length);
        Assert.Equal(fedBefore, fedAfter);
        Assert.DoesNotContain(
            services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk),
            sample => Math.Abs(sample - 0.91f) < 0.0001f);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_WaitsForDelayedRecognitionFramesThroughSpeechEndBoundary()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
                options.BargeIn.ActiveNearEndAudioDrainTimeoutMs = 1000;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));
        int finishCountBeforeSpeechEnd = services.Asr.FinishCount;

        var delayed = new List<(CaptureRange Range, float[] Audio)>();
        float[] speechTail = ConstantSamples(0.31f, 320);
        delayed.Add((
            services.Audio.EmitDuplexOnly(
                new float[320], speechTail, speechTail, speechTail),
            speechTail));
        for (int i = 0; i < 12; i++)
        {
            float[] silence = new float[320];
            // 最后两帧位于检测线程已经确定的结束边界之后；让处理链产出独特样本，
            // 验证追赶到边界时会精确停止，不把下一段音频带进本次 ASR。
            float[] processed = i >= 10 ? ConstantSamples(0.47f, 320) : silence;
            delayed.Add((
                services.Audio.EmitDuplexOnly(
                    ConstantSamples(0.08f, 320), silence, silence, silence),
                processed));
        }

        // 双工检测线程已经判停，但录音/DNR处理线程尚未产出上述范围，不能提前 Finish ASR。
        Assert.Equal(finishCountBeforeSpeechEnd, services.Asr.FinishCount);

        foreach ((CaptureRange range, float[] audio) in delayed)
            services.Audio.EmitProcessedCapture(in range, audio);

        await WaitUntilAsync(() => services.Asr.FinishCount == finishCountBeforeSpeechEnd + 1);
        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        Assert.Equal(speechTail.Length, fed.Count(sample => Math.Abs(sample - 0.31f) < 0.0001f));
        Assert.DoesNotContain(fed, sample => Math.Abs(sample - 0.47f) < 0.0001f);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_DrainTimeoutFinishesWhenRecognitionPipelineStopsProducingFrames()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
                options.BargeIn.ActiveNearEndAudioDrainTimeoutMs = 40;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.09, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));
        int finishCountBeforeSpeechEnd = services.Asr.FinishCount;

        for (int i = 0; i < 20; i++)
        {
            float[] silence = new float[320];
            services.Audio.EmitDuplexOnly(
                ConstantSamples(0.08f, 320), silence, silence, silence);
        }

        await WaitUntilAsync(() => services.Asr.FinishCount == finishCountBeforeSpeechEnd + 1);
    }

    /// <summary>
    /// Drives the KWS-enabled assistant through the owner's wake-up on the real speaker module so the
    /// near-end tests start from a conversation session with owner evidence on the identity timeline.
    /// </summary>
    private static async Task WakeAsOwnerAsync(TestServices services, VoiceAssistant assistant)
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

    /// <summary>
    /// Confirmation.Enable also opens SingleTalk, so the first command after wake-up must be
    /// locatable owner audio — dummy VAD samples are denied and never start a reply.
    /// </summary>
    private static async Task SpeakOwnerCommandUntilSpeakingAsync(
        TestServices services,
        VoiceAssistant assistant)
    {
        SpeakLocatedUtterance(services, FakeSpeakerService.OwnerAudio(16000));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_FromOtherSpeaker_DoesNotInterruptPlayback()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.PlaybackMaxCandidateMs = 700;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);

            // Another person talks over the reply: DTD sees genuine near-end speech (it would even
            // reach Interrupt), but every identity window scores against the owner. With KWS
            // enabled that is a denied command: no Heard, no stop, no gain change, no cooldown
            // leak into a later owner attempt.
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(1);
            EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
            for (int i = 0; i < 45; i++)
            {
                EmitNearEndFrames(
                    services,
                    count: 1,
                    renderRms: 0.08,
                    captureRms: 0.09,
                    emitMainAudio: true,
                    alternatingCapture: true);
                await Task.Delay(2);
            }
            await Task.Delay(100);

            lock (validationEvents)
                Assert.Empty(validationEvents);
            Assert.Equal(0, services.Audio.ClearQueueCount);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.Equal(AssistantState.Speaking, assistant.State);
            Assert.True(services.Speaker.FeedCount > 0);
            Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
            await assistant.StopAsync();
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    [Fact]
    public async Task ActiveNearEndTrigger_FromOwner_SpeakerMatchStopsPlaybackAndKeepsPreRollForAsr()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.PlaybackMaxCandidateMs = 700;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 360;
                options.BargeIn.ActiveNearEndAudioDrainTimeoutMs = 1;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        var bargeInAccepted = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
            if (e.Kind == BargeInValidationKind.Accepted)
                bargeInAccepted.TrySetResult();
        };

        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);

            EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
            // 判定窗之前的低证据句首也必须跨过 Speaker 等待期完整保留。这里故意把
            // drain timeout 设为 1ms，确保容量来自候选区间本身，而不是碰巧借用了
            // 结束排空预算。
            const int preRollFrameSamples = 320;
            const int preRollFrames = 18;
            for (int i = 0; i < preRollFrames; i++)
            {
                float[] render = AlternatingSamples(0.08, preRollFrameSamples);
                float[] echo = Scale(render, 0.02 / 0.08);
                services.Audio.EmitFrame(
                    render,
                    echo,
                    echo,
                    echo,
                    processedCapture: ConstantSamples(0.19f, preRollFrameSamples));
            }

            // Owner speech over the reply. The candidate opens on the first accepted near-end frame;
            // the reply keeps playing at full gain while the identity window fills (~500ms of
            // speech). The first scored window matches with high confidence and, by itself, issues
            // the single owner-scoped stop: no DTD Interrupt and no ASR text is needed for it.
            EmitNearEndFrames(
                services,
                count: 8,
                renderRms: 0.08,
                captureRms: 0.11,
                emitMainAudio: true,
                alternatingCapture: true);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.Equal(0, services.Audio.ClearQueueCount);
            for (int i = 0; i < 60 && !bargeInAccepted.Task.IsCompleted; i++)
            {
                EmitNearEndFrames(
                    services,
                    count: 1,
                    renderRms: 0.08,
                    captureRms: 0.09,
                    emitMainAudio: true,
                    alternatingCapture: true);
                await Task.Delay(5);
            }
            await bargeInAccepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.Equal(1, services.Audio.ClearQueueCount);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.Equal(1f, services.Audio.PlaybackGain);

            await WaitUntilAsync(() =>
                services.Asr.FedAudio.TryGetValue("session-2", out List<float[]>? chunks) &&
                chunks.SelectMany(chunk => chunk).Any(sample => Math.Abs(sample - 0.11f) < 0.0001f));
            float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
            Assert.Equal(
                preRollFrameSamples * preRollFrames,
                fed.Count(sample => Math.Abs(sample - 0.19f) < 0.0001f));
            var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
                .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(assistant)!;
            Assert.Equal(0, runtime.DeferredRecognitionReadyTurnCount);
            Assert.Equal(AssistantState.Listening, assistant.State);
            lock (validationEvents)
            {
                Assert.Equal(
                    [BargeInValidationKind.Heard, BargeInValidationKind.Accepted],
                    validationEvents.Select(static e => e.Kind).ToArray());
                Assert.Equal("speaker-match", validationEvents[^1].Reason);
            }
            await assistant.StopAsync();
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    /// <summary>
    /// A confirmed physical stop for one reply must not touch a newer reply: the owner-scoped stop
    /// compares the turn that issued it with the reply that is actually playing.
    /// </summary>
    [Fact]
    public async Task PlaybackStop_ForCompletedOwnerTurn_DoesNotStopTheNextReply()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "第二个问题");
        using var assistant = services.CreateAssistant(enableBargeIn: true);
        var replies = new List<string>();
        assistant.AssistantReplied += (_, e) =>
        {
            lock (replies)
                replies.Add(e.Text);
        };
        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle && replies.Count == 1);

        var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        var interruption = RuntimeGraphProbe.RequireField<PlaybackInterruptionController>(runtime, "_playbackInterruption");

        // A late stop for the finished first reply is ignored while nothing is playing...
        Assert.False(interruption.InterruptActiveResponse(expectedOwnerTurnId: 1, responseCts: null));
        Assert.Equal(0, services.Audio.ClearQueueCount);

        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        // ...and still ignored once the second reply plays: the owner turn no longer matches.
        Assert.False(interruption.InterruptActiveResponse(expectedOwnerTurnId: 1, responseCts: null));
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Equal(AssistantState.Speaking, assistant.State);
        Assert.Equal(2, interruption.IgnoredStops);
        Assert.Equal(0, interruption.OwnedStops);

        services.Tts.BlockAfterFirstChunk.TrySetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
    }

    [Fact]
    public void ActiveNearEndTrigger_SpeakerAuthorizationSurvivesMainChainCatchUp()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 120,
            MinRms = 0.03f,
            ActiveNearEndWindowMs = 160,
            ActiveNearEndPreRollMs = 0,
        };
        var turns = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(
                options.MinDurationMs,
                options.MinRms,
                sampleRate: 16000,
                channels: 1));
        long responseTurnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryTransitionForTurn(
            responseTurnId,
            AssistantState.Speaking,
            out _,
            out _));

        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 8,
            AcceptedFrameCount: 8,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 160,
            AcceptedDurationMs: 160,
            RequiredDurationMs: 120,
            ScoreDurationMs: 160,
            PeakScoreDurationMs: 160,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.03,
            MinRequiredRms: 0.03,
            MaxRequiredRms: 0.03,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 800,
            EndSampleExclusive: 2000,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 800);
        BargeInDecision GetRecentDecision(TimeSpan window, int minDurationMs, float minRms)
            => decision;
        var speakerEvidence = new SequencedSpeakerEvidenceProvider(
            SpeakerVerificationStatus.Match,
            SpeakerVerificationStatus.Mismatch);
        int confirmedCount = 0;
        long confirmationOperationId = 0;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            () => turns.IsSpeakingWithoutPendingCandidate,
            GetRecentDecision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: operationId =>
            {
                confirmationOperationId = operationId;
                return true;
            },
            publishSpeechEnded: static (_, _) => { },
             speakerEvidence,
            reportDataPlaneFault: static _ => { });

        var firstMainRange = new CaptureRange(
            StreamEpoch: 1,
            StartSequenceNumber: 1,
            EndSequenceNumber: 1,
            StartSample: 0,
            EndSampleExclusive: 1600,
            TimestampTicks: 1);
        coordinator.ObserveRecognitionAudio(new ProcessedCaptureFrame(
            firstMainRange,
            SampleRate: 16000,
            Channels: 1,
            Audio: ConstantSamples(0.1f, 1600),
            DelaySamples: 0,
            HasDiscontinuity: false));

        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 2, startSample: 1600, sampleCount: 640),
            playbackStartupGuardActive: false);
        ActiveNearEndDiagnosticSnapshot authorized =
            coordinator.GetDiagnosticSnapshot();
        Assert.True(authorized.CandidateActive);
        Assert.True(authorized.CandidateAuthorized);
        Assert.Equal(1, speakerEvidence.EvaluateCount);
        Assert.Equal(0, confirmedCount);

        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 3, startSample: 2240, sampleCount: 320),
            playbackStartupGuardActive: false);
        Assert.Equal(1, speakerEvidence.EvaluateCount);
        Assert.True(coordinator.GetDiagnosticSnapshot().CandidateAuthorized);

        var catchUpRange = new CaptureRange(
            StreamEpoch: 1,
            StartSequenceNumber: 2,
            EndSequenceNumber: 2,
            StartSample: 1600,
            EndSampleExclusive: 2240,
            TimestampTicks: 2);
        coordinator.ObserveRecognitionAudio(new ProcessedCaptureFrame(
            catchUpRange,
            SampleRate: 16000,
            Channels: 1,
            Audio: ConstantSamples(0.1f, 640),
            DelaySamples: 0,
            HasDiscontinuity: false));

        Assert.True(coordinator.TryTakeConfirmation(
            confirmationOperationId,
            out ActiveNearEndConfirmationRequest request));
        Assert.True(request.IdentityRequired);
        Assert.True(request.IdentityAuthorized);
        Assert.Equal(new SpeakerCandidateId(1), request.CandidateId);
        Assert.Equal(SpeakerVerificationStatus.Match, request.SpeakerResult.Status);
        Assert.True(turns.TryConfirmBargeInFromNearEndWindow(
            request.InitialAudio,
            request.Decision,
            out SpeakingDecision confirmed,
            identityAuthorized: request.IdentityAuthorized));
        coordinator.ApplyConfirmedCapture(request, confirmed);
        confirmedCount++;

        Assert.Equal(1, confirmedCount);
        Assert.False(coordinator.GetDiagnosticSnapshot().CandidateActive);
        Assert.True(turns.HasPendingBargeInCandidateOrValidation);
    }

    /// <summary>
    /// KWS enabled: the speaker Match alone authorizes the candidate. A DTD decision that never
    /// reaches <c>Interrupt</c> (only accepted near-end speech) still yields a confirmation whose
    /// identity facts carry the Match, and the applier confirms it without an acoustic Interrupt.
    /// </summary>
    [Fact]
    public void ActiveNearEndTrigger_SpeakerMatchConfirmsWithoutDtdInterrupt()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 120,
            MinRms = 0.03f,
            ActiveNearEndWindowMs = 160,
            ActiveNearEndPreRollMs = 0,
        };
        var turns = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(options.MinDurationMs, options.MinRms, 16000, 1));
        long responseTurnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryTransitionForTurn(responseTurnId, AssistantState.Speaking, out _, out _));

        var decision = new BargeInDecision(
            BargeInAction.Continue,
            FrameCount: 4,
            AcceptedFrameCount: 4,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 80,
            AcceptedDurationMs: 80,
            RequiredDurationMs: 120,
            ScoreDurationMs: 80,
            PeakScoreDurationMs: 80,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.03,
            MinRequiredRms: 0.03,
            MaxRequiredRms: 0.03,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 1920,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        var speakerEvidence = new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match);
        long confirmationOperationId = 0;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            () => turns.IsSpeakingWithoutPendingCandidate,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: operationId =>
            {
                confirmationOperationId = operationId;
                return true;
            },
            publishSpeechEnded: static (_, _) => { },
            speakerEvidence,
            reportDataPlaneFault: static _ => { });

        coordinator.ObserveRecognitionAudio(new ProcessedCaptureFrame(
            new CaptureRange(1, 1, 1, 0, 1920, 1),
            SampleRate: 16000,
            Channels: 1,
            Audio: ConstantSamples(0.1f, 1920),
            DelaySamples: 0,
            HasDiscontinuity: false));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 1920),
            playbackStartupGuardActive: false);

        Assert.Equal(1, speakerEvidence.EvaluateCount);
        Assert.Equal(SpeakerVerificationPurpose.BargeInCommand, speakerEvidence.Queries[0].Purpose);
        Assert.Equal(1, coordinator.CommandAuthorizedCount);
        Assert.True(coordinator.TryTakeConfirmation(confirmationOperationId, out ActiveNearEndConfirmationRequest request));
        Assert.Equal(BargeInAction.Continue, request.Decision.Action);
        Assert.True(request.IdentityAuthorized);

        // The acoustic path alone would not confirm this window...
        Assert.False(turns.TryObserveBargeInFromNearEndWindow(request.InitialAudio, request.Decision, out _));
        // ...but the candidate's own speaker Match does.
        Assert.True(turns.TryConfirmBargeInFromNearEndWindow(
            request.InitialAudio,
            request.Decision,
            out SpeakingDecision confirmed,
            identityAuthorized: true));
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, confirmed.Kind);
        Assert.Equal(responseTurnId, confirmed.InterruptedTurnId);
    }

    [Fact]
    public async Task ActiveNearEndCapture_DelayedAsrAttachAfterCaptureEnd_PreservesPcmOffProducerThread()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 1,
            ActiveNearEndWindowMs = 360,
            ActiveNearEndEndSilenceMs = 1,
            ActiveNearEndAudioDrainTimeoutMs = 100,
        };
        int drainRequests = 0;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            isSpeakingWithoutPendingCandidate: static () => true,
            getRecentDecision: (_, _, _) => new BargeInDecision(
                BargeInAction.Continue,
                FrameCount: 0,
                AcceptedFrameCount: 0,
                RejectedFrameCount: 0,
                EchoLikeFrameCount: 0,
                TotalDurationMs: 0,
                AcceptedDurationMs: 0,
                RequiredDurationMs: 0,
                ScoreDurationMs: 0,
                PeakScoreDurationMs: 0,
                LastFrameRms: 0,
                LastRequiredRms: 0,
                MinRequiredRms: 0,
                MaxRequiredRms: 0,
                Reason: "silence",
                StreamEpoch: 1,
                StartSample: 0,
                EndSampleExclusive: 0,
                LatestFrameAccepted: false),
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: static _ => true,
             publishSpeechEnded: static (_, _) => { },
            reportDataPlaneFault: static _ => { });

        var request = new ActiveNearEndConfirmationRequest(
            OperationId: 1,
            InitialAudio: [],
            Decision: default,
            Selection: new ActiveNearEndPreRollSelection(
                "test",
                Samples: 0,
                StreamEpoch: 1,
                DecisionStartSample: 0,
                DecisionEndSample: 0,
                SelectedStartSample: 0,
                SelectedEndSample: 0),
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            CandidateId: default,
            OwnerTurnId: 0,
            SpeakerResult: default);
        var confirmed = new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            ResponseToCancel: null,
            InterruptedTurnId: 0,
            OldState: AssistantState.Speaking,
            TurnId: 1,
            ConfirmedCandidateSamples: 0,
            ConfirmedCandidateRms: 0,
            ConfirmedConfirmationSamples: 0,
            ConfirmedConfirmationRms: 0,
            ConfirmationMinDurationMs: 0,
            ConfirmationMinRms: 0,
            ConfirmationInitialSuppressionMs: 0,
            ConfirmedBargeInDiagnostics: default);

        coordinator.ApplyConfirmedCapture(request, confirmed);

        int producerThreadId = Environment.CurrentManagedThreadId;
        FeedRecognitionAudio(coordinator, startSample: 0, samples: [1f, 2f, 3f, 4f]);

        Assert.DoesNotContain(producerThreadId, services.Asr.FeedThreadIds);
        Assert.Equal(0, drainRequests);
        Assert.Empty(services.Asr.FedAudio);

        Assert.True(coordinator.StopCapture(1, startCooldown: false));
        await coordinator.AttachAsrSessionAsync(1, "validation-1");
        Assert.Equal(
            [1f, 2f, 3f, 4f],
            services.Asr.FedAudio["validation-1"].SelectMany(static chunk => chunk));

        static void FeedRecognitionAudio(
            ActiveNearEndCoordinator coordinator,
            long startSample,
            float[] samples)
        {
            var range = new CaptureRange(
                StreamEpoch: 1,
                StartSequenceNumber: startSample + 1,
                EndSequenceNumber: startSample + 1,
                StartSample: startSample,
                EndSampleExclusive: startSample + samples.Length,
                TimestampTicks: startSample + 1);
            var frame = new ProcessedCaptureFrame(
                range,
                SampleRate: 16000,
                Channels: 1,
                Audio: samples,
                DelaySamples: 0,
                HasDiscontinuity: false);
            MainAudioObservation observation =
                coordinator.ObserveRecognitionAudio(frame);
            coordinator.FeedFromRecognitionAudio(samples, observation);
        }
    }

    [Fact]
    public async Task ControlLoop_WakeupSideQueueOperationMismatchTerminatesIngress()
    {
        using var ingress = new AssistantControlIngress();
        var queue = new CorrelatedObservationQueue<WakeupAudioDecision>(
            ingress,
            capacity: 8,
            "wakeup-data-plane",
            overflowCode: "VBX-CTRL-008",
            mismatchCode: "VBX-CTRL-009",
            mismatchMessageKey: "wakeup-operation-correlation");

        Assert.True(queue.TryWrite(1, default, "wakeup-observation-overflow"));
        Assert.False(ingress.HasTerminalFault);

        Assert.False(queue.TryTake(2, out _));

        Assert.True(ingress.HasTerminalFault);
        await Task.CompletedTask;
    }

    [Fact]
    public async Task ControlLoop_SpeakerGateSideQueueOperationMismatchTerminatesIngress()
    {
        using var ingress = new AssistantControlIngress();
        var host = new VadSpeakerGateHost(
            ingress,
            new ControlOperationRegistry(),
            new AssistantTaskSupervisor(NullLogger.Instance),
            new RuntimeRunScope());

        host.SubmitGateResolved(1, SpeakerVerificationPurpose.BargeInCommand, BargeInSpeakerGateStatus.Rejected, pendingSpeechEnd: null);
        Assert.False(ingress.HasTerminalFault);

        Assert.False(host.TryTakeObservation(2, out _, out _, out _));

        Assert.True(ingress.HasTerminalFault);
        await Task.CompletedTask;
    }

    [Fact]
    public async Task ControlLoop_CanceledSignalAdmissionRemovesPendingCompletion()
    {
        using var ingress = new AssistantControlIngress();
        using var loopHost = new AssistantControlLoopHost(ingress, TimeProvider.System);
        loopHost.Bind(new IdleControlLoopHandler());
        loopHost.EnsureStarted();
        var runScope = new RuntimeRunScope();
        runScope.Begin();
        var operations = new ControlOperationRegistry();
        var gateway = new AssistantControlSignalGateway(
            ingress,
            operations,
            new OperationCorrelationGate(),
            new ControlPlaneAvailability(runScope, ingress, loopHost),
            new TurnSnapshotStore(),
            new TurnContextStore(new OperationCorrelationGate()),
            new InputGate(),
            runScope);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            gateway.PublishRecognitionCompletedAsync(
                turnId: 1,
                text: "canceled",
                speechDuration: TimeSpan.Zero,
                cancellation.Token).AsTask());

        Assert.Equal(0, operations.PendingSignalCount);
        loopHost.RequestStop();
    }

    [Fact]
    public async Task ControlLoop_CancellationAfterSignalAdmissionReleasesWaitingProducer()
    {
        using var ingress = new AssistantControlIngress();
        using var loopHost = new AssistantControlLoopHost(ingress, TimeProvider.System);
        var handler = new IdleControlLoopHandler();
        loopHost.Bind(handler);
        loopHost.EnsureStarted();
        var runScope = new RuntimeRunScope();
        runScope.Begin();
        var operations = new ControlOperationRegistry();
        var gateway = new AssistantControlSignalGateway(
            ingress,
            operations,
            new OperationCorrelationGate(),
            new ControlPlaneAvailability(runScope, ingress, loopHost),
            new TurnSnapshotStore(),
            new TurnContextStore(new OperationCorrelationGate()),
            new InputGate(),
            runScope);
        using var cancellation = new CancellationTokenSource();

        Task publish = gateway.PublishRecognitionCompletedAsync(
            turnId: 1,
            text: "cancel-after-admission",
            speechDuration: TimeSpan.Zero,
            cancellation.Token).AsTask();
        await handler.SignalObserved.Task.WaitAsync(TimeSpan.FromSeconds(1));

        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => publish);
        Assert.Equal(0, operations.PendingSignalCount);
        loopHost.RequestStop();
    }

    [Fact]
    public void ControlLoop_FailedRegistryRejectsLateSignalRegistration()
    {
        var operations = new ControlOperationRegistry();
        operations.FailPending("test-stop");

        OperationCanceledException exception = Assert.Throws<OperationCanceledException>(
            () => operations.RegisterSignal(AssistantSignal.ResponseStarted(1, 1)));

        Assert.Contains("test-stop", exception.Message, StringComparison.Ordinal);
        Assert.Equal(0, operations.PendingSignalCount);
    }

    [Fact]
    public void ControlLoop_CompletionCleanupCannotRemoveTheNextSignalForTheSameOperation()
    {
        var operations = new ControlOperationRegistry();
        AssistantSignal recognition = AssistantSignal.RecognitionCompleted(
            turnId: 1,
            operationId: 5,
            text: "recognized",
            speechDuration: TimeSpan.Zero);
        AssistantSignal response = AssistantSignal.ResponseStarted(turnId: 1, operationId: 5);

        TaskCompletionSource<AssistantControlSignalResult> recognitionCompletion =
            operations.RegisterSignal(recognition);
        operations.CompleteSignal(
            recognition,
            new AssistantControlSignalResult(true, true, 1));
        TaskCompletionSource<AssistantControlSignalResult> responseCompletion =
            operations.RegisterSignal(response);

        operations.ForgetSignal(recognition, recognitionCompletion);
        operations.CompleteSignal(
            response,
            new AssistantControlSignalResult(true, false, 0));

        Assert.True(responseCompletion.Task.IsCompletedSuccessfully);
        Assert.Equal(0, operations.PendingSignalCount);
    }

    /// <summary>Consumer that accepts every envelope so admission can be tested in isolation.</summary>
    private sealed class IdleControlLoopHandler : IAssistantControlLoopHandler
    {
        public TaskCompletionSource SignalObserved { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public ValueTask HandleRuntimeCommandAsync(
            RuntimeCommand command,
            ControlEnvelope envelope,
            CancellationToken cancellationToken) => ValueTask.CompletedTask;

        public ValueTask HandleAssistantSignalAsync(
            AssistantSignal signal,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            SignalObserved.TrySetResult();
            return ValueTask.CompletedTask;
        }

        public ValueTask HandleTerminalFaultAsync(
            AssistantFailure failure,
            ControlEnvelope envelope,
            CancellationToken cancellationToken) => ValueTask.CompletedTask;

        public ValueTask HandleAbandonedAfterBarrierAsync(
            ControlEnvelope envelope,
            CancellationToken cancellationToken) => ValueTask.CompletedTask;
    }

    [Fact]
    public async Task ActiveNearEndCapture_AsrFeedQueueOverflowReportsOneTerminalFault()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 1,
            ActiveNearEndWindowMs = 360,
            ActiveNearEndEndSilenceMs = 1,
            ActiveNearEndAudioDrainTimeoutMs = 100,
        };
        var faults = new List<AssistantFailure>();
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr, directFeedAttached: false),
            isSpeakingWithoutPendingCandidate: static () => true,
            getRecentDecision: (_, _, _) => new BargeInDecision(
                BargeInAction.Continue,
                FrameCount: 0,
                AcceptedFrameCount: 0,
                RejectedFrameCount: 0,
                EchoLikeFrameCount: 0,
                TotalDurationMs: 0,
                AcceptedDurationMs: 0,
                RequiredDurationMs: 0,
                ScoreDurationMs: 0,
                PeakScoreDurationMs: 0,
                LastFrameRms: 0,
                LastRequiredRms: 0,
                MinRequiredRms: 0,
                MaxRequiredRms: 0,
                Reason: "silence",
                StreamEpoch: 1,
                StartSample: 0,
                EndSampleExclusive: 0,
                LatestFrameAccepted: false),
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: static _ => true,
             publishSpeechEnded: static (_, _) => { },
            reportDataPlaneFault: faults.Add);

        var request = new ActiveNearEndConfirmationRequest(
            OperationId: 1,
            InitialAudio: [],
            Decision: default,
            Selection: new ActiveNearEndPreRollSelection(
                "test",
                Samples: 0,
                StreamEpoch: 1,
                DecisionStartSample: 0,
                DecisionEndSample: 0,
                SelectedStartSample: 0,
                SelectedEndSample: 0),
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            CandidateId: default,
            OwnerTurnId: 0,
            SpeakerResult: default);
        var confirmed = new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            ResponseToCancel: null,
            InterruptedTurnId: 0,
            OldState: AssistantState.Speaking,
            TurnId: 1,
            ConfirmedCandidateSamples: 0,
            ConfirmedCandidateRms: 0,
            ConfirmedConfirmationSamples: 0,
            ConfirmedConfirmationRms: 0,
            ConfirmationMinDurationMs: 0,
            ConfirmationMinRms: 0,
            ConfirmationInitialSuppressionMs: 0,
            ConfirmedBargeInDiagnostics: default);

        coordinator.ApplyConfirmedCapture(request, confirmed);
        await coordinator.AttachAsrSessionAsync(1, "validation-1");

        for (int index = 0; index < 258; index++)
            FeedRecognitionAudio(coordinator, index, [(float)index]);

        AssistantFailure failure = Assert.Single(faults);
        Assert.Equal("VBX-CTRL-018", failure.Code);
        Assert.Equal("asr-feed-queue-overflow", failure.MessageKey);

        static void FeedRecognitionAudio(
            ActiveNearEndCoordinator coordinator,
            long startSample,
            float[] samples)
        {
            var range = new CaptureRange(
                StreamEpoch: 1,
                StartSequenceNumber: startSample + 1,
                EndSequenceNumber: startSample + 1,
                StartSample: startSample,
                EndSampleExclusive: startSample + samples.Length,
                TimestampTicks: startSample + 1);
            var frame = new ProcessedCaptureFrame(
                range,
                SampleRate: 16000,
                Channels: 1,
                Audio: samples,
                DelaySamples: 0,
                HasDiscontinuity: false);
            MainAudioObservation observation =
                coordinator.ObserveRecognitionAudio(frame);
            coordinator.FeedFromRecognitionAudio(samples, observation);
        }
    }

    [Fact]
    public void ActiveNearEndCapture_PendingAsrQueueOverflowReportsOneTerminalFault()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());

        for (int index = 0; index < 258; index++)
            FeedRecognitionAudio(coordinator, index, [(float)index]);

        AssistantFailure failure = Assert.Single(faults);
        Assert.Equal("VBX-CTRL-018", failure.Code);
        Assert.Equal("asr-pending-queue-overflow", failure.MessageKey);
        Assert.Empty(services.Asr.FedAudio);
    }

    [Fact]
    public async Task ActiveNearEndCapture_PreservesDirectFeedCallbackBoundariesBeforePublishingSpeechEnd()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 1,
            ActiveNearEndWindowMs = 360,
            ActiveNearEndEndSilenceMs = 1,
            ActiveNearEndAudioDrainTimeoutMs = 100,
        };
        int speechEnded = 0;
        ActiveNearEndCoordinator? drainOwner = null;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            isSpeakingWithoutPendingCandidate: static () => true,
            getRecentDecision: (_, _, _) => new BargeInDecision(
                BargeInAction.Continue,
                FrameCount: 0,
                AcceptedFrameCount: 0,
                RejectedFrameCount: 0,
                EchoLikeFrameCount: 0,
                TotalDurationMs: 0,
                AcceptedDurationMs: 0,
                RequiredDurationMs: 0,
                ScoreDurationMs: 0,
                PeakScoreDurationMs: 0,
                LastFrameRms: 0,
                LastRequiredRms: 0,
                MinRequiredRms: 0,
                MaxRequiredRms: 0,
                Reason: "silence",
                StreamEpoch: 1,
                StartSample: 0,
                EndSampleExclusive: 0,
                LatestFrameAccepted: false),
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: static _ => true,
             publishSpeechEnded: (_, _) => speechEnded++,
            reportDataPlaneFault: static _ => { });
        drainOwner = coordinator;

        var request = new ActiveNearEndConfirmationRequest(
            OperationId: 1,
            InitialAudio: [],
            Decision: default,
            Selection: new ActiveNearEndPreRollSelection(
                "test",
                Samples: 0,
                StreamEpoch: 1,
                DecisionStartSample: 0,
                DecisionEndSample: 0,
                SelectedStartSample: 0,
                SelectedEndSample: 0),
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            CandidateId: default,
            OwnerTurnId: 0,
            SpeakerResult: default);
        var confirmed = new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            ResponseToCancel: null,
            InterruptedTurnId: 0,
            OldState: AssistantState.Speaking,
            TurnId: 1,
            ConfirmedCandidateSamples: 0,
            ConfirmedCandidateRms: 0,
            ConfirmedConfirmationSamples: 0,
            ConfirmedConfirmationRms: 0,
            ConfirmationMinDurationMs: 0,
            ConfirmationMinRms: 0,
            ConfirmationInitialSuppressionMs: 0,
            ConfirmedBargeInDiagnostics: default);

        coordinator.ApplyConfirmedCapture(request, confirmed);
        await coordinator.AttachAsrSessionAsync(1, "validation-1");

        FeedRecognitionAudio(coordinator, startSample: 0, samples: [1f, 2f, 3f, 4f]);
        FeedRecognitionAudio(coordinator, startSample: 4, samples: [5f, 6f, 7f, 8f]);
        FeedRecognitionAudio(coordinator, startSample: 8, samples: [9f, 10f, 11f, 12f]);
        FeedRecognitionAudio(coordinator, startSample: 12, samples: [13f, 14f, 15f, 16f]);
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 16, sampleCount: 4),
            playbackStartupGuardActive: false);
        FeedRecognitionAudio(coordinator, startSample: 16, samples: [17f, 18f, 19f, 20f]);

        Assert.Equal(1, speechEnded);
        Assert.Equal(
            [
                [1f, 2f, 3f, 4f],
                [5f, 6f, 7f, 8f],
                [9f, 10f, 11f, 12f],
                [13f, 14f, 15f, 16f],
                [17f, 18f, 19f, 20f],
            ],
            services.Asr.FedAudio["validation-1"]);
        Assert.Equal(
            Enumerable.Range(1, 20).Select(static value => (float)value).ToArray(),
            services.Asr.FedAudio["validation-1"].SelectMany(static chunk => chunk));

        static void FeedRecognitionAudio(
            ActiveNearEndCoordinator coordinator,
            long startSample,
            float[] samples)
        {
            var range = new CaptureRange(
                StreamEpoch: 1,
                StartSequenceNumber: startSample + 1,
                EndSequenceNumber: startSample + 1,
                StartSample: startSample,
                EndSampleExclusive: startSample + samples.Length,
                TimestampTicks: startSample + 1);
            var frame = new ProcessedCaptureFrame(
                range,
                SampleRate: 16000,
                Channels: 1,
                Audio: samples,
                DelaySamples: 0,
                HasDiscontinuity: false);
            MainAudioObservation observation =
                coordinator.ObserveRecognitionAudio(frame);
            coordinator.FeedFromRecognitionAudio(samples, observation);
        }
    }

    [Fact]
    public async Task ActiveNearEndCapture_StartsDirectFeedAtConfirmationTail()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 180,
            ActiveNearEndWindowMs = 360,
            ActiveNearEndEndSilenceMs = 1,
            ActiveNearEndAudioDrainTimeoutMs = 100,
        };
        int speechEnded = 0;
        ActiveNearEndCoordinator? drainOwner = null;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            isSpeakingWithoutPendingCandidate: static () => true,
            getRecentDecision: (_, _, _) => new BargeInDecision(
                BargeInAction.Continue,
                FrameCount: 0,
                AcceptedFrameCount: 0,
                RejectedFrameCount: 0,
                EchoLikeFrameCount: 0,
                TotalDurationMs: 0,
                AcceptedDurationMs: 0,
                RequiredDurationMs: 0,
                ScoreDurationMs: 0,
                PeakScoreDurationMs: 0,
                LastFrameRms: 0,
                LastRequiredRms: 0,
                MinRequiredRms: 0,
                MaxRequiredRms: 0,
                Reason: "silence",
                StreamEpoch: 1,
                StartSample: 0,
                EndSampleExclusive: 0,
                LatestFrameAccepted: false),
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: static _ => true,
             publishSpeechEnded: (_, _) => speechEnded++,
            reportDataPlaneFault: static _ => { });
        drainOwner = coordinator;

        FeedRecognitionAudio(
            coordinator,
            startSample: 0,
            samples: [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]);
        var request = new ActiveNearEndConfirmationRequest(
            OperationId: 1,
            InitialAudio: [],
            Decision: default,
            Selection: new ActiveNearEndPreRollSelection(
                "test",
                Samples: 4,
                StreamEpoch: 1,
                DecisionStartSample: 0,
                DecisionEndSample: 8,
                SelectedStartSample: 0,
                SelectedEndSample: 4),
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            CandidateId: default,
            OwnerTurnId: 0,
            SpeakerResult: default);
        var confirmed = new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            ResponseToCancel: null,
            InterruptedTurnId: 0,
            OldState: AssistantState.Speaking,
            TurnId: 1,
            ConfirmedCandidateSamples: 0,
            ConfirmedCandidateRms: 0,
            ConfirmedConfirmationSamples: 0,
            ConfirmedConfirmationRms: 0,
            ConfirmationMinDurationMs: 0,
            ConfirmationMinRms: 0,
            ConfirmationInitialSuppressionMs: 0,
            ConfirmedBargeInDiagnostics: default);

        coordinator.ApplyConfirmedCapture(request, confirmed);
        Assert.Equal(8L, GetPrivateField<long>(coordinator, "_activeDirectFeedNextSample"));
        await coordinator.AttachAsrSessionAsync(1, "validation-1");
        Assert.Equal(
            [5f, 6f, 7f, 8f],
            services.Asr.FedAudio["validation-1"].SelectMany(static chunk => chunk));

        FeedRecognitionAudio(coordinator, startSample: 8, samples: [9f, 10f, 11f, 12f]);
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 2, startSample: 12, sampleCount: 4),
            playbackStartupGuardActive: false);
        FeedRecognitionAudio(coordinator, startSample: 12, samples: [13f, 14f, 15f, 16f]);

        Assert.Equal(1, speechEnded);
        Assert.Equal(
            [5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f],
            services.Asr.FedAudio["validation-1"].SelectMany(static chunk => chunk));

        static void FeedRecognitionAudio(
            ActiveNearEndCoordinator coordinator,
            long startSample,
            float[] samples)
        {
            var range = new CaptureRange(
                StreamEpoch: 1,
                StartSequenceNumber: startSample + 1,
                EndSequenceNumber: startSample + 1,
                StartSample: startSample,
                EndSampleExclusive: startSample + samples.Length,
                TimestampTicks: startSample + 1);
            var frame = new ProcessedCaptureFrame(
                range,
                SampleRate: 16000,
                Channels: 1,
                Audio: samples,
                DelaySamples: 0,
                HasDiscontinuity: false);
            MainAudioObservation observation =
                coordinator.ObserveRecognitionAudio(frame);
            coordinator.FeedFromRecognitionAudio(samples, observation);
        }
    }

    [Fact]
    public async Task ActiveNearEndCapture_ProcessedFrameAfterConfirmationRequestIsFedFromReservoir()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(selectedEndSample: 4));

        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        ObserveRecognitionAudio(coordinator, 4, [5f, 6f, 7f, 8f]);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-processed-gap");

        Assert.Empty(faults);
        Assert.Equal(
            [5f, 6f, 7f, 8f],
            services.Asr.FedAudio["validation-processed-gap"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_PendingConfirmationReplaysMatchingTailOnce()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        var request = CreateConfirmationRequest(selectedEndSample: 2);
        SetPrivateField(coordinator, "_pendingConfirmation", request);

        FeedRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 4),
            playbackStartupGuardActive: false);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-tail");

        Assert.Empty(faults);
        Assert.Equal([3f, 4f], services.Asr.FedAudio["validation-tail"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_FirstTailReplaysFromConfirmationBoundary()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(selectedEndSample: 2));

        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]);
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 2, startSample: 4, sampleCount: 4),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: false);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-full-first-tail");

        Assert.Empty(faults);
        Assert.Equal(
            [3f, 4f, 5f, 6f, 7f, 8f],
            services.Asr.FedAudio["validation-full-first-tail"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public void ActiveNearEndCapture_QueuedConfirmationDoesNotApplyEarlierVadEndpoint()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => speechEnded++);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(0));

        // Before control-plane confirmation, this VAD endpoint belongs to the still-pending
        // candidate; it must be ignored while no capture is active yet.
        coordinator.ObserveEndpointSpeechEnded();
        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));

        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());

        Assert.Equal(1, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Equal(0, speechEnded);
        Assert.Empty(faults);
    }

    [Fact]
    public async Task ActiveNearEndCapture_EndpointProcessedBeforeConfirmationDrainsKnownTail()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => speechEnded++);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(0));

        // VAD records its endpoint and the control loop consumes that endpoint before the
        // queued confirmation. The DTD tail arrives in the same callback burst.
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);
        coordinator.ObserveUnownedEndpointSpeechEnded();
        coordinator.ObserveEndpointSpeechEnded();
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 2, startSample: 2, sampleCount: 2),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: false);
        ObserveRecognitionAudio(coordinator, 2, [3f, 4f]);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-endpoint-before-confirmation");

        Assert.Equal(1, speechEnded);
        Assert.Empty(faults);
        Assert.Equal(
            [1f, 2f, 3f, 4f],
            services.Asr.FedAudio["validation-endpoint-before-confirmation"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_EndpointBeforeConfirmationWithNoTailFinalizesImmediately()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => speechEnded++);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));

        // The endpoint shares the confirmation boundary, leaving no DTD frame for the
        // confirmation-tail queue to replay. Its control-plane observation has also completed,
        // so capture must still complete without waiting for a future callback.
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);
        coordinator.ObserveUnownedEndpointSpeechEnded();
        coordinator.ObserveEndpointSpeechEnded();

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-endpoint-no-tail");

        Assert.Equal(1, speechEnded);
        Assert.Empty(faults);
        Assert.Empty(services.Asr.FedAudio);
    }

    [Fact]
    public async Task ActiveNearEndCapture_PendingEndpointDoesNotFinalizeSubsequentCapture()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => speechEnded++);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));

        // The capture callback records an endpoint, but its queued control-plane signal has
        // not run when near-end confirmation commits. Later recognition frames must remain
        // eligible for the new ASR session rather than being cut at the pre-roll boundary.
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);
        coordinator.ObserveUnownedEndpointSpeechEnded();
        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));

        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        FeedRecognitionAudio(coordinator, 2, [3f, 4f]);
        await coordinator.AttachAsrSessionAsync(1, "validation-pending-endpoint-continuation");

        Assert.Equal(1, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Equal(0, speechEnded);
        Assert.Empty(faults);
        Assert.Equal(
            [3f, 4f],
            services.Asr.FedAudio["validation-pending-endpoint-continuation"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_EndpointBeforeConfirmationBoundaryDoesNotFinalize()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => speechEnded++);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(selectedEndSample: 4));

        // The endpoint reached the control plane, but DTD advanced the immutable confirmation
        // boundary beyond it. This is the E2E ordering that previously truncated ASR to pre-roll.
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);
        coordinator.ObserveUnownedEndpointSpeechEnded();
        coordinator.ObserveEndpointSpeechEnded();
        ObserveRecognitionAudio(coordinator, 2, [3f, 4f]);
        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));

        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        FeedRecognitionAudio(coordinator, 4, [5f, 6f]);
        await coordinator.AttachAsrSessionAsync(1, "validation-stale-endpoint-continuation");

        Assert.Equal(1, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Equal(0, speechEnded);
        Assert.Empty(faults);
        Assert.Equal(
            [5f, 6f],
            services.Asr.FedAudio["validation-stale-endpoint-continuation"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_TakenConfirmationReplaysExactTailAndDuration()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f, 5f]);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 5),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));

        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-taken-tail");

        Assert.Empty(faults);
        Assert.False(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        Assert.Equal(0.6d, GetPrivateField<double>(coordinator, "_speechDurationMs"), 6);
        Assert.Equal(
            [3f, 4f, 5f],
            services.Asr.FedAudio["validation-taken-tail"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_TailWaitsForRecognitionTimelineThenDrainsOnce()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 5),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());

        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        Assert.Equal(0d, GetPrivateField<double>(coordinator, "_speechDurationMs"));

        FeedRecognitionAudio(coordinator, 2, [3f, 4f, 5f]);
        await coordinator.AttachAsrSessionAsync(1, "validation-late-tail");

        Assert.Empty(faults);
        Assert.False(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        Assert.Equal(0.6d, GetPrivateField<double>(coordinator, "_speechDurationMs"), 6);
        Assert.Equal(
            [3f, 4f, 5f],
            services.Asr.FedAudio["validation-late-tail"].SelectMany(static chunk => chunk));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ActiveNearEndCapture_ConfirmationTailRejectsEitherTimelineIdentityMismatch(
        bool generationMismatch)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 4),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);

        if (generationMismatch)
        {
            long generation = GetPrivateField<long>(coordinator, "_mainAudioGeneration");
            SetPrivateField(coordinator, "_mainAudioGeneration", generation + 1);
        }
        else
        {
            SetPrivateField(coordinator, "_mainAudioStreamEpoch", 2L);
        }

        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());

        AssistantFailure failure = Assert.Single(faults);
        Assert.Equal("confirmation-tail-timeline-mismatch", failure.MessageKey);
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ActiveNearEndCapture_DiscontinuityRetiresConfirmationTailWithoutFault(
        bool explicitDiscontinuity)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        await coordinator.AttachAsrSessionAsync(1, "validation-discontinuity-tail");
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);

        Assert.True(coordinator.TryTakeConfirmation(1, out var taken));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 5),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);
        coordinator.ApplyConfirmedCapture(taken, CreateConfirmedDecision());
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));

        float[] discontinuousAudio = [5f, 6f];
        var discontinuousRange = new CaptureRange(
            StreamEpoch: 1,
            StartSequenceNumber: 2,
            EndSequenceNumber: 2,
            StartSample: 5,
            EndSampleExclusive: 7,
            TimestampTicks: 2);
        var discontinuousFrame = new ProcessedCaptureFrame(
            discontinuousRange,
            SampleRate: 16000,
            Channels: 1,
            Audio: discontinuousAudio,
            DelaySamples: 0,
            HasDiscontinuity: explicitDiscontinuity);
        MainAudioObservation discontinuousObservation =
            coordinator.ObserveRecognitionAudio(discontinuousFrame);
        coordinator.FeedFromRecognitionAudio(discontinuousAudio, discontinuousObservation);

        Assert.Empty(faults);
        Assert.False(HasQueuedItem(coordinator, "_confirmationTailFrames"));

        FeedRecognitionAudio(coordinator, 7, [8f, 9f]);
        Assert.Equal(
            [3f, 4f, 8f, 9f],
            services.Asr.FedAudio["validation-discontinuity-tail"].SelectMany(static chunk => chunk));
    }

    [Theory]
    [InlineData(0L, true)]
    [InlineData(1L, false)]
    public void ActiveNearEndCapture_ConfirmationTailWaitsForFullyActiveDirectFeed(
        long activeTurnId,
        bool directFeed)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(2));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        Assert.True(coordinator.TryTakeConfirmation(1, out _));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 4),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);
        SetPrivateField(coordinator, "_activeTurnId", activeTurnId);
        SetPrivateField(coordinator, "_directFeed", directFeed);

        InvokePrivate(coordinator, "DrainConfirmationTail");

        Assert.Empty(faults);
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        Assert.Equal(0, GetPrivateField<int>(coordinator, "_confirmationTailDrainActive"));
        Assert.Empty(services.Asr.FedAudio);
    }

    [Fact]
    public void ActiveNearEndCapture_ZeroLengthConfirmationTailIsNotQueued()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(4));
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        Assert.True(coordinator.TryTakeConfirmation(1, out _));

        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 4),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);

        Assert.Empty(faults);
        Assert.False(HasQueuedItem(coordinator, "_confirmationTailFrames"));
    }

    [Fact]
    public async Task ActiveNearEndCapture_StartedAsrPumpConsumesQueuedFeedAndStops()
    {
        var services = TestServices.Create();
        ActiveNearEndCoordinator? coordinator = null;
        var faults = new List<AssistantFailure>();
        coordinator = CreateDataPlaneCoordinator(
             services,
             faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-pump");

        using var cancellation = new CancellationTokenSource();
        var feedObserved = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Asr.FeedObserved = feedObserved;
        Task pump = coordinator.StartAsrFeedPump(cancellation.Token);
        FeedRecognitionAudio(coordinator, 0, [11f, 12f, 13f]);
        await feedObserved.Task.WaitAsync(TimeSpan.FromMilliseconds(250));

        cancellation.Cancel();
        await pump;

        Assert.Empty(faults);
        Assert.Equal([11f, 12f, 13f], services.Asr.FedAudio["validation-pump"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_RoutesPcmThroughRecognitionOwnerBeforeAndAfterAttachment()
    {
        var attachedServices = TestServices.Create();
        var attachedFaults = new List<AssistantFailure>();
        var attached = CreateDataPlaneCoordinator(attachedServices, attachedFaults);
        attached.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await attached.AttachAsrSessionAsync(1, "attached-session");
        FeedRecognitionAudio(attached, 0, [1f]);

        Assert.False(attached.HasPendingAsrFeedAudio());
        Assert.Equal([1f], attachedServices.Asr.FedAudio["attached-session"].SelectMany(static chunk => chunk));
        Assert.Empty(attachedFaults);

        var pendingServices = TestServices.Create();
        var pendingFaults = new List<AssistantFailure>();
        var pending = CreateDataPlaneCoordinator(pendingServices, pendingFaults);
        pending.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        FeedRecognitionAudio(pending, 0, [2f]);

        Assert.True(pending.HasPendingAsrFeedAudio());
        await pending.AttachAsrSessionAsync(1, "pending-session");
        Assert.False(pending.HasPendingAsrFeedAudio());
        Assert.Equal([2f], pendingServices.Asr.FedAudio["pending-session"].SelectMany(static chunk => chunk));
        Assert.Empty(pendingFaults);
    }

    [Fact]
    public async Task ActiveNearEndCapture_AttachedFeedCapacityBreachIsClassifiedByRecognitionOwner()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "capacity-session");

        TestRecognitionAudioPort audioPort = GetPrivateField<TestRecognitionAudioPort>(
            coordinator,
            "_recognitionAudio");
        audioPort.RejectNextFeed = true;
        FeedRecognitionAudio(coordinator, 0, [2f]);

        AssistantFailure fault = Assert.Single(faults);
        Assert.Equal("asr-feed-queue-overflow", fault.MessageKey);
    }

    [Fact]
    public void ActiveNearEndCapture_DoesNotOwnRecognitionPumpSynchronization()
    {
        FieldInfo[] fields = typeof(ActiveNearEndCoordinator)
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.DoesNotContain(fields, field => field.Name.StartsWith("_asrFeed", StringComparison.Ordinal));
        Assert.DoesNotContain(fields, field => field.FieldType == typeof(SemaphoreSlim));
        Assert.Contains(fields, field => field.FieldType == typeof(IRecognitionAudioPort));
    }

    [Fact]
    public async Task ActiveNearEndCapture_AttachedPcmDoesNotRequireANearEndPumpSignal()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-explicit-pump");

        var feedObserved = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Asr.FeedObserved = feedObserved;
        FeedRecognitionAudio(coordinator, 0, [21f, 22f, 23f]);

        await feedObserved.Task.WaitAsync(TimeSpan.FromMilliseconds(250));

        Assert.Empty(faults);
        Assert.Equal(
            [21f, 22f, 23f],
            services.Asr.FedAudio["validation-explicit-pump"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public void ActiveNearEndCapture_DrainPolicyHonorsTurnTimelineDeadlineAndAlignmentBoundaries()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int ended = 0;
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            publishSpeechEnded: (_, _) => ended++);

        SetPrivateField(coordinator, "_activeTurnId", 2L);
        SetPrivateField(coordinator, "_directFeed", true);
        SetPrivateField(coordinator, "_speechDurationMs", 0d);
        InvokePrivate(coordinator, "ProcessActiveAudioFacts", 1L, 1L, 1L, false, 1d, false);
        Assert.Equal(0d, GetPrivateField<double>(coordinator, "_speechDurationMs"));

        SetPrivateField(coordinator, "_activeTurnId", 1L);
        SetPrivateField(coordinator, "_directFeed", false);
        InvokePrivate(coordinator, "ProcessActiveAudioFacts", 1L, 1L, 1L, false, 1d, false);
        Assert.Equal(0d, GetPrivateField<double>(coordinator, "_speechDurationMs"));

        SetPrivateField(coordinator, "_directFeed", true);
        SetPrivateField(coordinator, "_endDrainPending", true);
        SetPrivateField(coordinator, "_endDrainStreamEpoch", 1L);
        SetPrivateField(coordinator, "_endDrainTargetSample", 10L);
        SetPrivateField(coordinator, "_endDrainDeadlineSample", 10L);
        SetPrivateField(coordinator, "_activeDirectFeedNextSample", 0L);
        SetPrivateField(coordinator, "_endDrainForced", false);
        InvokePrivate(coordinator, "ProcessActiveAudioFacts", 1L, 2L, 1L, false, 1d, false);
        Assert.Equal(1, ended);

        ended = 0;
        SetPrivateField(coordinator, "_activeTurnId", 1L);
        SetPrivateField(coordinator, "_directFeed", true);
        SetPrivateField(coordinator, "_endDrainPending", true);
        SetPrivateField(coordinator, "_endDrainStreamEpoch", 1L);
        SetPrivateField(coordinator, "_endDrainTargetSample", 10L);
        SetPrivateField(coordinator, "_endDrainDeadlineSample", 10L);
        SetPrivateField(coordinator, "_activeDirectFeedNextSample", 0L);
        SetPrivateField(coordinator, "_endDrainForced", false);
        InvokePrivate(coordinator, "ProcessActiveAudioFacts", 1L, 1L, 10L, false, 1d, false);
        Assert.Equal(1, ended);

        SetPrivateField(coordinator, "_activeTurnId", 1L);
        SetPrivateField(coordinator, "_directFeed", true);
        SetPrivateField(coordinator, "_endDrainPending", false);
        SetPrivateField(coordinator, "_endDrainForced", false);
        InvokePrivate(coordinator, "BeginEndDrainLocked", 1L, 0L, 1L, false);
        Assert.True(GetPrivateField<bool>(coordinator, "_endDrainForced"));

        SetPrivateField(coordinator, "_endDrainPending", false);
        SetPrivateField(coordinator, "_endDrainForced", false);
        InvokePrivate(coordinator, "BeginEndDrainLocked", 1L, 1L, 0L, false);
        Assert.False(GetPrivateField<bool>(coordinator, "_endDrainForced"));

        SetPrivateField(coordinator, "_endDrainPending", false);
        SetPrivateField(coordinator, "_endDrainForced", false);
        InvokePrivate(coordinator, "BeginEndDrainLocked", 1L, 1L, 1L, true);
        Assert.True(GetPrivateField<bool>(coordinator, "_endDrainForced"));

        SetPrivateField(coordinator, "_endDrainPending", false);
        SetPrivateField(coordinator, "_endDrainForced", false);
        SetPrivateField(coordinator, "_activeDirectFeedNextSample", 5L);
        InvokePrivate(coordinator, "BeginEndDrainLocked", 1L, 1L, 5L, false);
        Assert.Equal("aligned", GetPrivateField<string>(coordinator, "_endDrainReason"));

        SetPrivateField(coordinator, "_activeTurnId", 1L);
        SetPrivateField(coordinator, "_directFeed", false);
        Assert.False((bool)InvokePrivate(coordinator, "IsDirectFeedActive", 1L)!);
        SetPrivateField(coordinator, "_activeTurnId", 2L);
        SetPrivateField(coordinator, "_directFeed", true);
        Assert.False((bool)InvokePrivate(coordinator, "IsDirectFeedActive", 1L)!);
        Assert.Empty(faults);
    }

    [Fact]
    public async Task ActiveNearEndCapture_DelegatesDrainAndRetainsSingleTerminalFault()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "validation-drain-one");

        FeedRecognitionAudio(coordinator, 0, [1f, 2f]);
        await coordinator.DrainAsrFeedQueueAsync();
        Assert.Equal([1f, 2f], services.Asr.FedAudio["validation-drain-one"].SelectMany(static chunk => chunk));

        InvokePrivate(coordinator, "ReportDataPlaneFault", "first-fault");
        InvokePrivate(coordinator, "ReportDataPlaneFault", "second-fault");
        AssistantFailure fault = Assert.Single(faults);
        Assert.Equal("first-fault", fault.MessageKey);
    }

    [Fact]
    public async Task ActiveNearEndCapture_ResetAndNextCaptureRejectQueuedPriorEpochAudio()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(
            services,
            faults,
            directFeedAttached: false);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "old-session");
        FeedRecognitionAudio(coordinator, 0, [1f, 2f]);
        Assert.True(coordinator.HasPendingAsrFeedAudio());

        coordinator.Reset();
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "new-session");

        Assert.Empty(faults);
        Assert.False(coordinator.HasPendingAsrFeedAudio());
        Assert.Empty(services.Asr.FedAudio);
    }

    [Fact]
    public async Task ActiveNearEndCapture_ResetClearsPendingPreAttachAudio()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        FeedRecognitionAudio(coordinator, 0, [1f, 2f]);

        coordinator.Reset();
        coordinator.ApplyConfirmedCapture(CreateConfirmationRequest(0), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "new-session");

        Assert.Empty(faults);
        Assert.Empty(services.Asr.FedAudio);
    }

    [Fact]
    public void ActiveNearEndCapture_HasNoSupersededPumpOrSessionSurface()
    {
        string[] supersededMethods =
        [
            "StartAsrFeedPump",
            "SignalAsrFeedPump",
            "DrainAsrFeedQueueAsync",
            "AttachAsrSessionAsync",
            "CanAttachAsrSessionLocked",
        ];
        MethodInfo[] methods = typeof(ActiveNearEndCoordinator)
            .GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        Assert.DoesNotContain(methods, method => supersededMethods.Contains(method.Name, StringComparer.Ordinal));
    }

    [Fact]
    public void ActiveNearEndCapture_DelegatesSessionAdmissionStateToRecognitionPort()
    {
        FieldInfo[] fields = typeof(ActiveNearEndCoordinator)
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.Contains(fields, field => field.FieldType == typeof(IRecognitionAudioPort));
        Assert.DoesNotContain(fields, field => field.Name.Contains("AsrSession", StringComparison.Ordinal));
        Assert.DoesNotContain(fields, field => field.Name.Contains("SessionAttached", StringComparison.Ordinal));
    }

    [Fact]
    public void ActiveNearEndCapture_ConfirmationCleanupMatchesOnlyTheTakenOperation()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        INearEndConfirmationStore confirmations = coordinator;
        SetPrivateField(coordinator, "_pendingConfirmation", CreateConfirmationRequest(0));

        ObserveRecognitionAudio(coordinator, 0, [1f, 2f]);

        Assert.True(confirmations.TryTakeConfirmation(1, out _));
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 2),
            playbackStartupGuardActive: false,
            latestNearEndSpeech: true);
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        confirmations.DiscardTakenConfirmation(2);
        Assert.True(confirmations.GetDiagnosticSnapshot().ConfirmationPending);
        Assert.True(HasQueuedItem(coordinator, "_confirmationTailFrames"));

        confirmations.DiscardTakenConfirmation(1);
        Assert.False(confirmations.GetDiagnosticSnapshot().ConfirmationPending);
        Assert.False(HasQueuedItem(coordinator, "_confirmationTailFrames"));
        Assert.Empty(faults);
    }

    [Fact]
    public void ActiveNearEndTrigger_SpeakerRejectionLeavesPlaybackUntouched()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantBargeInOptions
        {
            Enable = true,
            MinDurationMs = 120,
            MinRms = 0.03f,
            PlaybackMaxCandidateMs = 500,
            ActiveNearEndWindowMs = 160,
            ActiveNearEndPreRollMs = 0,
        };
        var turns = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(options.MinDurationMs, options.MinRms, 16000, 1));
        long responseTurnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryTransitionForTurn(
            responseTurnId,
            AssistantState.Speaking,
            out _,
            out _));

        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 8,
            AcceptedFrameCount: 8,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 160,
            AcceptedDurationMs: 160,
            RequiredDurationMs: 120,
            ScoreDurationMs: 160,
            PeakScoreDurationMs: 160,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.03,
            MinRequiredRms: 0.03,
            MaxRequiredRms: 0.03,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 1920,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        var speakerEvidence = new SequencedSpeakerEvidenceProvider(
            SpeakerVerificationStatus.Pending,
            SpeakerVerificationStatus.Mismatch);
        int confirmations = 0;
        var coordinator = CreateActiveNearEndCoordinator(
            options,
            new TestRecognitionAudioPort(services.Asr),
            () => turns.IsSpeakingWithoutPendingCandidate,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: _ =>
            {
                confirmations++;
                return true;
            },
            publishSpeechEnded: static (_, _) => { },
            speakerEvidence: speakerEvidence,
            reportDataPlaneFault: static _ => { });

        // Frame 1: near-end speech opens the candidate; the speaker is still Pending even though the
        // DTD already says Interrupt. Nothing is ducked, nothing is requested.
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 1, startSample: 0, sampleCount: 1920),
            playbackStartupGuardActive: false);
        ActiveNearEndDiagnosticSnapshot pending = coordinator.GetDiagnosticSnapshot();
        Assert.True(pending.CandidateActive);
        Assert.False(pending.CandidateAuthorized);
        Assert.True(pending.DtdInterruptObserved);
        Assert.Equal(1, pending.CandidateId);
        Assert.Equal(0, confirmations);

        // Frame 2: Mismatch. The candidate is dropped and its inference cancelled; the reply was
        // never touched so there is nothing to restore.
        coordinator.ObserveDuplexAudio(
            CreateDuplexFrame(sequence: 2, startSample: 1920, sampleCount: 320),
            playbackStartupGuardActive: false);

        Assert.False(coordinator.GetDiagnosticSnapshot().CandidateActive);
        Assert.Equal(0, confirmations);
        Assert.Equal(1, coordinator.CommandDeniedCount);
        Assert.Equal(0, coordinator.CommandAuthorizedCount);
        Assert.Contains(new SpeakerCandidateId(1), speakerEvidence.Cancelled);
        Assert.Equal(AssistantState.Speaking, turns.State);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Empty(services.Audio.PlaybackGainChanges);
    }

    /// <summary>
    /// Keeps the duplex callback alive with silent frames, as a real device does between utterances,
    /// until <paramref name="condition"/> holds or <paramref name="maxFrames"/> were emitted.
    /// </summary>
    private static async Task PumpSilentFramesUntilAsync(TestServices services, Func<bool> condition, int maxFrames = 60)
    {
        for (int i = 0; i < maxFrames && !condition(); i++)
        {
            // Duplex and processed frames advance together, as they do on a real device.
            EmitSilenceFrames(services, count: 1, renderRms: 0);
            await Task.Delay(5);
        }
    }

    [Fact]
    public async Task OwnerBargeIn_SpeechEndWhileSpeakerPending_PreservesCandidateForAsr()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.Speaker.Confirmation.InferenceTimeoutMs = 2000;
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);

            // Block the embedding model: the owner's whole utterance (one duplex block, one VAD
            // chunk) ends while its identity window is still being inferred. During playback the
            // near-end window is the only interruption candidate; the VAD only contributes the endpoint.
            services.Speaker.FeedEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Speaker.FeedBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            float[] candidate = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(candidate);
            await services.Speaker.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(candidate);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(800));
            await PumpSilentFramesUntilAsync(services, static () => false, maxFrames: 5);

            // Pending speaker: the reply keeps playing untouched, and no ASR session is fed yet.
            Assert.Equal(AssistantState.Speaking, assistant.State);
            Assert.Equal(0, services.Audio.ClearQueueCount);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
            lock (validationEvents)
                Assert.Empty(validationEvents);

            // The verdict arrives after the endpoint: Match for this candidate stops the reply once
            // and the preserved utterance is fed to the new turn's ASR session in full.
            services.Speaker.FeedBlock.TrySetResult();
            await PumpSilentFramesUntilAsync(services, () => services.Audio.ClearQueueCount == 1);
            await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);
            // The whole utterance (every ±0.3 sample) reaches the new turn's ASR session, together
            // with the configured pre-roll before its onset.
            await WaitUntilAsync(() =>
                services.Asr.FedAudio.TryGetValue("session-2", out List<float[]>? chunks) &&
                chunks.SelectMany(static chunk => chunk).Count(static sample => Math.Abs(Math.Abs(sample) - 0.3f) < 0.0001f) >= candidate.Length);
            await WaitUntilAsync(() =>
            {
                lock (validationEvents)
                    return validationEvents.Any(e => e.Kind == BargeInValidationKind.Accepted);
            });
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.Equal(AssistantState.Listening, assistant.State);
            lock (validationEvents)
                Assert.Equal("speaker-match", validationEvents.Last(e => e.Kind == BargeInValidationKind.Accepted).Reason);
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    [Fact]
    public async Task OtherSpeakerDuringPlayback_IsDeniedWithoutTouchingPlayback_AndOwnerStillInterrupts()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            useRealSpeaker: true,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);

            var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
                .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(assistant)!;
            var nearEnd = RuntimeGraphProbe.RequireField<ActiveNearEndCoordinator>(runtime, "_confirmations");

            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(1);
            int feedsBeforeImpostor = services.Speaker.FeedCount;
            float[] candidate = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(candidate);
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(candidate);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(800));

            // The candidate is denied on the next frames once its windows are scored; the reply
            // survives untouched and nothing reaches the ASR.
            await PumpSilentFramesUntilAsync(services, () => nearEnd.CommandDeniedCount == 1);
            Assert.Equal(1, nearEnd.CommandDeniedCount);
            Assert.Equal(0, nearEnd.CommandAuthorizedCount);
            Assert.True(services.Speaker.FeedCount > feedsBeforeImpostor);
            Assert.Equal(AssistantState.Speaking, assistant.State);
            Assert.Equal(0, services.Audio.ClearQueueCount);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
            lock (validationEvents)
                Assert.Empty(validationEvents);

            // The owner can still interrupt afterwards: a denied impostor leaves no residue beyond
            // the short cooldown.
            await Task.Delay(150);
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(0);
            services.Asr.CompleteStart("session-2");
            float[] owner = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(owner);
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(owner);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(800));
            await PumpSilentFramesUntilAsync(services, () => services.Audio.ClearQueueCount == 1);
            await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);
            await WaitUntilAsync(() =>
            {
                lock (validationEvents)
                    return validationEvents.Any(e => e.Kind == BargeInValidationKind.Accepted);
            });
            Assert.Equal(1, nearEnd.CommandAuthorizedCount);
            lock (validationEvents)
                Assert.Equal("speaker-match", validationEvents.Last(e => e.Kind == BargeInValidationKind.Accepted).Reason);
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    [Fact]
    public async Task ActiveNearEndTrigger_WithDnrEnabled_UsesPostDnrRecognitionAudio()
    {
        var services = TestServices.Create();
        services.Audio.EnableDnr = true;
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                // 预缓冲设 0: 首个 TTS 块立即入队并跑完 MarkPlaybackStarted→ClearAudioReservoirs,
                // 使下方 AcceptedSamples>0 屏障在 block 期间即可达 (阈值>0 时 4 样本块会被 PrerollGate
                // 憋住不入队, 屏障死等到超时)。与 PlaybackStartupGuard_Skips... 测试同一手法。
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        // 屏障必须等到"首个 TTS 块入队"完成, 而非仅仅 State==Speaking (后者在首个 LLM 可见
        // 增量即置位, 早于播放泵入队)。入队 (AcceptedSamples>0) 之后, 泵线程同一路径紧接着
        // NotifyPlaybackStarted → MarkPlaybackStarted → ClearAudioReservoirs 清空识别蓄水池;
        // 若在此之前抢先 Emit 预滚(0.23), 会被随后的清空抹掉, 预滚回退到检测回声 → 偶发失败。
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        services.Audio.EmitProcessedCapture(ConstantSamples(0.23f, 320 * 8));
        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(
            services,
            count: 8,
            renderRms: 0.08,
            captureRms: 0.09,
            processedRms: 0.23);
        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Any(e => e.Kind == BargeInValidationKind.Heard);
        });
        services.Audio.EmitProcessedCapture(ConstantSamples(0.24f, 320 * 2));
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);

        string reply;
        try
        {
            reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        catch (TimeoutException exception)
        {
            throw CreateRuntimeTimeoutException(
                nameof(ActiveNearEndTrigger_WithDnrEnabled_UsesPostDnrRecognitionAudio),
                assistant,
                services,
                exception);
        }

        Assert.Equal("好的。", reply);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        float[] fedSamples = services.Asr.FedAudio["session-2"].SelectMany(chunk => chunk).ToArray();
        Assert.Contains(fedSamples, sample => Math.Abs(sample - 0.23f) < 0.0001f);
        Assert.Contains(fedSamples, sample => Math.Abs(sample - 0.24f) < 0.0001f);
        Assert.DoesNotContain(fedSamples, sample => Math.Abs(sample - 0.09f) < 0.0001f);
    }

    [Fact]
    public async Task RenderAwareDecision_FeedsOnlyFinalMainCaptureToAsr()
    {
        // DTD authorizes from raw capture plus render while ASR must receive only the
        // final primary capture chain represented by ProcessedCaptureFrame.
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
                options.BargeIn.ActiveNearEndCooldownMs = 100;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Any(e => e.Kind == BargeInValidationKind.Heard);
        });
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        float[] fedSamples = services.Asr.FedAudio["session-2"].SelectMany(chunk => chunk).ToArray();
        // The ASR path receives the final main-chain samples, never a detector-side signal.
        Assert.Contains(fedSamples, sample => Math.Abs(sample - 0.12f) < 0.0001f);
        Assert.DoesNotContain(fedSamples, sample => Math.Abs(sample - 0.24f) < 0.0001f);
        lock (validationEvents)
            Assert.All(validationEvents, e => Assert.Equal(BargeInTriggerSource.NearEndWindow, e.Source));
    }

    [Fact]
    public async Task RenderAware_FirstPlaybackInterruptsWithoutPriorCalibration()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var validationEvents = new List<BargeInValidationEventArgs>();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;
        await Task.Delay(550);

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);

        await WaitUntilAsync(() =>
        {
            lock (validationEvents)
                return validationEvents.Any(e => e.Kind == BargeInValidationKind.Heard);
        });
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);
        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        lock (validationEvents)
        {
            Assert.Contains(validationEvents, e =>
                e.Kind == BargeInValidationKind.Heard &&
                e.Source == BargeInTriggerSource.NearEndWindow);
            Assert.Contains(validationEvents, e =>
                e.Kind == BargeInValidationKind.Accepted &&
                e.Source == BargeInTriggerSource.NearEndWindow);
        }
    }

    [Fact]
    public async Task RenderAwareDtd_InterruptStopsPlaybackBeforeAsrResult()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "");
        TaskCompletionSource finishBlock = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var validationEvents = new List<BargeInValidationKind>();
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);

        // KWS disabled: the DTD Interrupt is the authorization. The stop is issued at confirmation,
        // before the endpoint and before any ASR text; the blocked finish cannot delay it.
        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(1, services.Audio.ClearQueueCount);
        await WaitUntilAsync(() => services.Audio.PlaybackGain == 1f);
        Assert.Equal(new[] { 0.15f, 1f }, services.Audio.PlaybackGainChanges);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(AssistantState.Listening, assistant.State);
        Assert.False(finishBlock.Task.IsCompleted);
        lock (validationEvents)
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents);

        EmitSilenceFrames(services, count: 12, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        finishBlock.SetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Single(services.Llm.Prompts);
    }

    [Fact]
    public async Task RenderAwareDtd_InterruptsInsideEchoEnvelope()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "");
        TaskCompletionSource finishBlock = services.Asr.BlockFinish("session-2");
        var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = firstTtsBlock;
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.MinRms = 0.01f;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var validationEvents = new List<BargeInValidationKind>();
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e.Kind);
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 120, renderRms: 0.08, echoRms: 0.02);

        // The learned echo envelope is negative evidence rather than a hard veto. A strong,
        // time-aligned near-end probability can therefore win the fused decision.
        EmitNearEndFrames(services, count: 8, renderRms: 0.08, captureRms: 0.015, emitMainAudio: true);
        EmitSilenceFrames(services, count: 12, renderRms: 0.08);

        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(1, services.Audio.ClearQueueCount);
        await WaitUntilAsync(() => services.Audio.PlaybackGain == 1f);
        Assert.Equal(new[] { 0.15f, 1f }, services.Audio.PlaybackGainChanges);
        Assert.Equal(1f, services.Audio.PlaybackGain);
        Assert.Equal(AssistantState.Listening, assistant.State);
        lock (validationEvents)
            Assert.Equal([BargeInValidationKind.Heard, BargeInValidationKind.Accepted], validationEvents);

        finishBlock.SetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        Assert.Equal(1, services.Audio.ClearQueueCount);
    }

    [Fact]
    public async Task RenderAwareDtd_PreRollUsesDecisionTimelineWithoutClippingPrefix()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "不好听，换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.MinDurationMs = 300;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.ActiveNearEndWindowMs = 360;
                options.BargeIn.ActiveNearEndPreRollMs = 360;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        // 判定窗口之前的 600ms 弱句首尚未形成近端证据，且主 VAD/DTD
        // 都没有提供可靠起点。main-timeline 回退必须使用现有候选与识别音频预算
        // 保留完整句首，不能只取判定窗口之前固定 360ms。
        const int preRollFrameSamples = 320;
        const int preRollFrames = 30;
        for (int i = 0; i < preRollFrames; i++)
        {
            float[] render = AlternatingSamples(0.08, preRollFrameSamples);
            float[] echo = Scale(render, 0.02 / 0.08);
            services.Audio.EmitFrame(
                render,
                echo,
                echo,
                echo,
                processedCapture: ConstantSamples(0.23f, preRollFrameSamples));
        }

        // 完整复现日志的判定帧构成：256 samples/16ms，4 帧未通过门槛，
        // 随后 19 帧通过，累计 304ms 时确认；360ms 窗口因整帧聚合实际为
        // 23*16=368ms。再让主识别线程领先检测线程 4 帧，模拟真实的两个
        // LongRunning 消费线程调度。旧 SnapshotLast(360ms) 会从 main tail
        // 截取并同时丢掉窗口前句首及部分判定窗口。
        const int frameSamples = 256;
        const int rejectedPrefixFrames = 4;
        const int acceptedFrames = 19;
        const int mainLeadFrames = 4;
        var ranges = new List<CaptureRange>();
        for (int i = 0; i < rejectedPrefixFrames + acceptedFrames + mainLeadFrames; i++)
            ranges.Add(services.Audio.ReserveCaptureRange(frameSamples));

        for (int i = 0; i < ranges.Count; i++)
        {
            float value = i < rejectedPrefixFrames
                ? 0.27f
                : i < rejectedPrefixFrames + acceptedFrames
                    ? 0.31f
                    : 0.41f;
            CaptureRange range = ranges[i];
            services.Audio.EmitProcessedCapture(
                in range,
                ConstantSamples(value, frameSamples));
        }

        for (int i = 0; i < rejectedPrefixFrames; i++)
        {
            CaptureRange range = ranges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] echo = Scale(render, 0.02 / 0.08);
            services.Audio.EmitDuplex(
                in range,
                render,
                echo,
                echo,
                echo);
        }

        for (int i = rejectedPrefixFrames; i < rejectedPrefixFrames + acceptedFrames; i++)
        {
            CaptureRange range = ranges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] capture = ConstantSamples(0.12f, frameSamples);
            services.Audio.EmitDuplex(
                in range,
                render,
                capture,
                capture,
                capture);
        }

        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));
        for (int i = rejectedPrefixFrames + acceptedFrames; i < ranges.Count; i++)
        {
            CaptureRange range = ranges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] capture = ConstantSamples(0.12f, frameSamples);
            services.Audio.EmitDuplex(
                in range,
                render,
                capture,
                capture,
                capture);
        }
        EmitSilenceFrames(services, count: 24, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FedAudio.ContainsKey("session-2"));
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);

        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        int preRollSamples = preRollFrameSamples * preRollFrames;
        int rejectedSamples = frameSamples * rejectedPrefixFrames;
        int acceptedSamples = frameSamples * acceptedFrames;
        int leadSamples = frameSamples * mainLeadFrames;
        int preRollOffset = Array.FindIndex(
            fed,
            sample => Math.Abs(sample - 0.23f) < 0.0001f);
        Assert.True(preRollOffset >= 0);
        Assert.True(fed.Length >= preRollOffset + preRollSamples + rejectedSamples + acceptedSamples + leadSamples);
        Assert.All(fed.AsSpan(preRollOffset, preRollSamples).ToArray(), sample => Assert.Equal(0.23f, sample));
        Assert.All(fed.AsSpan(preRollOffset + preRollSamples, rejectedSamples).ToArray(), sample => Assert.Equal(0.27f, sample));
        Assert.All(fed.AsSpan(preRollOffset + preRollSamples + rejectedSamples, acceptedSamples).ToArray(), sample => Assert.Equal(0.31f, sample));
        Assert.All(fed.AsSpan(preRollOffset + preRollSamples + rejectedSamples + acceptedSamples, leadSamples).ToArray(), sample => Assert.Equal(0.41f, sample));
        Assert.Equal(preRollSamples, fed.Count(sample => Math.Abs(sample - 0.23f) < 0.0001f));
        Assert.Equal(rejectedSamples, fed.Count(sample => Math.Abs(sample - 0.27f) < 0.0001f));
        Assert.Equal(acceptedSamples, fed.Count(sample => Math.Abs(sample - 0.31f) < 0.0001f));
        Assert.Equal(leadSamples, fed.Count(sample => Math.Abs(sample - 0.41f) < 0.0001f));
    }

    [Fact]
    public async Task RenderAwareDtd_PreRollStartsFromRetainedHistoryWhileMainAudioCatchesUp()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "不好听，换一个");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.MinDurationMs = 300;
                options.BargeIn.MinRms = 0.08f;
                options.BargeIn.ActiveNearEndWindowMs = 360;
                options.BargeIn.ActiveNearEndPreRollMs = 360;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            });
        var heard = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Heard)
                heard.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        const int frameSamples = 256;
        const int preRollFrames = 23;
        const int rejectedPrefixFrames = 4;
        const int acceptedFrames = 19;
        var preRollRanges = new List<CaptureRange>();
        for (int i = 0; i < preRollFrames; i++)
            preRollRanges.Add(services.Audio.ReserveCaptureRange(frameSamples));
        var decisionRanges = new List<CaptureRange>();
        for (int i = 0; i < rejectedPrefixFrames + acceptedFrames; i++)
            decisionRanges.Add(services.Audio.ReserveCaptureRange(frameSamples));

        for (int i = 0; i < preRollRanges.Count; i++)
        {
            CaptureRange range = preRollRanges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] echo = Scale(render, 0.02 / 0.08);
            services.Audio.EmitDuplex(in range, render, echo, echo, echo);
        }
        for (int i = 0; i < rejectedPrefixFrames; i++)
        {
            CaptureRange range = decisionRanges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] echo = Scale(render, 0.02 / 0.08);
            services.Audio.EmitDuplex(in range, render, echo, echo, echo);
        }
        for (int i = rejectedPrefixFrames; i < decisionRanges.Count; i++)
        {
            CaptureRange range = decisionRanges[i];
            float[] render = AlternatingSamples(0.08, frameSamples);
            float[] capture = ConstantSamples(0.12f, frameSamples);
            services.Audio.EmitDuplex(in range, render, capture, capture, capture);
        }

        Assert.False(heard.Task.IsCompleted);

        for (int i = 0; i < preRollRanges.Count; i++)
        {
            CaptureRange range = preRollRanges[i];
            services.Audio.EmitProcessedCapture(
                in range,
                ConstantSamples(0.23f, frameSamples));
        }
        for (int i = 0; i < decisionRanges.Count; i++)
        {
            CaptureRange range = decisionRanges[i];
            float value = i < rejectedPrefixFrames ? 0.27f : 0.31f;
            services.Audio.EmitProcessedCapture(
                in range,
                ConstantSamples(value, frameSamples));
        }

        // The last main-chain frame reaches the fixed acoustic decision end. Confirmation
        // must retry from this callback; no later duplex detector frame should be required.
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        CaptureRange retryRange = services.Audio.ReserveCaptureRange(frameSamples);
        float[] retryRender = AlternatingSamples(0.08, frameSamples);
        float[] retryCapture = ConstantSamples(0.12f, frameSamples);
        services.Audio.EmitDuplex(
            in retryRange,
            retryRender,
            retryCapture,
            retryCapture,
            retryCapture);
        services.Audio.EmitProcessedCapture(
            in retryRange,
            ConstantSamples(0.41f, frameSamples));
        EmitSilenceFrames(services, count: 24, renderRms: 0.08);
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);

        float[] fed = services.Asr.FedAudio["session-2"].SelectMany(static chunk => chunk).ToArray();
        int preRollSamples = preRollFrames * frameSamples;
        int rejectedSamples = rejectedPrefixFrames * frameSamples;
        int acceptedSamples = acceptedFrames * frameSamples;
        int preRollOffset = Array.FindIndex(
            fed,
            sample => Math.Abs(sample - 0.23f) < 0.0001f);
        Assert.True(preRollOffset >= 0);
        Assert.All(fed.AsSpan(preRollOffset, preRollSamples).ToArray(), sample => Assert.Equal(0.23f, sample));
        Assert.All(fed.AsSpan(preRollOffset + preRollSamples, rejectedSamples).ToArray(), sample => Assert.Equal(0.27f, sample));
        Assert.All(fed.AsSpan(preRollOffset + preRollSamples + rejectedSamples, acceptedSamples).ToArray(), sample => Assert.Equal(0.31f, sample));
        Assert.Equal(preRollSamples, fed.Count(sample => Math.Abs(sample - 0.23f) < 0.0001f));
        Assert.Equal(rejectedSamples, fed.Count(sample => Math.Abs(sample - 0.27f) < 0.0001f));
        Assert.Equal(acceptedSamples, fed.Count(sample => Math.Abs(sample - 0.31f) < 0.0001f));
        Assert.Equal(frameSamples, fed.Count(sample => Math.Abs(sample - 0.41f) < 0.0001f));
    }

    [Fact]
    public async Task ActiveNearEndTrigger_StablePlaybackEcho_DoesNotStartValidation()
    {
        var services = TestServices.Create();
        var benchmarkProbe = new VoiceAssistantBenchmarkProbe();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 160;
                options.BargeIn.ActiveNearEndPreRollMs = 160;
                options.BargeIn.ActiveNearEndEndSilenceMs = 40;
            },
            benchmarkProbe: benchmarkProbe);
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        EmitPlaybackEchoFrames(services, count: 40, renderRms: 0.08, echoRms: 0.10);
        await Task.Delay(50);

        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
        lock (validationEvents)
        {
            VoiceAssistantBenchmarkSnapshot snapshot = benchmarkProbe.GetSnapshot();
            Assert.True(
                validationEvents.Count == 0,
                $"Stable echo triggered: probability={snapshot.PeakNearEndProbability:F3}/" +
                $"{snapshot.PeakSmoothedNearEndProbability:F3}, decision={snapshot.Decision}.");
        }
        Assert.Equal(AssistantState.Speaking, assistant.State);
    }

    [Fact]
    public async Task ActiveNearEndTrigger_PlaybackTransportRenderGap_DoesNotUseEnergyOnly()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "");
        services.Tts.BlockAfterFirstChunk =
            new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var benchmarkProbe = new VoiceAssistantBenchmarkProbe();
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options =>
            {
                options.PlaybackPrerollMs = 0;
                options.BargeIn.MinDurationMs = 120;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndWindowMs = 240;
                options.BargeIn.ActiveNearEndPreRollMs = 240;
            },
            benchmarkProbe: benchmarkProbe);
        var validationEvents = new List<BargeInValidationEventArgs>();
        assistant.BargeInValidation += (_, e) =>
        {
            lock (validationEvents)
                validationEvents.Add(e);
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() =>
            assistant.State == AssistantState.Speaking &&
            services.Audio.AcceptedSamples > 0);
        services.Audio.HoldPlayback();
        services.Tts.BlockAfterFirstChunk = null;

        EmitPlaybackEchoFrames(services, count: 20, renderRms: 0.08, echoRms: 0.02);

        // The output transport remains active while the delay-aligned render reference
        // has a transient gap. A main-VAD speech edge models the real failure mode where
        // TTS leakage freezes noise learning; the render-aware DTD must prevent
        // the gap frames from authorizing a candidate before the reference resumes.
        services.Vad.EmitSpeechStarted();
        float[] silence = new float[320];
        float[] capture = ConstantSamples(0.12f, 320);
        for (int frame = 0; frame < 58; frame++)
        {
            services.Audio.EmitDuplexOnly(
                silence,
                capture,
                capture,
                capture,
                renderStatus: RenderTimelineStatus.Gap);
        }
        EmitNearEndFrames(
            services,
            count: 1,
            renderRms: 0.08,
            captureRms: 0.12);
        services.Vad.EmitSpeaking(ConstantSamples(0.12f, 4800));

        VoiceAssistantBenchmarkSnapshot snapshot = benchmarkProbe.GetSnapshot();
        Assert.Equal(BargeInAction.Continue, snapshot.Decision.Action);
        Assert.True(
            snapshot.IsSpeakingWithoutPendingCandidate,
            $"Expected speaking-ready state, got {snapshot.AssistantState}.");
        Assert.True(
            !snapshot.Coordinator.CandidateActive,
            $"Unexpected candidate: decision={snapshot.Decision}; coordinator={snapshot.Coordinator}.");
        Assert.True(snapshot.PlaybackActive);
        Assert.True(snapshot.RenderActive);
        Assert.True(snapshot.Detector.LatestNearEndSpeech);
        lock (validationEvents)
            Assert.Empty(validationEvents);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        Assert.Equal(AssistantState.Speaking, assistant.State);
    }

    [Fact]
    public async Task BargeInAfterResponseGenerated_KeepsCompletedExchangeInHistory()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        // 让首轮回复停在"播放排空"阶段, 模拟回复文本已完整产出但 TTS 仍在播放。
        services.Audio.HoldPlayback();
        var drainPolled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.DrainPolled = drainPolled;

        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        // 首轮: LLM 完整产出 "好的。", 随后卡在播放排空阶段。
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        // 排空开始 => 回复文本此刻已完整生成并落入 history。
        await drainPolled.Task.WaitAsync(TimeSpan.FromSeconds(2));

        // 在播放阶段开口打断首轮。
        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        // 第二轮正常完成。
        string reply;
        try
        {
            reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        catch (TimeoutException exception)
        {
            throw CreateRuntimeTimeoutException(
                nameof(BargeInAfterResponseGenerated_KeepsCompletedExchangeInHistory),
                assistant,
                services,
                exception);
        }

        Assert.Equal("好的。", reply);
        Assert.Equal(AssistantState.Idle, assistant.State);

        // 第二轮的 LLM prompt 必须包含首轮"已完整生成但播放被打断"的完整问答,
        // 证明该轮被记入了 history (旧实现会把它整轮丢弃)。
        ChatMessage[] lastPrompt = services.Llm.Prompts[^1];
        Assert.Equal(2, lastPrompt.Count(m => m.Role == ChatRole.User));
        Assert.Contains(lastPrompt, m => m.Role == ChatRole.Assistant && MessageText(m) == "好的。");
    }

    [Fact]
    public async Task BargeInAfterLlmFinalWhileTtsStillRunning_KeepsCompletedExchangeInHistory()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Llm.EmitSeparateFinalChunk = true;
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        var generated = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantResponseGenerated += (_, e) => generated.TrySetResult(e.Text);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => services.Tts.EmittedSamples > 0);
        string generatedText = await generated.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", generatedText);

        // 首轮 TTS 仍未完成; 此时打断也应把"已完整生成"的首轮记为正常回话。
        services.Tts.BlockAfterFirstChunk = null;
        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        ChatMessage[] lastPrompt = services.Llm.Prompts[^1];
        Assert.Equal(2, lastPrompt.Count(m => m.Role == ChatRole.User));
        Assert.Contains(lastPrompt, m => m.Role == ChatRole.Assistant && MessageText(m) == "好的。");
    }

    [Fact]
    public async Task StopAfterBargeIn_WaitsForInterruptedPipelineThatOutlivesCancellation()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        var interruptedPipelineBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockNextStreamAfterFirstChunkIgnoringCancellation = interruptedPipelineBlock;

        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);

        await assistant.StartAsync();

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        await WaitUntilAsync(() => services.Tts.IgnoringCancellationBlockEntries == 1);

        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        // The second playback request remains behind the interrupted provider until that provider
        // actually returns. Wait for semantic interruption, not for a reply that this completion
        // barrier intentionally cannot start while the fake provider ignores cancellation.
        await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);

        Task stopTask = assistant.StopAsync();
        Task completedEarly = await Task.WhenAny(stopTask, Task.Delay(150));
        Assert.NotSame(stopTask, completedEarly);

        interruptedPipelineBlock.SetResult();
        await stopTask.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    private static string MessageText(ChatMessage message) =>
        string.Concat(message.Content.OfType<TextContent>().Select(c => c.Text));

    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        while (!condition())
        {
            await Task.Delay(10, cts.Token);
        }
    }

    private static float[] BargeInVoiceSamples() => ConstantSamples(0.4f, 8000);

    private static float[] PlaybackResidualSamples() => ConstantSamples(0.05f, 16000);

    private static void EmitVadAudioWithDuplex(TestServices services, float[] audio)
    {
        services.Audio.Emit(audio);
        services.Vad.AfterNextSpeechEnded = () =>
            EmitSilenceFrames(services, count: 30, renderRms: 0);
        services.Vad.EmitSpeaking(audio);
    }

    private static void EmitCorrelatedPlaybackInput(TestServices services, float[] audio)
        => services.Audio.EmitFrame(audio, audio, audio, audio);

    private static DuplexAudioFrame CreateDuplexFrame(
        long sequence,
        long startSample,
        int sampleCount)
    {
        float[] audio = ConstantSamples(0.1f, sampleCount);
        return new DuplexAudioFrame(
            SequenceNumber: sequence,
            TimestampTicks: sequence,
            SampleRate: 16000,
            Channels: 1,
            FrameCount: (uint)sampleCount,
            CaptureProcessed: true,
            RenderReference: audio,
            CaptureRaw: audio,
            CaptureBeforeAec: audio,
            CaptureAfterAec: audio,
            StreamEpoch: 1,
            StartSample: startSample,
            HasDiscontinuity: false);
    }

    private static ActiveNearEndCoordinator CreateActiveNearEndCoordinator(
        VoiceAssistantBargeInOptions options,
        IRecognitionAudioPort recognitionAudio,
        Func<bool> isSpeakingWithoutPendingCandidate,
        Func<TimeSpan, int, float, BargeInDecision> getRecentDecision,
        AssistantAudioFormat audioFormat,
        Microsoft.Extensions.Logging.ILogger logger,
        Func<bool> isStopping,
        Func<bool> isPlaybackActive,
        Func<long, bool> requestConfirmation,
        Action<long, TimeSpan> publishSpeechEnded,
        ISpeakerEvidenceProvider? speakerEvidence = null,
        TimeProvider? timeProvider = null,
        Action<AssistantFailure>? reportDataPlaneFault = null,
        Func<long>? currentTurnId = null)
    {
        var ports = new TestActiveNearEndPorts(
            isSpeakingWithoutPendingCandidate,
            getRecentDecision,
            isStopping,
            isPlaybackActive,
            requestConfirmation,
            publishSpeechEnded,
            reportDataPlaneFault ?? (static _ => { }),
            currentTurnId ?? (static () => 1));
        return new ActiveNearEndCoordinator(
            options,
            recognitionAudio,
            ports,
            ports,
            audioFormat,
            logger,
            ports,
            ports,
            ports,
            speakerEvidence,
            timeProvider);
    }

    private static ActiveNearEndCoordinator CreateDataPlaneCoordinator(
        TestServices services,
        List<AssistantFailure> faults,
        Action<long, TimeSpan>? publishSpeechEnded = null,
        bool directFeedAttached = true)
        => CreateActiveNearEndCoordinator(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                MinDurationMs = 1,
                ActiveNearEndWindowMs = 360,
                ActiveNearEndEndSilenceMs = 1,
                ActiveNearEndAudioDrainTimeoutMs = 100,
            },
            new TestRecognitionAudioPort(services.Asr, directFeedAttached),
            isSpeakingWithoutPendingCandidate: static () => true,
            getRecentDecision: (_, _, _) => new BargeInDecision(
                BargeInAction.Continue,
                FrameCount: 0,
                AcceptedFrameCount: 0,
                RejectedFrameCount: 0,
                EchoLikeFrameCount: 0,
                TotalDurationMs: 0,
                AcceptedDurationMs: 0,
                RequiredDurationMs: 0,
                ScoreDurationMs: 0,
                PeakScoreDurationMs: 0,
                LastFrameRms: 0,
                LastRequiredRms: 0,
                MinRequiredRms: 0,
                MaxRequiredRms: 0,
                Reason: "test",
                StreamEpoch: 1,
                StartSample: 0,
                EndSampleExclusive: 0,
                LatestFrameAccepted: false),
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, voice: null, speed: null)),
            Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance,
            isStopping: static () => false,
            isPlaybackActive: static () => true,
            requestConfirmation: static _ => true,
             publishSpeechEnded: publishSpeechEnded ?? (static (_, _) => { }),
            reportDataPlaneFault: faults.Add);

    private sealed class TestActiveNearEndPorts(
        Func<bool> isSpeakingWithoutPendingCandidate,
        Func<TimeSpan, int, float, BargeInDecision> getRecentDecision,
        Func<bool> isStopping,
        Func<bool> isPlaybackActive,
        Func<long, bool> requestConfirmation,
        Action<long, TimeSpan> publishSpeechEnded,
        Action<AssistantFailure> reportDataPlaneFault,
        Func<long> currentTurnId) :
        IActiveNearEndConversationView,
        IActiveNearEndDecisionSource,
        IRuntimeLivenessView,
        IActiveNearEndPlaybackPort,
        IActiveNearEndSignalSink
    {
        public bool IsSpeakingWithoutPendingCandidate => isSpeakingWithoutPendingCandidate();

        public long CurrentTurnId => currentTurnId();

        public bool IsStopping => isStopping();

        public bool IsPlaybackActive => isPlaybackActive();

        public long? ActiveOwnerTurnId => isPlaybackActive() ? currentTurnId() : null;

        public bool TryDuckPlayback(long ownerTurnId, float gain, float originalGain) => false;

        public bool TryRestorePlayback(long ownerTurnId) => false;

        public BargeInDecision GetRecentDecision(TimeSpan window, int minDurationMs, float minRms)
            => getRecentDecision(window, minDurationMs, minRms);

        public bool TryEnqueueConfirmation(long operationId) => requestConfirmation(operationId);

        public long NextOperationId() => 1;

        public void EnqueueSpeechEnded(long turnId, TimeSpan duration)
            => publishSpeechEnded(turnId, duration);

        public void ReportDataPlaneFault(AssistantFailure failure) => reportDataPlaneFault(failure);
    }

        private sealed class TestRecognitionAudioPort(
        FakeAsrService asr,
        bool directFeedAttached = true) : IRecognitionAudioPort, IRecognitionAudioTestDriver
    {
        private readonly object _lock = new();
        private readonly Dictionary<long, Queue<float[]>> _pending = [];
        private readonly Dictionary<long, string> _sessions = [];

        public bool RejectNextFeed { get; set; }

        public bool TryFeedAudio(long turnId, ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty)
                return true;
            string? directSession = null;
            lock (_lock)
            {
                if (RejectNextFeed)
                {
                    RejectNextFeed = false;
                    return false;
                }
                if (directFeedAttached &&
                    _sessions.TryGetValue(turnId, out string? attachedSession))
                {
                    directSession = attachedSession;
                }
                else
                {
                    if (!_pending.TryGetValue(turnId, out Queue<float[]>? queue))
                    {
                        queue = new Queue<float[]>();
                        _pending.Add(turnId, queue);
                    }
                    if (queue.Count >= 256)
                    {
                        return false;
                    }
                    queue.Enqueue(audio.ToArray());
                }
            }
            if (directSession is not null)
                asr.FeedAudio(directSession, audio);
            return true;
        }

        public bool HasAttachedSession(long turnId)
        {
            lock (_lock)
                return _sessions.ContainsKey(turnId);
        }

        public bool TryGetProviderSession(long turnId, out long operationId, out string? sessionId)
        {
            lock (_lock)
            {
                operationId = 0;
                return _sessions.TryGetValue(turnId, out sessionId);
            }
        }

        public async Task StartAudioPump(CancellationToken cancellationToken)
        {
            try
            {
                while (true)
                {
                    await Task.Delay(1, cancellationToken);
                    await DrainAudioAsync();
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
            }
        }

        public bool HasPendingAudio
        {
            get
            {
                lock (_lock)
                    return _pending.Values.Any(static queue => queue.Count > 0);
            }
        }

        public ValueTask AttachSessionAsync(long turnId, string sessionId)
        {
            lock (_lock)
                _sessions[turnId] = sessionId;
            return DrainAudioAsync();
        }

        public void ResetAudio(long turnId)
        {
            lock (_lock)
            {
                _pending.Remove(turnId);
                _sessions.Remove(turnId);
            }
        }

        public ValueTask DrainAudioAsync()
        {
            while (DrainOne())
            {
            }
            return ValueTask.CompletedTask;
        }

        private bool DrainOne()
        {
            string? sessionId = null;
            float[]? audio = null;
            lock (_lock)
            {
                foreach ((long candidateTurnId, Queue<float[]> queue) in _pending)
                {
                    if (_sessions.TryGetValue(candidateTurnId, out string? candidateSession) &&
                        queue.Count > 0)
                    {
                        sessionId = candidateSession;
                        audio = queue.Dequeue();
                        break;
                    }
                }
            }
            if (audio is null || sessionId is null)
                return false;
            asr.FeedAudio(sessionId, audio);
            return true;
        }
    }

    private static ActiveNearEndConfirmationRequest CreateConfirmationRequest(
        long selectedEndSample)
        => new(
            OperationId: 1,
            InitialAudio: [],
            Decision: default,
            Selection: new ActiveNearEndPreRollSelection(
                "test",
                Samples: checked((int)selectedEndSample),
                StreamEpoch: 1,
                DecisionStartSample: 0,
                DecisionEndSample: selectedEndSample + 4,
                SelectedStartSample: 0,
                SelectedEndSample: selectedEndSample),
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            CandidateId: default,
            OwnerTurnId: 0,
            SpeakerResult: default);

    private static SpeakingDecision CreateConfirmedDecision()
        => new(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            ResponseToCancel: null,
            InterruptedTurnId: 0,
            OldState: AssistantState.Speaking,
            TurnId: 1,
            ConfirmedCandidateSamples: 0,
            ConfirmedCandidateRms: 0,
            ConfirmedConfirmationSamples: 0,
            ConfirmedConfirmationRms: 0,
            ConfirmationMinDurationMs: 0,
            ConfirmationMinRms: 0,
            ConfirmationInitialSuppressionMs: 0,
            ConfirmedBargeInDiagnostics: default);

    private static void FeedRecognitionAudio(
        ActiveNearEndCoordinator coordinator,
        long startSample,
        float[] samples)
    {
        MainAudioObservation observation =
            ObserveRecognitionAudio(coordinator, startSample, samples);
        coordinator.FeedFromRecognitionAudio(samples, observation);
    }

    private static MainAudioObservation ObserveRecognitionAudio(
        ActiveNearEndCoordinator coordinator,
        long startSample,
        float[] samples)
    {
        var range = new CaptureRange(
            StreamEpoch: 1,
            StartSequenceNumber: startSample + 1,
            EndSequenceNumber: startSample + 1,
            StartSample: startSample,
            EndSampleExclusive: startSample + samples.Length,
            TimestampTicks: startSample + 1);
        var frame = new ProcessedCaptureFrame(
            range,
            SampleRate: 16000,
            Channels: 1,
            Audio: samples,
            DelaySamples: 0,
            HasDiscontinuity: false);
        return coordinator.ObserveRecognitionAudio(frame);
    }

    private static bool HasQueuedItem(object instance, string fieldName)
    {
        object queue = instance.GetType()
            .GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)?
            .GetValue(instance)
            ?? throw new Xunit.Sdk.XunitException($"Queue field '{fieldName}' was not found.");
        MethodInfo tryPeek = queue.GetType().GetMethod("TryPeek")
            ?? throw new Xunit.Sdk.XunitException($"Queue field '{fieldName}' has no TryPeek method.");
        object?[] arguments = [null];
        return (bool)(tryPeek.Invoke(queue, arguments)
            ?? throw new Xunit.Sdk.XunitException($"Queue field '{fieldName}' returned no result."));
    }

    private static void SetPrivateField<T>(object instance, string name, T value)
    {
        if (TryResolveNearEndState(instance, name, out object owner, out string member))
        {
            SetStateMember(owner, member, value);
            return;
        }

        FieldInfo field = instance.GetType()
            .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException($"Field '{name}' was not found.");
        field.SetValue(instance, value);
    }

    private static T GetPrivateField<T>(object instance, string name)
    {
        if (TryResolveNearEndState(instance, name, out object owner, out string member))
        {
            return (T)(GetStateMember(owner, member)
                ?? throw new Xunit.Sdk.XunitException($"Member '{member}' was null."));
        }

        FieldInfo? field = instance.GetType()
            .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is not null)
        {
            return (T)(field.GetValue(instance)
                ?? throw new Xunit.Sdk.XunitException($"Field '{name}' was null."));
        }

        // The shell is a set of collaborators, so a field the facade used to hold now lives on the
        // object that owns it. Walk the composed graph instead of asserting a flat layout.
        return TryFindInGraph<T>(instance, name, new HashSet<object>(ReferenceEqualityComparer.Instance), 0, out T found)
            ? found
            : throw new Xunit.Sdk.XunitException($"Field '{name}' was not found in the runtime graph.");
    }

    private static bool TryFindInGraph<T>(
        object instance,
        string name,
        HashSet<object> visited,
        int depth,
        out T value)
    {
        value = default!;
        if (depth > 4 || !visited.Add(instance))
            return false;

        FieldInfo[] fields = instance.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic);
        foreach (FieldInfo field in fields)
        {
            if (field.Name != name)
                continue;
            if (field.GetValue(instance) is T typed)
            {
                value = typed;
                return true;
            }
        }

        foreach (FieldInfo field in fields)
        {
            object? child = field.GetValue(instance);
            if (child is null ||
                child is string ||
                field.FieldType.IsPrimitive ||
                field.FieldType.Namespace?.StartsWith("System", StringComparison.Ordinal) == true)
            {
                continue;
            }

            if (TryFindInGraph(child, name, visited, depth + 1, out value))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Near-end capture/endpoint/speaker state lives in dedicated owner types. These
    /// tests still address the state by its historical field name, so map that name onto its
    /// new owner instead of restoring a flattened field.
    /// </summary>
    private static readonly Dictionary<string, (string Field, string Member)> NearEndStateMap = new(StringComparer.Ordinal)
    {
        ["_activeTurnId"] = ("_capture", "ActiveTurnId"),
        ["_directFeed"] = ("_capture", "DirectFeed"),
        ["_activeDirectFeedNextSample"] = ("_capture", "DirectFeedNextSample"),
        ["_activeDirectFeedSamples"] = ("_capture", "DirectFeedSamples"),
        ["_directFeedsInFlight"] = ("_capture", "DirectFeedsInFlight"),
        ["_endDrainTargetSample"] = ("_capture", "EndDrainTargetSample"),
        ["_endDrainDeadlineSample"] = ("_capture", "EndDrainDeadlineSample"),
        ["_endDrainPending"] = ("_capture", "EndDrainPending"),
        ["_endDrainForced"] = ("_capture", "EndDrainForced"),
        ["_endDrainReason"] = ("_capture", "EndDrainReason"),
        ["_endDrainStreamEpoch"] = ("_capture", "EndDrainStreamEpoch"),
        ["_speakerCandidateActive"] = ("_speakerCandidate", "Active"),
        ["_speakerCandidateAuthorized"] = ("_speakerCandidate", "Authorized"),
        ["_speakerCandidateEndSample"] = ("_speakerCandidate", "EndSample"),
        ["_speakerCandidateMainStartSample"] = ("_speakerCandidate", "MainStartSample"),
        ["_endpointVadCandidateActive"] = ("_endpoint", "CandidateActive"),
        ["_unownedEndpointVadObserved"] = ("_endpoint", "UnownedObserved"),
        ["_unownedEndpointVadProcessed"] = ("_endpoint", "UnownedProcessed"),
    };

    private static bool TryResolveNearEndState(
        object instance,
        string name,
        out object owner,
        out string member)
    {
        owner = null!;
        member = string.Empty;
        if (!NearEndStateMap.TryGetValue(name, out (string Field, string Member) mapping))
            return false;

        FieldInfo? stateField = instance.GetType()
            .GetField(mapping.Field, BindingFlags.Instance | BindingFlags.NonPublic);
        if (stateField?.GetValue(instance) is not { } value)
            return false;

        owner = value;
        member = mapping.Member;
        return true;
    }

    private static void SetStateMember(object owner, string member, object? value)
    {
        Type type = owner.GetType();
        PropertyInfo? property = type.GetProperty(member, BindingFlags.Instance | BindingFlags.Public);
        if (property?.SetMethod is not null)
        {
            property.SetValue(owner, value);
            return;
        }

        // Read-only projections are backed by an auto-property field; tests may still seed them.
        FieldInfo backing = type.GetField(
            $"<{member}>k__BackingField",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException($"State member '{member}' is not settable.");
        backing.SetValue(owner, value);
    }

    private static object? GetStateMember(object owner, string member)
        => owner.GetType()
            .GetProperty(member, BindingFlags.Instance | BindingFlags.Public)
            ?.GetValue(owner)
            ?? throw new Xunit.Sdk.XunitException($"State member '{member}' was not found.");

    private static object? InvokePrivate(object instance, string name, params object?[] args)
    {
        MethodInfo method = instance.GetType().GetMethod(
                name,
                BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException($"Method '{name}' was not found.");
        return method.Invoke(instance, args);
    }

    private static void EmitPlaybackEchoFrames(TestServices services, int count, double renderRms, double echoRms)
    {
        for (int i = 0; i < count; i++)
        {
            float[] render = AlternatingSamples(renderRms, 320);
            float[] echo = Scale(render, echoRms / renderRms);
            services.Audio.EmitFrame(render, echo, echo, echo);
        }
    }

    /// <param name="alternatingCapture">
    /// Emit the identity-tap capture (<c>CaptureAfterAec</c>) as a zero-mean alternating signal so the
    /// real speaker module's DC gate accepts it; the processed marker fed to ASR stays constant.
    /// </param>
    private static void EmitNearEndFrames(
        TestServices services,
        int count,
        double renderRms,
        double captureRms,
        bool emitMainAudio = false,
        double? processedRms = null,
        bool alternatingCapture = false)
    {
        services.PlaybackDetector.NearEndSpeechOverride = true;
        try
        {
            for (int i = 0; i < count; i++)
            {
                float[] render = AlternatingSamples(renderRms, 320);
                float[] capture = alternatingCapture
                    ? AlternatingSamples(captureRms, 320)
                    : ConstantSamples((float)captureRms, 320);
                float[] processed = ConstantSamples((float)(processedRms ?? captureRms), 320);
                services.Audio.EmitFrame(
                    render,
                    capture,
                    capture,
                    capture,
                    processedCapture: processed);
            }
        }
        finally
        {
            services.PlaybackDetector.NearEndSpeechOverride = null;
        }
    }

    private static void EmitSilenceFrames(TestServices services, int count, double renderRms)
    {
        for (int i = 0; i < count; i++)
        {
            float[] render = AlternatingSamples(renderRms, 320);
            float[] silence = new float[320];
            services.Audio.EmitFrame(render, silence, silence, silence);
        }
    }

    private static float[] AlternatingSamples(double rms, int count)
    {
        var samples = new float[count];
        float value = (float)rms;
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (i & 1) == 0 ? value : -value;
        return samples;
    }

    private static float[] Scale(float[] source, double scale)
    {
        var samples = new float[source.Length];
        for (int i = 0; i < source.Length; i++)
            samples[i] = (float)(source[i] * scale);
        return samples;
    }

    private static float[] ConstantSamples(float value, int count)
    {
        var samples = new float[count];
        Array.Fill(samples, value);
        return samples;
    }

    private static InvalidOperationException CreateRuntimeTimeoutException(
        string scenario,
        VoiceAssistant assistant,
        TestServices services,
        TimeoutException innerException)
    {
        var runtime = (VoiceBox.Assistant.Runtime.AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        var operations = RuntimeGraphProbe.RequireField<ControlOperationRegistry>(runtime, "_operations");
        var tasks = RuntimeGraphProbe.RequireField<AssistantTaskSupervisor>(runtime, "_tasks");
        return new InvalidOperationException(
            $"{scenario} timed out: state={assistant.State}, asrFinish={services.Asr.FinishCount}, " +
            $"prompts={services.Llm.Prompts.Count}, tts={services.Tts.SynthesizedTexts.Count}, " +
            $"clearQueue={services.Audio.ClearQueueCount}, pendingSignals={operations.PendingSignalCount}, " +
            $"activeTasks={tasks.ActiveTaskCount}, poolThreads={ThreadPool.ThreadCount}, " +
            $"pendingWork={ThreadPool.PendingWorkItemCount}.",
            innerException);
    }

    private sealed class TestServices
    {
        public FakeDuplexService Audio { get; } = new();
        public FakeVadService Vad { get; } = new();
        public FakeAsrService Asr { get; } = new();
        public FakeLlmService Llm { get; } = new();
        public FakeTtsService Tts { get; } = new();
        public FakeWakeupService Wakeup { get; } = new();
        public FakeSpeakerService Speaker { get; } = new();
        public FakePlaybackBargeInDetector PlaybackDetector { get; } = new();

        private TestServices()
        {
            Tts.StreamCompleted = Audio.CompletePlayback;
        }

        public static TestServices Create() => new();

        /// <summary>Speaker module installed for the last KWS-enabled assistant unless the real one was requested.</summary>
        public AlwaysOwnerSpeakerModuleFactory OwnerSpeaker { get; } = new();

        public VoiceAssistant CreateAssistant(
            float? ttsSpeed = null,
            bool enableBargeIn = false,
            bool enableThinking = false,
            bool enableWakeWord = false,
            TimeSpan? continuousConversationIdleTimeout = null,
            int wakeupPreRollMs = 2500,
            int wakeupReplayLeadInMs = 120,
            Action<VoiceAssistantOptions>? configure = null,
            VoiceAssistantBenchmarkProbe? benchmarkProbe = null,
            TimeProvider? timeProvider = null,
            bool useRealSpeaker = false,
            Microsoft.Extensions.Logging.ILogger<VoiceAssistant>? logger = null)
        {
            var options = CreateOptions(
                ttsSpeed,
                enableBargeIn,
                enableThinking,
                enableWakeWord,
                continuousConversationIdleTimeout,
                wakeupPreRollMs,
                wakeupReplayLeadInMs);
            if (enableWakeWord)
                options.Speaker.Confirmation.Enable = true;
            configure?.Invoke(options);

            VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
                .UseOptions(options)
                .UseAudio(Audio)
                .UseTurnBoundary(Vad)
                .UseSpeechRecognition(Asr)
                .UseResponseGeneration(Llm)
                .UseSpeechSynthesis(Tts);
            if (enableBargeIn)
                builder.AddBargeIn(
                    options.BargeIn,
                    TestAcousticPathDelayEstimatorFactory.Instance);
            if (enableWakeWord)
            {
                builder.AddWakeup(Wakeup);
                builder.AddSpeakerVerification(Speaker);
            }

            return builder.BuildForBenchmark(
                timeProvider ?? TimeProvider.System,
                benchmarkProbe ?? new VoiceAssistantBenchmarkProbe(),
                logger,
                playbackBargeInDetectorFactory:
                    new FakePlaybackBargeInDetectorFactory(PlaybackDetector),
                speakerModuleFactory: enableWakeWord && !useRealSpeaker ? OwnerSpeaker : null);
        }

        /// <summary>Identity tap descriptor the real speaker module will see for this fake duplex service.</summary>
        public SpeakerCaptureDescriptor SpeakerDescriptor
            => SpeakerCaptureDescriptor.FromDuplexService(Audio, acousticDiagnostics: null);

        /// <summary>Writes an owner-speaker-profile-v2 for the fake speaker service on this frontend.</summary>
        public async Task<string> WriteOwnerProfileAsync(float[]? ownerAxis = null)
        {
            ownerAxis ??= FakeSpeakerService.UnitEmbedding(0);
            string path = Path.Combine(Path.GetTempPath(), $"voicebox-owner-{Guid.NewGuid():N}.json");
            float[] second = (float[])ownerAxis.Clone();
            second[1] = 0.1f;
            var bank = new SpeakerPrototypeBank(
                ownerAxis,
                [new SpeakerPrototype("p1", ownerAxis, 1f), new SpeakerPrototype("p2", second, 0.99f)]);
            var variant = new OwnerSpeakerProfileVariant(
                "test",
                SpeakerDescriptor.FrontendFingerprint,
                SpeakerDescriptor.TapId,
                DateTimeOffset.UtcNow,
                new Dictionary<SpeakerPurposeBank, SpeakerPrototypeBank>
                {
                    [SpeakerPurposeBank.CleanNearEnd] = bank,
                    [SpeakerPurposeBank.DoubleTalkNearEnd] = bank,
                },
                OwnerSpeakerProfileQuality.Empty);
            await SpeakerProfileStore.SaveAsync(
                Speaker,
                path,
                new OwnerSpeakerProfile(
                    OwnerSpeakerProfile.OwnerSpeakerId,
                    DateTimeOffset.UtcNow,
                    SpeakerProfileStore.DescribeProvider(Speaker),
                    [variant]));
            return path;
        }

        private static VoiceAssistantOptions CreateOptions(
            float? ttsSpeed,
            bool enableBargeIn,
            bool enableThinking,
            bool enableWakeWord,
            TimeSpan? continuousConversationIdleTimeout,
            int wakeupPreRollMs,
            int wakeupReplayLeadInMs)
        {
            return new VoiceAssistantOptions
            {
                MinSentenceChars = 1,
                PlaybackPrerollMs = 0,
                PlaybackTailBufferMs = 0,
                TtsSpeed = ttsSpeed,
                BargeIn =
                {
                    Enable = enableBargeIn,
                    MinDurationMs = 0,
                    MinRms = 0,
                },
                Conversation =
                {
                    EnableThinking = enableThinking,
                    ContinuousConversationIdleTimeout = continuousConversationIdleTimeout ?? TimeSpan.FromMinutes(1),
                },
                // 生命周期测试用极短的合成音频驱动流程, 关闭 Idle 监听门槛
                // (门槛行为单独在 AssistantTurnObservationStoreTests 中覆盖)。
                Listening =
                {
                    MinDurationMs = 0,
                    MinRms = 0,
                },
                Wakeup =
                {
                    Enable = enableWakeWord,
                    PreRollMs = wakeupPreRollMs,
                    ReplayLeadInMs = wakeupReplayLeadInMs,
                },
            };
        }
    }

    private sealed class FakePlaybackBargeInDetectorFactory(
        FakePlaybackBargeInDetector detector) : IPlaybackBargeInDetectorFactory
    {
        public IPlaybackBargeInDetector Create() => detector;
    }

    private sealed class FakePlaybackBargeInDetector : IPlaybackBargeInDetector
    {
        private readonly object _lock = new();
        private readonly BargeInDecisionAccumulator _decisions = new(512);
        private RenderAwareBargeInSnapshot _snapshot;
        private bool _disposed;
        private int _disposeCount;

        public Exception? PushException { get; set; }
        public int PushCount { get; private set; }
        public bool LastDecisionEnabled { get; private set; }
        public bool LastEchoLearningEnabled { get; private set; }
        public bool? NearEndSpeechOverride { get; set; }
        public int DisposeCount => Volatile.Read(ref _disposeCount);

        public DtdProcessBatch Push(
            in DuplexAudioFrame frame,
            RenderDelayHint? primaryAecDelay,
            bool decisionEnabled,
            bool echoLearningEnabled)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                PushCount++;
                LastDecisionEnabled = decisionEnabled;
                LastEchoLearningEnabled = echoLearningEnabled;
                if (PushException is { } exception)
                    throw exception;

                RenderTimelineStatus renderStatus = frame.RenderTimeline.IsValid
                    ? frame.RenderTimeline.Status
                    : frame.RenderTimeline != default || frame.RenderReference.IsEmpty
                        ? RenderTimelineStatus.Missing
                        : RenderTimelineStatus.Valid;
                if (frame.HasDiscontinuity || renderStatus != RenderTimelineStatus.Valid)
                {
                    _decisions.Reset();
                    _snapshot = _snapshot with
                    {
                        HasObservation = false,
                        LatestProbability = 0,
                        LatestSmoothedProbability = 0,
                        LatestNearEndSpeech = false,
                        RenderStatus = renderStatus,
                    };
                    return new DtdProcessBatch(
                        0,
                        0,
                        false,
                        0,
                        0,
                        false,
                        StateReset: true,
                        renderStatus);
                }

                ReadOnlySpan<float> capture = frame.CaptureBeforeAecValid.Span;
                ReadOnlySpan<float> render = frame.RenderReference.Span;
                double captureRms = ComputeRms(capture);
                double renderRms = ComputeRms(render);
                double correlation = ComputeAbsoluteCorrelation(capture, render, captureRms, renderRms);
                bool echoLike = renderRms > 0.001 && correlation >= 0.90;
                bool detectedNearEnd = captureRms >= 0.01 &&
                    (!echoLike || captureRms >= renderRms * 1.8);
                bool nearEnd = NearEndSpeechOverride ?? detectedNearEnd;
                CaptureRange range = frame.CaptureBeforeAecTimeline.SourceRange;
                double durationMs = frame.FrameCount * 1000d / frame.SampleRate;

                _snapshot = new RenderAwareBargeInSnapshot(
                    HasObservation: true,
                    LatestProbability: nearEnd ? 0.95f : 0.05f,
                    LatestSmoothedProbability: nearEnd ? 0.95f : 0.05f,
                    LatestNearEndSpeech: nearEnd,
                    AlignmentReady: true,
                    EstimatedDelayMs: primaryAecDelay?.DelayMilliseconds ?? 0,
                    DriftPpm: 0,
                    AlignmentConfidence: 1,
                    AlignmentGeneration: 1,
                    StreamEpoch: range.StreamEpoch,
                    LastStartSample: range.StartSample,
                    LastEndSampleExclusive: range.EndSampleExclusive,
                    RenderStatus: renderStatus,
                    IsFaulted: false);

                if (decisionEnabled)
                {
                    _decisions.Add(new NearEndFrameDecision(
                        frame.TimestampTicks,
                        durationMs,
                        Accepted: nearEnd,
                        EchoLike: echoLike,
                        RenderActive: renderRms > 0.001,
                        CaptureRms: captureRms,
                        RequiredRms: 0,
                        RejectDecay: 1,
                        Reason: nearEnd ? "fake-render-aware-dtd" : "fake-render-aware-dtd-rejected",
                        range.StreamEpoch,
                        range.StartSample,
                        range.EndSampleExclusive,
                        frame.HasDiscontinuity));
                }

                return new DtdProcessBatch(
                    HopCount: 1,
                    ObservationCount: 1,
                    HasObservation: true,
                    Probability: _snapshot.LatestProbability,
                    SmoothedProbability: _snapshot.LatestSmoothedProbability,
                    IsNearEndSpeech: nearEnd,
                    StateReset: false,
                    RenderStatus: renderStatus);
            }
        }

        public BargeInDecision GetRecentDecision(
            TimeSpan window,
            int minDurationMs,
            float minRms)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return _decisions.AggregateRecent(
                    window.TotalMilliseconds,
                    minDurationMs,
                    minRms,
                    BargeInDecisionParameters.Default);
            }
        }

        public RenderAwareBargeInSnapshot GetSnapshot()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return _snapshot;
            }
        }

        public void ResetDecisionState()
        {
            lock (_lock)
                _decisions.Reset();
        }

        public void ResetStream()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _decisions.Reset();
                _snapshot = default;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                Interlocked.Increment(ref _disposeCount);
                _disposed = true;
            }
        }

        private static double ComputeRms(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty)
                return 0;
            double energy = 0;
            for (int i = 0; i < samples.Length; i++)
                energy += samples[i] * samples[i];
            return Math.Sqrt(energy / samples.Length);
        }

        private static double ComputeAbsoluteCorrelation(
            ReadOnlySpan<float> capture,
            ReadOnlySpan<float> render,
            double captureRms,
            double renderRms)
        {
            if (capture.IsEmpty || capture.Length != render.Length || captureRms <= 0 || renderRms <= 0)
                return 0;
            double dot = 0;
            for (int i = 0; i < capture.Length; i++)
                dot += capture[i] * render[i];
            return Math.Abs(dot / (capture.Length * captureRms * renderRms));
        }
    }

    private sealed class FakeDuplexService : IAudioDuplexService
    {
        private volatile bool _holdPlayback;
        private volatile bool _playbackActive;
        private int _stopCount;

        public AudioOptionsBase Options { get; } = new TestAudioOptions();
        public bool IsRecording { get; private set; }
        public bool IsPlaying => _holdPlayback || _playbackActive;
        // 排空判定是 `CurrentQueueSize == 0 && !IsPlaying`, && 会短路, 因此把"已进入排空"的信号
        // 挂在必被求值的 CurrentQueueSize 上 (IsPlaying 可能因短路不被求值)。
        public int CurrentQueueSize
        {
            get
            {
                if (!_holdPlayback && !_playbackActive) return 0;
                DrainPolled?.TrySetResult();
                return 1;
            }
        }

        /// <summary>Set to keep the playback drain loop spinning (simulates audio still playing) until ClearQueue.</summary>
        public void HoldPlayback() => _holdPlayback = true;

        public void CompletePlayback() => _playbackActive = false;

        /// <summary>Completed the first time the drain loop polls the queue while held.</summary>
        public TaskCompletionSource? DrainPolled { get; set; }
        public AudioServiceState State { get; private set; } = AudioServiceState.Idle;
        public int StopCount => Volatile.Read(ref _stopCount);
        public int StartCount { get; private set; }
        public TaskCompletionSource? StartEntered { get; set; }
        public TaskCompletionSource? StartBlock { get; set; }
        public Action? OnStarted { get; set; }
        public Exception? StopException { get; set; }
        public int FailuresBeforeSuccess { get; set; }
        public int EnqueueCalls { get; private set; }
        public int AcceptedSamples { get; private set; }
        public int ClearQueueCount { get; private set; }
        public Exception? ClearQueueException { get; set; }
        public TaskCompletionSource? ClearQueueEntered { get; set; }
        public TaskCompletionSource? ClearQueueBlock { get; set; }
        public float PlaybackGain { get; private set; } = 1f;
        public List<float> PlaybackGainChanges { get; } = [];
        public bool CompleteStopAsynchronously { get; set; }
        public TaskCompletionSource? StopEntered { get; set; }
        public TaskCompletionSource? StopBlock { get; set; }
        public bool EnableAec { get; set; }
        public bool EnableDnr { get; set; }
        public bool EnableHighPassFilter { get; set; }
        public event Action<ReadOnlySpan<float>>? OnDataAvailable;
        public event EventHandler? OnPlaybackCompleted;
        public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;
        public event EventHandler<AudioErrorEventArgs>? OnError;
        public event EventHandler<AudioXrunEventArgs>? OnXrun;
        public event DuplexAudioFrameHandler? OnDuplexAudioFrame;
        public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;
        private long _captureSequence;
        private long _capturePosition;
        private long _renderEpoch = 1;
        private long _renderPosition;

        public long CapturePosition => Volatile.Read(ref _capturePosition);

        public IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType = null, bool enumAllDevices = false) => [];
        public int GetDefaultInputDevice() => 0;
        public int GetDefaultOutputDevice() => 0;

        public async Task StartAsync(int inputDeviceIndex = -1, int outputDeviceIndex = -1, CancellationToken cancellationToken = default)
        {
            StartCount++;
            StartEntered?.TrySetResult();
            if (StartBlock is not null)
                await StartBlock.Task.WaitAsync(cancellationToken);
            IsRecording = true;
            State = AudioServiceState.Active;
            OnStarted?.Invoke();
        }

        public Task StopAsync(CancellationToken cancellationToken = default)
        {
            if (StopException is { } exception)
                return Task.FromException(exception);
            if (StopBlock is not null)
                return StopAsyncBlocked();
            if (CompleteStopAsynchronously)
                return StopAsyncCore();

            Interlocked.Increment(ref _stopCount);
            IsRecording = false;
            State = AudioServiceState.Idle;
            return Task.CompletedTask;
        }

        private async Task StopAsyncBlocked()
        {
            StopEntered?.TrySetResult();
            await StopBlock!.Task.ConfigureAwait(false);
            Interlocked.Increment(ref _stopCount);
            IsRecording = false;
            State = AudioServiceState.Idle;
        }

        private async Task StopAsyncCore()
        {
            await Task.Delay(10).ConfigureAwait(false);
            Interlocked.Increment(ref _stopCount);
            IsRecording = false;
            State = AudioServiceState.Idle;
        }

        public Task StartRecordingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
            => StartAsync(deviceIndex, -1, cancellationToken);
        public Task StopRecordingAsync(CancellationToken cancellationToken = default)
            => StopAsync(cancellationToken);
        public Task StartPlayingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
            => StartAsync(-1, deviceIndex, cancellationToken);
        public Task StopPlayingAsync(CancellationToken cancellationToken = default)
            => StopAsync(cancellationToken);
        public Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public void ClearQueue()
        {
            if (ClearQueueException is { } exception)
                throw exception;
            ClearQueueCount++;
            _holdPlayback = false;
            _playbackActive = false;
            ClearQueueEntered?.TrySetResult();
            ClearQueueBlock?.Task.Wait(TimeSpan.FromSeconds(2));
        }

        public void SetPlaybackGain(float gain)
        {
            PlaybackGain = gain;
            PlaybackGainChanges.Add(gain);
        }

        public bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples)
        {
            EnqueueCalls++;
            if (State != AudioServiceState.Active)
            {
                droppedSamples = data.Length;
                return false;
            }

            if (FailuresBeforeSuccess > 0)
            {
                FailuresBeforeSuccess--;
                droppedSamples = data.Length;
                OnQueueOverflow?.Invoke(this, new AudioQueueOverflowEventArgs(droppedSamples, CurrentQueueSize, 1));
                return false;
            }

            AcceptedSamples += data.Length;
            _playbackActive = true;
            droppedSamples = 0;
            return true;
        }

        public void Dispose() { }
        public void Emit(ReadOnlySpan<float> audio) => Emit(audio.ToArray());

        public void EmitRecordData(float[] audio) => OnDataAvailable?.Invoke(audio);

        public void EmitError(Exception exception)
            => OnError?.Invoke(this, new AudioErrorEventArgs(exception, exception.Message, State));

        public void EmitXrun(AudioXrunType type, uint frameCount)
            => OnXrun?.Invoke(this, new AudioXrunEventArgs(type, frameCount));

        public void Emit(float[] audio)
        {
            float[] render = new float[audio.Length];
            EmitFrame(render, audio, audio, audio, processedCapture: audio);
        }

        public void EmitProcessedCapture(float[] processedAudio)
        {
            float[] detection = new float[processedAudio.Length];
            EmitFrame(
                new float[processedAudio.Length],
                detection,
                detection,
                detection,
                processedCapture: processedAudio);
        }
        public void EmitMetrics(double renderRms, double captureOutputRms, bool captureProcessed = true)
        {
            float[] render = CreateSignal(renderRms, 320);
            float[] capture = CreateSignal(captureOutputRms, 320);
            EmitFrame(render, capture, capture, capture, captureProcessed);
        }

        public void EmitFrame(
            float[] render,
            float[] captureRaw,
            float[] captureBeforeAec,
            float[] captureAfterAec,
            bool captureProcessed = true,
            float[]? processedCapture = null)
        {
            processedCapture ??= captureAfterAec;
            Assert.Equal(captureAfterAec.Length, processedCapture.Length);
            CaptureRange range = EmitDuplexOnly(
                render,
                captureRaw,
                captureBeforeAec,
                captureAfterAec,
                captureProcessed);
            EmitProcessedCapture(in range, processedCapture);
        }

        public CaptureRange EmitDuplexOnly(
            float[] render,
            float[] captureRaw,
            float[] captureBeforeAec,
            float[] captureAfterAec,
            bool captureProcessed = true,
            bool hasDiscontinuity = false,
            RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
            bool hasRenderDiscontinuity = false)
        {
            Assert.Equal(captureAfterAec.Length, render.Length);
            Assert.Equal(captureAfterAec.Length, captureRaw.Length);
            Assert.Equal(captureAfterAec.Length, captureBeforeAec.Length);
            CaptureRange range = ReserveCaptureRange(captureAfterAec.Length);
            EmitDuplex(
                in range,
                render,
                captureRaw,
                captureBeforeAec,
                captureAfterAec,
                captureProcessed,
                hasDiscontinuity,
                renderStatus,
                hasRenderDiscontinuity);
            return range;
        }

        public CaptureRange ReserveCaptureRange(int samples)
        {
            long start = _capturePosition;
            long sequence = ++_captureSequence;
            _capturePosition += samples;
            long timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            return new CaptureRange(1, sequence, sequence, start, _capturePosition, timestamp);
        }

        public void EmitDuplex(
            in CaptureRange range,
            float[] render,
            float[] captureRaw,
            float[] captureBeforeAec,
            float[] captureAfterAec,
            bool captureProcessed = true,
            bool hasDiscontinuity = false,
            RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
            bool hasRenderDiscontinuity = false)
        {
            Assert.Equal(range.Length, captureAfterAec.LongLength);
            Assert.Equal(captureAfterAec.Length, render.Length);
            Assert.Equal(captureAfterAec.Length, captureRaw.Length);
            Assert.Equal(captureAfterAec.Length, captureBeforeAec.Length);
            if (hasRenderDiscontinuity)
            {
                _renderEpoch++;
                _renderPosition = 0;
            }
            long renderStart = _renderPosition;
            _renderPosition = checked(_renderPosition + captureAfterAec.Length);
            var frame = new DuplexAudioFrame(
                SequenceNumber: range.StartSequenceNumber,
                TimestampTicks: range.TimestampTicks,
                SampleRate: 16000,
                Channels: 1,
                FrameCount: (uint)captureAfterAec.Length,
                CaptureProcessed: captureProcessed,
                RenderReference: render,
                CaptureRaw: captureRaw,
                CaptureBeforeAec: captureBeforeAec,
                CaptureAfterAec: captureAfterAec,
                StreamEpoch: range.StreamEpoch,
                StartSample: range.StartSample,
                HasDiscontinuity: hasDiscontinuity,
                RenderTimeline: new RenderSignalTimeline(
                    _renderEpoch,
                    renderStart,
                    _renderPosition,
                    renderStatus,
                    hasRenderDiscontinuity),
                PlaybackGain: PlaybackGain);
            OnDuplexAudioFrame?.Invoke(in frame);
        }

        public void EmitProcessedCapture(
            in CaptureRange range,
            float[] processedCapture,
            bool hasDiscontinuity = false)
        {
            Assert.Equal(range.Length, processedCapture.LongLength);
            var processedFrame = new ProcessedCaptureFrame(
                range,
                16000,
                1,
                processedCapture,
                0,
                hasDiscontinuity);
            OnProcessedCaptureFrame?.Invoke(in processedFrame);
        }

        private static float[] CreateSignal(double rms, int length)
        {
            float[] signal = new float[length];
            if (!double.IsFinite(rms) || rms <= 0)
                return signal;

            float sample = (float)rms;
            Array.Fill(signal, sample);
            return signal;
        }
    }

    private sealed class TestAudioOptions : AudioOptionsBase
    {
    }

    private sealed class NonPumpingSynchronizationContext : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state)
        {
        }
    }

    private sealed class FakeVadService : IVoiceActivityDetectService
    {
        public event EventHandler<VADEventArgs>? OnSpeechEnded;
        public event EventHandler<EventArgs>? OnSpeechStarted;
        public event EventHandler<VADEventArgs>? OnSpeaking;
        public event EventHandler<VadProbabilityEventArgs>? OnProbability;

        private readonly List<float[]> _predictedAudio = [];
        public IReadOnlyList<float[]> PredictedAudio
        {
            get { lock (_predictedAudio) return _predictedAudio.ToArray(); }
        }
        public int ResetCount { get; private set; }
        public Action? PredictAction { get; set; }
        public Action? AfterNextSpeechEnded { get; set; }

        public void Predict(ReadOnlySpan<float> audioData)
        {
            float[] copy = audioData.ToArray();
            lock (_predictedAudio)
                _predictedAudio.Add(copy);
            Action? action = PredictAction;
            PredictAction = null;
            action?.Invoke();
        }

        public void EmitSpeechStarted() => OnSpeechStarted?.Invoke(this, EventArgs.Empty);
        public void EmitSpeaking(float[] audio) => OnSpeaking?.Invoke(this, new VADEventArgs(audio, VADEventDataType.Chunk));
        public void EmitSpeechEnded(TimeSpan duration)
        {
            OnSpeechEnded?.Invoke(this, new VADEventArgs([], VADEventDataType.End, duration));
            Action? action = AfterNextSpeechEnded;
            AfterNextSpeechEnded = null;
            action?.Invoke();
        }
        public void EmitProbability(float probability)
            => OnProbability?.Invoke(this, new VadProbabilityEventArgs(
                probability,
                probability,
                probability >= 0.5f,
                1,
                TimeSpan.Zero,
                TimeSpan.FromMilliseconds(10)));
        public void Dispose() { }

        public void Reset()
        {
            ResetCount++;
        }
    }

    private sealed class SequencedSpeakerEvidenceProvider(
        params SpeakerVerificationStatus[] statuses) : ISpeakerEvidenceProvider
    {
        private int _evaluateCount;
        private long _candidates;

        public int EvaluateCount => Volatile.Read(ref _evaluateCount);

        public List<SpeakerEvidenceQuery> Queries { get; } = [];

        public List<SpeakerCandidateId> Cancelled { get; } = [];

        public SpeakerCandidateId NewCandidateId() => new(Interlocked.Increment(ref _candidates));

        public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query)
        {
            int index = Interlocked.Increment(ref _evaluateCount) - 1;
            lock (Queries)
                Queries.Add(query);
            SpeakerVerificationStatus status = statuses[Math.Min(index, statuses.Length - 1)];
            return new SpeakerVerificationResult(
                status,
                status == SpeakerVerificationStatus.Match ? 1f : 0f,
                Threshold: 0.5f,
                query.StreamEpoch,
                query.CandidateStartSample,
                query.CandidateEndSampleExclusive,
                status switch
                {
                    SpeakerVerificationStatus.Match => SpeakerVerificationReason.MajorityMatch,
                    SpeakerVerificationStatus.Mismatch => SpeakerVerificationReason.MajorityMismatch,
                    SpeakerVerificationStatus.Pending => SpeakerVerificationReason.WaitingForEvidence,
                    SpeakerVerificationStatus.InsufficientEvidence => SpeakerVerificationReason.InsufficientSpeech,
                    _ => SpeakerVerificationReason.ProviderError,
                },
                CandidateId: query.CandidateId);
        }

        public Task<SpeakerVerificationResult> WaitForDecisionAsync(
            SpeakerEvidenceQuery query,
            TimeSpan timeout,
            CancellationToken cancellationToken = default)
            => Task.FromResult(Evaluate(in query));

        public void Cancel(SpeakerCandidateId candidateId)
        {
            lock (Cancelled)
                Cancelled.Add(candidateId);
        }

        public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => TimeSpan.FromMilliseconds(250);
    }

    /// <summary>
    /// Speaker provider fake. <see cref="ExtractEmbedding"/> returns <see cref="StreamingEmbedding"/> so a
    /// test scripts the owner/impostor outcome by swapping the embedding; the identity path never uses
    /// <see cref="FeedAudio"/>.
    /// </summary>
    internal sealed class FakeSpeakerService : ISpeakerService
    {
        private float[] _streamingEmbedding = UnitEmbedding(0);
        private long _samples;
        private int _feedCount;

        public int EmbeddingSize => 192;
        public int SampleRate => 16000;
        public float Threshold => 0.5f;
        public int StreamingWindowSamples => 8000;
        public int StreamingHopSamples => 1600;
        public IReadOnlyCollection<string> EnrolledSpeakers => [];
        public int FeedCount => Volatile.Read(ref _feedCount);
        public long StreamedSamples => Volatile.Read(ref _samples);
        public int ExtractDelayMs { get; set; }
        public TaskCompletionSource? FeedEntered { get; set; }
        public TaskCompletionSource? FeedBlock { get; set; }
        public float[] StreamingEmbedding
        {
            get => Volatile.Read(ref _streamingEmbedding);
            set => Volatile.Write(ref _streamingEmbedding, value);
        }

        public event EventHandler<SpeakerEventArgs>? OnResult;

        public static float[] UnitEmbedding(int dimension)
        {
            var embedding = new float[192];
            embedding[dimension] = 1f;
            return embedding;
        }

        public float[] ExtractEmbedding(ReadOnlySpan<float> samples)
        {
            FeedEntered?.TrySetResult();
            FeedBlock?.Task.GetAwaiter().GetResult();
            Interlocked.Increment(ref _feedCount);
            if (ExtractDelayMs > 0)
                Thread.Sleep(ExtractDelayMs);
            Interlocked.Add(ref _samples, samples.Length);
            return StreamingEmbedding;
        }

        /// <summary>Owner speech for the identity tap: alternating ±0.3 so energy, DC and clipping gates pass.</summary>
        public static float[] OwnerAudio(int samples)
        {
            var audio = new float[samples];
            for (int i = 0; i < samples; i++)
                audio[i] = (i & 1) == 0 ? 0.3f : -0.3f;
            return audio;
        }

        public float CompareEmbeddings(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
        {
            double dot = 0;
            double norm1 = 0;
            double norm2 = 0;
            for (int i = 0; i < embedding1.Length; i++)
            {
                dot += embedding1[i] * embedding2[i];
                norm1 += embedding1[i] * embedding1[i];
                norm2 += embedding2[i] * embedding2[i];
            }
            return (float)(dot / Math.Sqrt(norm1 * norm2));
        }

        public float Verify(ReadOnlySpan<float> samples1, ReadOnlySpan<float> samples2) => 1f;
        public void Enroll(string speakerName, ReadOnlySpan<float> samples) { }
        public bool Unenroll(string speakerName) => false;

        public void FeedAudio(ReadOnlySpan<float> samples)
        {
            // The online identity path never streams into the provider; a call here would mean the
            // retired continuous tracker came back.
            throw new InvalidOperationException("FeedAudio must not be used by the identity path.");
        }

        public void Reset() => Interlocked.Exchange(ref _samples, 0);

        public void Dispose() { }
    }

    private sealed class FakeWakeupService : IWakeupService
    {
        public event EventHandler<WakeupEventArgs>? OnWakeup;

        public List<float[]> FedAudio { get; } = [];
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public bool IsListening { get; private set; }
        public TaskCompletionSource? StartEntered { get; set; }
        public TaskCompletionSource? StartBlock { get; set; }
        public Exception? StartException { get; set; }

        public void Start()
        {
            StartCount++;
            StartEntered?.TrySetResult();
            StartBlock?.Task.GetAwaiter().GetResult();
            if (StartException is { } exception)
                throw exception;
            IsListening = true;
        }

        public void Stop()
        {
            StopCount++;
            IsListening = false;
        }

        public void FeedAudio(ReadOnlySpan<float> audioData)
        {
            FedAudio.Add(audioData.ToArray());
        }

        public void EmitWakeup(long startTimeMs = 0, long durationMs = 0)
        {
            OnWakeup?.Invoke(this, new WakeupEventArgs("小爱同学", 90, 20, durationMs, startTimeMs));
        }

        public void Dispose() { }
    }

    private sealed class FakeAsrService : IASRService
    {
        private readonly object _lock = new();
        private readonly Queue<TaskCompletionSource<SessionInfo>> _pendingStarts = new();
        private readonly Queue<string> _readyStarts = new();
        private readonly Queue<Exception> _readyStartFailures = new();
        private readonly Dictionary<string, string> _results = new();
        private readonly Dictionary<string, TaskCompletionSource> _finishBlocks = new();

        public Dictionary<string, List<float[]>> FedAudio { get; } = new();
        public List<int> FeedThreadIds { get; } = [];
        public List<bool> FeedContexts { get; } = [];
        public int FinishCount { get; private set; }
        public TaskCompletionSource<string>? FinishEntered { get; set; }
        public TaskCompletionSource? FeedObserved { get; set; }
        public Func<bool>? FeedContext { get; set; }

        public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
        {
            lock (_lock)
            {
                if (_readyStartFailures.Count > 0)
                    return Task.FromException<SessionInfo>(_readyStartFailures.Dequeue());
                if (_readyStarts.Count > 0)
                    return Task.FromResult(new SessionInfo(_readyStarts.Dequeue()));

                var pending = new TaskCompletionSource<SessionInfo>(TaskCreationOptions.RunContinuationsAsynchronously);
                _pendingStarts.Enqueue(pending);
                return pending.Task.WaitAsync(ct);
            }
        }

        public void CompleteStart(string sessionId)
        {
            TaskCompletionSource<SessionInfo>? pending = null;
            lock (_lock)
            {
                if (_pendingStarts.Count > 0)
                    pending = _pendingStarts.Dequeue();
                else
                    _readyStarts.Enqueue(sessionId);
            }

            pending?.TrySetResult(new SessionInfo(sessionId));
        }

        public void FailStart(Exception exception)
        {
            ArgumentNullException.ThrowIfNull(exception);
            TaskCompletionSource<SessionInfo>? pending = null;
            lock (_lock)
            {
                if (_pendingStarts.Count > 0)
                    pending = _pendingStarts.Dequeue();
                else
                    _readyStartFailures.Enqueue(exception);
            }

            pending?.TrySetException(exception);
        }

        public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
        {
            FeedThreadIds.Add(Environment.CurrentManagedThreadId);
            FeedContexts.Add(FeedContext?.Invoke() ?? false);
            if (!FedAudio.TryGetValue(sessionId, out var chunks))
            {
                chunks = [];
                FedAudio.Add(sessionId, chunks);
            }
            chunks.Add(audioData.ToArray());
            FeedObserved?.TrySetResult();
        }

        public void SetResult(string sessionId, string text)
        {
            lock (_lock)
                _results[sessionId] = text;
        }

        public TaskCompletionSource BlockFinish(string sessionId)
        {
            var block = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            lock (_lock)
                _finishBlocks[sessionId] = block;
            return block;
        }

        public async Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
        {
            FinishCount++;
            FinishEntered?.TrySetResult(sessionId);
            TaskCompletionSource? block = null;
            string text;
            lock (_lock)
            {
                _finishBlocks.TryGetValue(sessionId, out block);
                text = _results.TryGetValue(sessionId, out string? configured)
                    ? configured
                    : "你好";
            }

            if (block is not null)
                await block.Task.WaitAsync(ct);

            return new AsrResult { Text = text };
        }

        public Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            return Task.CompletedTask;
        }

        public void Dispose() { }
    }

    private sealed class FakeLlmService : IStreamLLMService
    {
        private readonly object _lock = new();
        private readonly List<ChatMessage[]> _prompts = new();
        private readonly List<LlmCompleteOptions> _streamOptions = new();
        public string ResponseText { get; set; } = "好的。";
        public bool EmitSeparateFinalChunk { get; set; }
        public bool EmitThinkingWhenEnabled { get; set; }
        public Exception? StreamException { get; set; }
        public TaskCompletionSource? BlockAfterThinkingChunk { get; set; }
        public IReadOnlyList<LlmTextChunk>? StreamChunks { get; set; }

        /// <summary>Snapshot of the message list passed to each StreamAsync call, in call order.</summary>
        public IReadOnlyList<ChatMessage[]> Prompts
        {
            get { lock (_lock) return _prompts.ToArray(); }
        }

        public IReadOnlyList<LlmCompleteOptions> StreamOptions
        {
            get { lock (_lock) return _streamOptions.ToArray(); }
        }

        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            return Task.FromResult(new LlmCompletionResult { Text = ResponseText });
        }

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            lock (_lock)
            {
                _prompts.Add(messages.ToArray());
                _streamOptions.Add(new LlmCompleteOptions
                {
                    MaxTokens = options?.MaxTokens,
                    DoSample = options?.DoSample,
                    Temperature = options?.Temperature,
                    TopK = options?.TopK,
                    TopP = options?.TopP,
                    RepetitionPenalty = options?.RepetitionPenalty,
                    EnableThinking = options?.EnableThinking ?? false,
                });
            }
            await Task.Yield();
            ct.ThrowIfCancellationRequested();
            if (StreamException is { } exception)
                throw exception;

            if (EmitThinkingWhenEnabled && options?.EnableThinking == true)
            {
                yield return new LlmTextChunk { Text = "先分析一下。", Kind = LlmChunkKind.Thinking };
                if (BlockAfterThinkingChunk is { } block)
                    await block.Task.WaitAsync(ct);

                await Task.Yield();
                ct.ThrowIfCancellationRequested();
            }

            if (StreamChunks is { } configuredChunks)
            {
                foreach (LlmTextChunk chunk in configuredChunks)
                {
                    ct.ThrowIfCancellationRequested();
                    yield return chunk;
                }

                yield break;
            }

            yield return new LlmTextChunk
            {
                Text = ResponseText,
                Kind = LlmChunkKind.Text,
                IsFinal = !EmitSeparateFinalChunk,
            };

            if (EmitSeparateFinalChunk)
            {
                await Task.Yield();
                ct.ThrowIfCancellationRequested();
                yield return new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true };
            }
        }

        public void Dispose() { }
    }

    private sealed class FakeTtsService : IStreamTTSService
    {
        public int OutputSampleRate => 16000;
        public int OutputChannels => 1;
        public int EmittedSamples { get; private set; }
        public TtsSynthesisOptions? LastStreamOptions { get; private set; }
        public List<string> SynthesizedTexts { get; } = [];
        public TaskCompletionSource? BlockBeforeFirstChunk { get; set; }
        public TaskCompletionSource? BlockAfterFirstChunk { get; set; }
        public Exception? StreamException { get; set; }
        public Action? StreamCompleted { get; set; }
        private TaskCompletionSource? _blockNextStreamAfterFirstChunkIgnoringCancellation;
        public TaskCompletionSource? BlockNextStreamAfterFirstChunkIgnoringCancellation
        {
            get => _blockNextStreamAfterFirstChunkIgnoringCancellation;
            set => _blockNextStreamAfterFirstChunkIgnoringCancellation = value;
        }
        private int _ignoringCancellationBlockEntries;
        public int IgnoringCancellationBlockEntries => Volatile.Read(ref _ignoringCancellationBlockEntries);

        public IReadOnlyList<TtsVoice> ListVoices() => [new TtsVoice { Name = "test" }];

        public TtsVoice? GetDefaultVoice() => new TtsVoice { Name = "test" };

        public Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            SynthesizedTexts.Add(text);
            float[] samples = [0.1f, 0.2f, 0.3f, 0.4f];
            return Task.FromResult(new TtsResult { Samples = samples, SampleRate = OutputSampleRate, Channels = OutputChannels, Voice = "test" });
        }

        public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            string text,
            TtsSynthesisOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            try
            {
                var blockAfterFirstChunk = BlockAfterFirstChunk;
                TaskCompletionSource? ignoringCancellationBlock = Interlocked.Exchange(
                    ref _blockNextStreamAfterFirstChunkIgnoringCancellation,
                    null);
                await Task.Yield();
                ct.ThrowIfCancellationRequested();
                if (StreamException is { } exception)
                    throw exception;
                LastStreamOptions = options;
                SynthesizedTexts.Add(text);

                var blockBeforeFirstChunk = BlockBeforeFirstChunk;
                if (blockBeforeFirstChunk is not null)
                    await blockBeforeFirstChunk.Task.WaitAsync(ct);

                float[] samples = [0.1f, 0.2f, 0.3f, 0.4f];
                EmittedSamples += samples.Length;
                yield return new TtsAudioChunk
                {
                    Samples = samples,
                    SampleRate = OutputSampleRate,
                    Channels = OutputChannels,
                    IsFinal = true,
                    EmittedSeconds = samples.Length / (double)OutputSampleRate,
                };

                if (ignoringCancellationBlock is not null)
                {
                    Interlocked.Increment(ref _ignoringCancellationBlockEntries);
                    await ignoringCancellationBlock.Task;
                }

                if (blockAfterFirstChunk is not null)
                    await blockAfterFirstChunk.Task.WaitAsync(ct);
            }
            finally
            {
                StreamCompleted?.Invoke();
            }
        }

        public void Dispose() { }
    }
}

#pragma warning restore CS0067
