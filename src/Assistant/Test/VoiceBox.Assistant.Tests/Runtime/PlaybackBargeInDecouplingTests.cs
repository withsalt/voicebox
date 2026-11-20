using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(true, true, 360)]
    [InlineData(true, true, 680)]
    [InlineData(true, false, 360)]
    [InlineData(true, false, 680)]
    [InlineData(false, true, 360)]
    [InlineData(false, true, 680)]
    [InlineData(false, false, 360)]
    [InlineData(false, false, 680)]
    public void PlaybackBargeIn_MovingDecisionWindow_SubmitsWithoutMainCatchUp(bool identity, bool vadStart, int lagSamples)
    {
        var services = TestServices.Create();
        var speaker = identity ? new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match) : null;
        int confirmations = 0;
        long confirmationOperationId = 0;
        const int mainEnd = 3200;
        long decisionEnd = mainEnd + lagSamples;
        long decisionStart = vadStart ? 1600 : 800;
        BargeInDecision decision = new(
            identity ? BargeInAction.Continue : BargeInAction.Interrupt,
            1,
            1,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            0.1,
            0,
            0,
            0,
            "test",
            StreamEpoch: 1,
            StartSample: decisionStart,
            EndSampleExclusive: decisionEnd,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: decisionStart);
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
                MinDurationMs = 20,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            operationId => { confirmationOperationId = operationId; confirmations++; return true; },
            (_, _) => { },
            speaker);

        float[] audio = Enumerable.Range(0, mainEnd).Select(static i => 0.03f + i * 0.00001f).ToArray();
        ObserveRecognitionAudio(harness, 0, audio);
        if (vadStart)
        {
            harness.ObserveEndpointSpeechStarted();
            harness.ObserveEndpointSpeechAudio(audio.AsSpan((int)decisionStart));
        }

        // The DTD decision end is one whole frame ahead of the main timeline. No second frame or
        // catch-up callback is allowed before the confirmation must exist.
        harness.ObserveDuplexAudio(CreateDuplexFrame(1, mainEnd, lagSamples), playbackStartupGuardActive: false);

        Assert.Equal(1, confirmations);
        Assert.True(harness.TryAcquire(confirmationOperationId, out NearEndConfirmationLease lease));
        Assert.NotNull(lease.CapturePlan);
        Assert.True(lease.CapturePlan!.Value.Ready);
        Assert.Equal(identity, lease.IdentityRequired);
        Assert.Equal(identity, lease.IdentityAuthorized);
        Assert.False(harness.GetDiagnosticSnapshot().CandidateActive);
        Assert.Equal(identity ? 1 : 0, speaker?.EvaluateCount ?? 0);
    }

    [Fact]
    public async Task PlaybackBargeIn_DeferredCaptureWaitsForMainStartThenFeedsOnce()
    {
        var services = TestServices.Create();
        int confirmations = 0;
        long confirmationOperationId = 0;
        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 1,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 20,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 20,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 320,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
                ActiveNearEndWindowMs = 100,
                PlaybackMaxCandidateMs = 100,
                PlaybackRecognitionAudioLimitMs = 100,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            operationId => { confirmationOperationId = operationId; confirmations++; return true; },
            (_, _) => { });

        // No main audio exists yet: the capture can only be registered, not snapshotted.
        harness.ObserveDuplexAudio(CreateDuplexFrame(1, 0, 320), playbackStartupGuardActive: false);
        Assert.Equal(1, confirmations);
        Assert.True(harness.TryAcquire(confirmationOperationId, out NearEndConfirmationLease lease));
        Assert.NotNull(lease.CapturePlan);
        Assert.False(lease.CapturePlan!.Value.Ready);

        harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        Assert.Equal(1, harness.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.True(harness.Module.Capture.Pending);

        await harness.AttachAsrSessionAsync(1, "deferred-capture");
        float[] marker = ConstantSamples(0.42f, 320);
        MainAudioObservation observation = ObserveRecognitionAudio(harness, 0, marker);
        harness.FeedFromRecognitionAudio(marker, observation);

        Assert.False(harness.Module.Capture.Pending);
        Assert.Equal(1, harness.Module.Capture.ActiveTurnId);
        float[] fed = services.Asr.FedAudio["deferred-capture"].SelectMany(static chunk => chunk).ToArray();
        Assert.Equal(marker.Length, fed.Count(static sample => Math.Abs(sample - 0.42f) < 0.0001f));
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(false, true)]
    public async Task PlaybackBargeIn_LateVadLocationPreservesPreRoll(bool commitBeforeVadAudio, bool endBeforeCommit)
    {
        var services = TestServices.Create();
        var speaker = new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match);
        var pcm = Enumerable.Range(0, 19200).Select(i => i / 65536f).ToArray();
        var decision = new BargeInDecision(
            BargeInAction.Continue,
            FrameCount: 5,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 4,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 100,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 180,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 10000,
            EndSampleExclusive: pcm.Length + 320,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 10000);
        long operation = 0;
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 160,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            id => { operation = id; return true; },
            (_, _) => { },
            speaker);
        ObserveRecognitionAudio(harness, 0, pcm);
        harness.ObserveEndpointSpeechStarted();
        harness.ObserveDuplexAudio(CreateDuplexFrame(2, pcm.Length, 320), false);
        Assert.True(harness.TryAcquire(operation, out var lease));
        if (commitBeforeVadAudio)
            harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        harness.ObserveEndpointSpeechAudio(pcm.AsSpan(4800));
        if (endBeforeCommit)
        {
            harness.ObserveUnownedEndpointSpeechEnded();
            harness.ObserveEndpointSpeechEnded();
        }
        if (!commitBeforeVadAudio)
            harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        await harness.AttachAsrSessionAsync(1, "late-onset");
        Assert.Equal(pcm[(4800 - 2560)..], services.Asr.FedAudio["late-onset"].SelectMany(chunk => chunk));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void PlaybackBargeIn_OldEndpointCannotEndLaterConfirmedSpeech(bool acknowledgeBeforeCommit)
    {
        var services = TestServices.Create();
        const int oldEnd = 3200;
        const int interruptEnd = 3880;
        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 2,
            AcceptedFrameCount: 2,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 40,
            AcceptedDurationMs: 40,
            RequiredDurationMs: 20,
            ScoreDurationMs: 40,
            PeakScoreDurationMs: 40,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "confirmed",
            StreamEpoch: 1,
            StartSample: oldEnd + 40,
            EndSampleExclusive: interruptEnd,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: oldEnd + 40);
        int ended = 0;
        var harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 20,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            _ => true,
            (_, _) => ended++);
        ObserveRecognitionAudio(harness, 0, ConstantSamples(0.01f, oldEnd));
        harness.ObserveUnownedEndpointSpeechEnded();
        if (acknowledgeBeforeCommit)
            harness.ObserveEndpointSpeechEnded();
        harness.ObserveDuplexAudio(CreateDuplexFrame(2, oldEnd, interruptEnd - oldEnd), false);
        Assert.True(harness.TryAcquire(1, out var lease));
        Assert.Equal(oldEnd, lease.Selection.SelectedEndSample);
        harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        if (!acknowledgeBeforeCommit)
            harness.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint: true);
        Assert.Equal(0, ended);
        FeedRecognitionAudio(harness, oldEnd, ConstantSamples(0.25f, interruptEnd - oldEnd));
        harness.ObserveEndpointSpeechEnded();
        Assert.Equal(1, ended);
    }

    [Fact]
    public void PlaybackBargeIn_PlaybackStartPreservesOnlyTheSameOwnerConfirmation()
    {
        var services = TestServices.Create();
        long confirmationOperationId = 0;
        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 1,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 20,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 20,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 320,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            operationId => { confirmationOperationId = operationId; return true; },
            (_, _) => { });

        harness.ObserveDuplexAudio(CreateDuplexFrame(1, 0, 320), playbackStartupGuardActive: false);
        Assert.True(harness.GetDiagnosticSnapshot().ConfirmationPending);

        // The interrupted reply's own TTS boundary cannot erase its queued pre-roll...
        harness.Ports.Reset.ClearAudioReservoirs(playbackTurnId: 1);
        Assert.True(harness.GetDiagnosticSnapshot().ConfirmationPending);
        Assert.True(harness.TryAcquire(confirmationOperationId, out _));

        // ...but a later reply retires the taken confirmation together with its audio.
        harness.Ports.Reset.ClearAudioReservoirs(playbackTurnId: 2);
        Assert.False(harness.GetDiagnosticSnapshot().ConfirmationPending);
        Assert.False(harness.TryAcquire(confirmationOperationId, out _));
    }

    [Fact]
    public void PlaybackBargeIn_PendingVadEndpointStillDrainsAfterActivation()
    {
        var services = TestServices.Create();
        int ended = 0;
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
                ActiveNearEndAudioDrainTimeoutMs = 100,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => default,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            _ => true,
            (_, _) => ended++);

        ObserveRecognitionAudio(harness, 0, ConstantSamples(0.23f, 320));
        var plan = new NearEndCapturePlan(
            StreamEpoch: 1,
            StartSample: 320,
            Source: "test",
            EndpointVadCandidate: true,
            PreferEndpointVadStart: false,
            Ready: false);
        NearEndConfirmationLease lease = CreateConfirmationLease(320) with
        {
            CapturePlan = plan,
            EndpointVadCandidate = true,
        };
        harness.SeedPendingConfirmation(lease);
        harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        Assert.True(harness.Module.Capture.Pending);

        // The VAD end arrives while the capture is still waiting for its start sample.
        harness.ObserveEndpointSpeechEnded();
        float[] tail = ConstantSamples(0.0f, 320);
        MainAudioObservation observation = ObserveRecognitionAudio(harness, 320, tail);
        harness.FeedFromRecognitionAudio(tail, observation);

        Assert.Equal(1, ended);
        Assert.Equal(0, harness.GetDiagnosticSnapshot().ActiveTurnId);
    }

    [Fact]
    public void PlaybackBargeIn_AbortCaptureReleasesPendingOwnership()
    {
        var services = TestServices.Create();
        long confirmationOperationId = 0;
        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 1,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 20,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 20,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 320,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            operationId => { confirmationOperationId = operationId; return true; },
            (_, _) => { });

        harness.ObserveDuplexAudio(CreateDuplexFrame(1, 0, 320), playbackStartupGuardActive: false);
        Assert.True(harness.TryAcquire(confirmationOperationId, out NearEndConfirmationLease lease));
        harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        Assert.True(harness.Module.Capture.Pending);
        Assert.Equal(1, harness.GetDiagnosticSnapshot().ActiveTurnId);

        harness.Confirmations.AbortCapture(1);

        Assert.False(harness.Module.Capture.Pending);
        Assert.Equal(0, harness.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.False(harness.TryAcquire(confirmationOperationId, out _));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task PlaybackBargeIn_PendingCaptureExpiresWithoutFurtherCallbacks(bool hasPendingVadStart)
    {
        var services = TestServices.Create();
        long confirmationOperationId = 0;
        int ended = 0;
        var clock = new ManualTimerTimeProvider();
        var decision = new BargeInDecision(
            BargeInAction.Interrupt,
            FrameCount: 1,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 20,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 20,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: 320,
            LatestFrameAccepted: true,
            FirstAcceptedStartSample: 0);
        NearEndTestHarness harness = CreateNearEndTestHarness(
            new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
                ActiveNearEndWindowMs = 100,
                PlaybackMaxCandidateMs = 100,
                PlaybackRecognitionAudioLimitMs = 100,
                ActiveNearEndAudioDrainTimeoutMs = 20,
            },
            new TestRecognitionAudioPort(services.Asr),
            () => true,
            (_, _, _) => decision,
            new AssistantAudioFormat(
                new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)),
            NullLogger.Instance,
            () => false,
            () => true,
            operationId => { confirmationOperationId = operationId; return true; },
            (_, _) => ended++,
            timeProvider: clock);

        float[] retained = ConstantSamples(0.23f, 320);
        if (hasPendingVadStart)
        {
            ObserveRecognitionAudio(harness, 0, retained);
            harness.ObserveEndpointSpeechStarted();
        }
        harness.ObserveDuplexAudio(CreateDuplexFrame(1, 0, 320), playbackStartupGuardActive: false);
        Assert.True(harness.TryAcquire(confirmationOperationId, out NearEndConfirmationLease lease));
        harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        Assert.True(harness.Module.Capture.Pending);

        clock.Advance(TimeSpan.FromMilliseconds(20));
        Assert.Equal(1, ended);
        Assert.False(harness.Module.Capture.Pending);
        Assert.Equal(0, harness.GetDiagnosticSnapshot().ActiveTurnId);
        if (hasPendingVadStart)
        {
            await harness.AttachAsrSessionAsync(1, "expired-onset");
            Assert.Equal(retained, services.Asr.FedAudio["expired-onset"].SelectMany(chunk => chunk));
        }
    }

    [Fact]
    public async Task PlaybackBargeIn_SpeakerVerdictStopsWithoutFurtherAudioFrames()
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
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.BargeIn.ActiveNearEndPreRollMs = 0;
            });
        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);
            var runtime = GetPrivateField<VoiceBox.Assistant.Runtime.AssistantRuntime>(assistant, "_runtime");
            var nearEnd = RuntimeGraphProbe.RequireField<NearEndDataPlane>(runtime, "_dataPlane");
            services.PlaybackDetector.NearEndSpeechOverride = false;
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(0);
            services.Speaker.FeedEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Speaker.FeedBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

            float[] speech = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(speech);
            await services.Speaker.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(speech);
            await PumpSilentFramesUntilAsync(services, () => nearEnd.GetDiagnosticSnapshot().CandidateActive, maxFrames: 10);
            Assert.True(nearEnd.GetDiagnosticSnapshot().CandidateActive);
            Assert.False(nearEnd.GetDiagnosticSnapshot().CandidateAuthorized);
            Assert.Equal(0, services.Audio.ClearQueueCount);

            // From this point on the test emits no duplex and no processed frame. The speaker
            // worker completes and its wait must submit the authorization on its own.
            services.Speaker.FeedBlock.TrySetResult();
            await WaitUntilAsync(() => services.Audio.ClearQueueCount == 1);
            Assert.Equal(1, nearEnd.CommandAuthorizedCount);
            Assert.False(nearEnd.GetDiagnosticSnapshot().CandidateActive);
        }
        finally
        {
            services.Speaker.FeedBlock?.TrySetResult();
            services.Tts.BlockAfterFirstChunk.TrySetResult();
            await assistant.StopAsync();
            File.Delete(profilePath);
        }
    }
}
