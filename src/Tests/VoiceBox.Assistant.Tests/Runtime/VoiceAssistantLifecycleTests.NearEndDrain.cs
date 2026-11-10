using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ActiveNearEndCapture_RecognitionVadResumesBeforeCatchUpFrameFinalizesDrain(bool vadThrows)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
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
        EmitPlaybackEchoFrames(services, 20, 0.08, 0.02);
        EmitNearEndFrames(services, 8, 0.08, 0.12, emitMainAudio: true);
        await heard.Task.WaitAsync(TimeSpan.FromSeconds(2));

        AssistantRuntime runtime = GetPrivateField<AssistantRuntime>(assistant, "_runtime");
        INearEndEndpointPort endpoint = GetPrivateField<INearEndEndpointPort>(runtime, "_nearEnd");
        NearEndCaptureStateMachine capture = GetPrivateField<NearEndCaptureStateMachine>(endpoint, "_capture");
        long turnId = capture.ActiveTurnId;
        int finishCount = services.Asr.FinishCount;
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(160));

        float[] silence = new float[320];
        CaptureRange first = services.Audio.EmitDuplexOnly(silence, silence, silence, silence);
        _ = services.Audio.EmitDuplexOnly(silence, silence, silence, silence);
        Assert.True(capture.EndDrainPending);
        CaptureRange last = services.Audio.EmitDuplexOnly(silence, silence, silence, silence);
        CaptureRange combined = first with
        {
            EndSequenceNumber = last.EndSequenceNumber,
            EndSampleExclusive = last.EndSampleExclusive,
        };
        float[] resumedAudio = ConstantSamples(0.37f, 960);
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(resumedAudio);
            if (vadThrows)
                throw new InvalidOperationException("VAD failed after publishing speech.");
        };

        services.Audio.EmitProcessedCapture(in combined, resumedAudio);

        Assert.Equal(turnId, capture.ActiveTurnId);
        Assert.False(capture.EndDrainPending);
        Assert.Equal(combined.EndSampleExclusive, capture.DirectFeedNextSample);
        Assert.Equal(finishCount, services.Asr.FinishCount);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(220));
        EmitSilenceFrames(services, 8, 0);
        await WaitUntilAsync(() => services.Asr.FinishCount == finishCount + 1);
        Assert.Equal(resumedAudio.Length, services.Asr.FedAudio["session-2"]
            .SelectMany(static chunk => chunk).Count(static sample => sample == 0.37f));
    }

    [Theory]
    [InlineData(1L)]
    [InlineData(2L)]
    public void ActiveNearEndCapture_EndpointResumptionCannotReopenDiscontinuousDrain(long epoch)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(services, faults, (_, _) => speechEnded++);
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        coordinator.ApplyConfirmedCapture(CreateConfirmationLease(4), CreateConfirmedDecision());
        coordinator.ObserveDuplexAudio(CreateDuplexFrame(1, 4, 16), false);

        MainAudioObservation discontinuous = ObserveRecognitionAudio(coordinator, 24, [25f], epoch);
        Assert.True(discontinuous.HasDiscontinuity);
        coordinator.ObserveEndpointSpeechStarted();

        Assert.True(coordinator.Module.Capture.EndDrainPending);
        Assert.True(coordinator.Module.Capture.EndDrainForced);
        coordinator.FeedFromRecognitionAudio([25f], discontinuous);
        Assert.Equal(1, speechEnded);
        Assert.Empty(faults);
    }

    [Fact]
    public async Task ActiveNearEndCapture_ConfirmationDoesNotSkipAudioObservedDuringCatchUpFeed()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        ObserveRecognitionAudio(coordinator, 4, [5f, 6f, 7f, 8f]);
        await coordinator.AttachAsrSessionAsync(1, "confirmation-handoff");

        MainAudioObservation observedDuringFeed = default;
        services.Asr.FeedContext = () =>
        {
            // Interleave capture's observation between the catch-up snapshot and commit's
            // final lock. Its separate feed callback has not run yet.
            services.Asr.FeedContext = null;
            observedDuringFeed = ObserveRecognitionAudio(coordinator, 8, [9f, 10f, 11f, 12f]);
            return false;
        };

        coordinator.ApplyConfirmedCapture(CreateConfirmationLease(4), CreateConfirmedDecision());

        Assert.Equal(8, coordinator.Module.Capture.DirectFeedNextSample);
        coordinator.FeedFromRecognitionAudio([9f, 10f, 11f, 12f], observedDuringFeed);
        coordinator.FeedFromRecognitionAudio([9f, 10f, 11f, 12f], observedDuringFeed);

        Assert.Empty(faults);
        Assert.Equal(12, coordinator.Module.Capture.DirectFeedNextSample);
        Assert.Equal(8, coordinator.Module.Capture.DirectFeedSamples);
        Assert.Equal([5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f], services.Asr.FedAudio["confirmation-handoff"].SelectMany(static chunk => chunk));
    }

    [Theory]
    [InlineData(true, 20)]
    [InlineData(false, 20)]
    [InlineData(true, 16)]
    [InlineData(false, 16)]
    [InlineData(true, 8)]
    [InlineData(false, 8)]
    public async Task ActiveNearEndCapture_EndpointResumptionKeepsDelayedPcmPastOldDrainTarget(bool speechStarted, int firstResumedSamples)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(services, faults, (_, _) => speechEnded++);
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        coordinator.ApplyConfirmedCapture(CreateConfirmationLease(4), CreateConfirmedDecision());
        await coordinator.AttachAsrSessionAsync(1, "resumed-endpoint");
        coordinator.ObserveEndpointSpeechEnded();
        coordinator.ObserveDuplexAudio(CreateDuplexFrame(1, 4, 16), false);
        Assert.True(coordinator.Module.Capture.EndDrainPending);
        Assert.Equal(20, coordinator.Module.Capture.EndDrainTargetSample);

        float[] resumedAudio = Enumerable.Range(5, 20).Select(static value => (float)value).ToArray();
        float[] firstResumedAudio = resumedAudio[..firstResumedSamples];
        MainAudioObservation observation = ObserveRecognitionAudio(coordinator, 4, firstResumedAudio);
        if (speechStarted)
            coordinator.ObserveEndpointSpeechStarted();
        else
            coordinator.ObserveEndpointSpeechAudio(firstResumedAudio);

        // Recognition may catch up before another duplex callback. The synchronous VAD
        // fact must already have invalidated the old endpoint before that feed completes.
        Assert.False(coordinator.Module.Capture.EndDrainPending);
        coordinator.FeedFromRecognitionAudio(firstResumedAudio, observation);
        if (firstResumedSamples < resumedAudio.Length)
            FeedRecognitionAudio(coordinator, 4 + firstResumedSamples, resumedAudio[firstResumedSamples..]);
        Assert.Equal(0, speechEnded);
        Assert.Equal(1, coordinator.Module.Capture.ActiveTurnId);
        Assert.Equal(24, coordinator.Module.Capture.DirectFeedNextSample);

        coordinator.ObserveEndpointSpeechEnded();
        coordinator.ObserveDuplexAudio(CreateDuplexFrame(2, 24, 16), false);
        Assert.Equal(40, coordinator.Module.Capture.EndDrainTargetSample);
        float[] trailingSilence = new float[16];
        FeedRecognitionAudio(coordinator, 24, trailingSilence);

        Assert.Equal(1, speechEnded);
        Assert.Empty(faults);
        Assert.Equal(resumedAudio.Concat(trailingSilence), services.Asr.FedAudio["resumed-endpoint"].SelectMany(static chunk => chunk));
    }

    [Fact]
    public async Task ActiveNearEndCapture_ReplayResumptionIsAppliedBeforeItsPcmReachesOldTarget()
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        int speechEnded = 0;
        var coordinator = CreateDataPlaneCoordinator(services, faults, (_, _) => speechEnded++);
        ObserveRecognitionAudio(coordinator, 0, [1f, 2f, 3f, 4f]);
        coordinator.SeedPendingConfirmation(CreateConfirmationLease(4));
        coordinator.ObserveUnownedEndpointSpeechEnded();
        coordinator.ObserveEndpointSpeechEnded();
        coordinator.ObserveDuplexAudio(CreateDuplexFrame(1, 4, 20), false, latestNearEndSpeech: true);
        await coordinator.AttachAsrSessionAsync(1, "resumed-replay");
        float[] resumedAudio = Enumerable.Range(5, 20).Select(static value => (float)value).ToArray();
        coordinator.DataPlane.BeginEndpointObservation();
        try
        {
            // Confirmation commits during a capture callback, before the main PCM catches
            // up to the already queued DTD tail. The processed endpoint is still provisional.
            coordinator.ApplyConfirmedCapture(CreateConfirmationLease(4), CreateConfirmedDecision());
            Assert.True(coordinator.Module.Replay.HasFrames);
            Assert.True(coordinator.Module.Capture.EndDrainPending);
            FeedRecognitionAudio(coordinator, 4, resumedAudio);
        }
        finally
        {
            coordinator.DataPlane.CompleteEndpointObservation();
        }

        Assert.False(coordinator.Module.Replay.HasFrames);
        Assert.False(coordinator.Module.Capture.EndDrainPending);
        Assert.Equal(0, speechEnded);
        Assert.Empty(faults);
        Assert.Equal(resumedAudio, services.Asr.FedAudio["resumed-replay"].SelectMany(static chunk => chunk));
    }
}
