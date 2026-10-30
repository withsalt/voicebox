using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public async Task ActiveNearEndTrigger_EachRequiredSpeakerWindowGetsItsOwnInferenceBudget(
        int requiredMatches,
        bool pollBetweenWindows)
    {
        var clock = new ManualTimerTimeProvider();
        var provider = new ScriptedSpeakerService(windowSamples: 8000, hopSamples: 3200);
        using SpeechGatedSpeakerVerifier verifier = CreateDeadlineVerifier(provider, requiredMatches);
        var evidence = new DeadlineSpeakerEvidence(verifier);
        NearEndTestHarness harness = CreateInferenceDeadlineHarness(evidence, clock);

        try
        {
            long start = 0;
            for (int window = 0; window < requiredMatches; window++)
            {
                if (window > 0)
                    clock.Advance(TimeSpan.FromMilliseconds(51));
                provider.ExtractEntered = new(TaskCreationOptions.RunContinuationsAsynchronously);
                provider.ExtractBlock = new(TaskCreationOptions.RunContinuationsAsynchronously);
                float[] audio = ScriptedSpeakerService.OwnerScoreAudio(0.6f, window == 0 ? 8000 : 3200);
                long end = start + audio.Length;
                verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, start, audio));
                harness.ObserveRecognitionAudio(new ProcessedCaptureFrame(
                    new CaptureRange(1, window + 1, window + 1, start, end, 1), 16000, 1, audio, 0, false));
                DuplexAudioFrame frame = CreateDuplexFrame(window + 1, start, audio.Length);

                harness.ObserveDuplexAudio(frame, playbackStartupGuardActive: false);
                await provider.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
                Assert.Equal(SpeakerVerificationReason.WaitingForInference, evidence.Last.Reason);
                Assert.Equal(window, evidence.Last.MatchedWindows);
                Assert.Equal(window * 3200, evidence.Last.PendingWindowStartSample);
                Assert.Equal(0, harness.CommandDeniedCount);

                // Poll just before this job's deadline. Prior windows and audio waiting must
                // not consume its budget, including when no intermediate poll saw completion.
                clock.Advance(TimeSpan.FromMilliseconds(149));
                harness.ObserveDuplexAudio(frame, playbackStartupGuardActive: false);
                Assert.True(harness.GetDiagnosticSnapshot().CandidateActive);
                Assert.Equal(0, harness.CommandDeniedCount);

                provider.ExtractBlock.TrySetResult();
                int expectedWindows = window + 1;
                await WaitUntilAsync(() => verifier.Cache.Count >= expectedWindows);
                if (pollBetweenWindows || expectedWindows == requiredMatches)
                    harness.ObserveDuplexAudio(frame, playbackStartupGuardActive: false);
                if (expectedWindows < requiredMatches && pollBetweenWindows)
                    Assert.Equal(SpeakerVerificationReason.WaitingForEvidence, evidence.Last.Reason);
                start = end;
            }

            Assert.Equal(0, harness.CommandDeniedCount);
            Assert.Equal(1, harness.CommandAuthorizedCount);
            Assert.True(harness.TryAcquire(1, out NearEndConfirmationLease lease));
            Assert.True(lease.IdentityAuthorized);
            Assert.Equal(requiredMatches, lease.SpeakerResult.MatchedWindows);
            Assert.Equal(SpeakerVerificationReason.MajorityMatch, lease.SpeakerResult.Reason);
        }
        finally
        {
            provider.ExtractBlock?.TrySetResult();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ActiveNearEndTrigger_SlidingSelectionDoesNotExtendABlockedSpeakerJob(bool reuseRunningPrefetch)
    {
        var clock = new ManualTimerTimeProvider();
        var provider = new ScriptedSpeakerService(windowSamples: 8000, hopSamples: 3200)
        {
            ExtractEntered = new(TaskCreationOptions.RunContinuationsAsynchronously),
            ExtractBlock = new(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        using SpeechGatedSpeakerVerifier verifier = CreateDeadlineVerifier(provider, requiredMatches: 2);
        verifier.Prefetch.SetEnabled(reuseRunningPrefetch);
        var evidence = new DeadlineSpeakerEvidence(verifier);
        NearEndTestHarness harness = CreateInferenceDeadlineHarness(evidence, clock);

        try
        {
            verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, 0,
                ScriptedSpeakerService.OwnerScoreAudio(0.6f, 8000)));
            if (reuseRunningPrefetch)
                await provider.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            harness.ObserveDuplexAudio(CreateDuplexFrame(1, 0, 8000), false);
            await provider.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

            for (int hop = 0; hop < 4; hop++)
            {
                long start = 8000 + hop * 3200;
                verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, start,
                    ScriptedSpeakerService.OwnerScoreAudio(0.6f, 3200)));
                clock.Advance(TimeSpan.FromMilliseconds(40));
                harness.ObserveDuplexAudio(CreateDuplexFrame(hop + 2, start, 3200), false);
                // Even after the first window leaves the recent selection, it still owns the
                // worker. Reporting the next requested window would incorrectly renew its wait.
                Assert.Equal(0, evidence.Last.PendingWindowStartSample);
                Assert.Equal(8000, evidence.Last.PendingWindowEndSampleExclusive);
            }

            Assert.Equal(1, provider.ExtractCount);
            Assert.Equal(1, harness.CommandDeniedCount);
            Assert.Equal(0, harness.CommandAuthorizedCount);
            Assert.Equal("command-denied-speaker-unavailable:InferenceTimeout",
                harness.GetDiagnosticSnapshot().LastCandidateOutcome);
            Assert.True(evidence.Query.InputComplete);
            Assert.Equal(19200, evidence.Query.CandidateEndSampleExclusive); // Original 1200ms input limit.
        }
        finally
        {
            provider.ExtractBlock.TrySetResult();
        }
    }

    private static SpeechGatedSpeakerVerifier CreateDeadlineVerifier(
        ScriptedSpeakerService provider, int requiredMatches)
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.Confirmation.MinimumSpeechMs = 300;
        options.Confirmation.InferenceTimeoutMs = 150;
        options.Confirmation.RequiredMatches = requiredMatches;
        options.Confirmation.RecentOwnerPriorMs = 0;
        var verifier = new SpeechGatedSpeakerVerifier(provider,
            SpeakerTestProfiles.Policy(provider, options),
            SpeakerTestProfiles.Variant(SpeakerTestProfiles.Descriptor()), NullLogger.Instance);
        verifier.Start();
        verifier.Prefetch.SetEnabled(false);
        return verifier;
    }

    private static NearEndTestHarness CreateInferenceDeadlineHarness(
        ISpeakerEvidenceProvider evidence, TimeProvider clock)
    {
        var services = TestServices.Create();
        var decision = new BargeInDecision(BargeInAction.Continue,
            1, 1, 0, 0, 20, 20, 180, 20, 20, 0.1, 0, 0, 0, "insufficient-nearend-duration",
            StreamEpoch: 1, StartSample: 0, EndSampleExclusive: 8000,
            LatestFrameAccepted: true, FirstAcceptedStartSample: 0);
        return CreateNearEndTestHarness(new VoiceAssistantBargeInOptions
        {
            Enable = true,
            ActiveNearEndPreRollMs = 0,
        }, new TestRecognitionAudioPort(services.Asr), () => true, (_, _, _) => decision,
            new AssistantAudioFormat(new DuplexAssistantAudioFrontend(services.Audio).Input,
                new SpeechSynthesisStreamAdapter(services.Tts, null, null)), NullLogger.Instance,
            () => false, () => true, _ => true, (_, _) => { }, evidence, clock);
    }

    private sealed class DeadlineSpeakerEvidence(SpeechGatedSpeakerVerifier verifier) : ISpeakerEvidenceProvider
    {
        public SpeakerVerificationResult Last { get; private set; }
        public SpeakerEvidenceQuery Query { get; private set; }
        public SpeakerCandidateId NewCandidateId() => verifier.NewCandidateId();
        public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query)
        {
            Query = query;
            return Last = verifier.Evaluate(in query);
        }
        public Task<SpeakerVerificationResult> WaitForDecisionAsync(SpeakerEvidenceQuery query,
            TimeSpan timeout, CancellationToken cancellationToken = default)
            => verifier.WaitForDecisionAsync(query, timeout, cancellationToken);
        public void Cancel(SpeakerCandidateId candidateId) => verifier.Cancel(candidateId);
        public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => verifier.GetInferenceTimeout(purpose);
    }
}
