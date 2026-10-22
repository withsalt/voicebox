using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class WakeupSessionGateTests
{
    [Fact]
    public void WakeupEventArgs_PreservesLongMilliseconds()
    {
        long durationMs = (long)int.MaxValue + 123;
        long startTimeMs = (long)int.MaxValue + 456;

        var args = new WakeupEventArgs("小爱同学", 90, 20, durationMs, startTimeMs);

        Assert.Equal(durationMs, args.DurationMs);
        Assert.Equal(startTimeMs, args.StartTimeMs);
    }

    [Fact]
    public void WaitingMode_FeedsWakeDetectorAndDoesNotEnterConversationBeforeWakeup()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120);
        gate.Start();

        WakeupAudioDecision decision = gate.OnAudio([0.1f, 0.2f]);

        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, decision.Route);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.Single(wakeup.FedAudio);
    }

    [Fact]
    public void OwnerSpeakerGate_DoesNotWakeUntilCandidateVerificationMatches()
    {
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            PendingResult(),
            PendingResult(),
            VerificationResult(SpeakerVerificationStatus.Match, 0.8f, SpeakerVerificationReason.MajorityMatch),
        ]);
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 32000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: speakerEvidence);
        gate.Start();
        gate.OnAudio(new float[8000]);
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500);

        // One evaluation per capture position: the gate consults the authorization several times
        // per frame but never re-asks for a position it already found Waiting.
        WakeupAudioDecision pending = gate.OnAudio(new float[4000]);
        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, pending.Route);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.Equal(1, speakerEvidence.EvaluateCount);

        WakeupAudioDecision stillPending = gate.OnAudio(new float[4000]);
        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, stillPending.Route);
        Assert.Equal(2, speakerEvidence.EvaluateCount);

        WakeupAudioDecision accepted = gate.OnAudio(new float[4000]);
        Assert.Equal(WakeupAudioRoute.Conversation, accepted.Route);
        Assert.Equal(WakeupTriggerKind.InitialWakeup, accepted.TriggerKind);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
        Assert.Equal(3, speakerEvidence.EvaluateCount);
    }

    [Fact]
    public void OwnerSpeakerGate_MismatchRestartsWakeDetectorAndStaysAsleep()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: new FakeSpeakerEvidenceProvider([
                VerificationResult(
                    SpeakerVerificationStatus.Mismatch, 0.1f, SpeakerVerificationReason.MajorityMismatch)
            ]));
        gate.Start();
        gate.OnAudio(new float[8000]);
        int startsBeforeHit = wakeup.StartCount;
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500);

        WakeupAudioDecision rejected = gate.OnAudio(new float[8000]);

        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, rejected.Route);
        Assert.Null(rejected.Wakeup);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.True(wakeup.StartCount > startsBeforeHit);
    }

    [Fact]
    public void OwnerSpeakerGate_InSessionMismatchDoesNotExtendIdleTimeoutOrLeaveConversationRoute()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var verification = new Queue<SpeakerVerificationResult>([
            VerificationResult(SpeakerVerificationStatus.Match, 0.8f, SpeakerVerificationReason.MajorityMatch),
            VerificationResult(SpeakerVerificationStatus.Mismatch, 0.1f, SpeakerVerificationReason.MajorityMismatch),
        ]);
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0,
            timeProvider: time,
            speakerEvidence: new FakeSpeakerEvidenceProvider(verification));
        gate.Start();
        gate.OnAudio(new float[8000]);
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500);
        gate.OnAudio(new float[8000]);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);

        time.Advance(TimeSpan.FromSeconds(59));
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500);
        WakeupAudioDecision rejected = gate.OnAudio(new float[8000]);

        Assert.Equal(WakeupAudioRoute.Conversation, rejected.Route);
        Assert.Null(rejected.Wakeup);
        time.Advance(TimeSpan.FromSeconds(2));
        Assert.True(gate.CheckIdleTimeout().ModeChanged);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
    }

    [Fact]
    public void OwnerSpeakerGate_AllowsCandidateStartingAtCurrentBufferEndToCollectFutureAudio()
    {
        var wakeup = new FakeWakeupService
        {
            WakeupOnNextFeed = new WakeupEventArgs("小爱同学", 90, 20, durationMs: 0, startTimeMs: 100),
        };
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            PendingResult(),
            VerificationResult(SpeakerVerificationStatus.Match, 0.8f, SpeakerVerificationReason.MajorityMatch),
        ]);
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 1000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: speakerEvidence);
        gate.Start();

        // The keyword is reported at the very end of the buffer: the candidate [100,100) is empty,
        // so nothing is evaluated (and nothing denied) until future audio arrives.
        WakeupAudioDecision empty = gate.OnAudio(new float[100]);
        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, empty.Route);
        Assert.Equal(default, speakerEvidence.LastQuery);

        WakeupAudioDecision pending = gate.OnAudio(new float[500]);
        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, pending.Route);
        Assert.Equal(100, speakerEvidence.LastQuery.CandidateStartSample);
        Assert.Equal(600, speakerEvidence.LastQuery.CandidateEndSampleExclusive);
        Assert.False(speakerEvidence.LastQuery.HasFocusInterval);

        WakeupAudioDecision accepted = gate.OnAudio(new float[500]);
        Assert.Equal(WakeupTriggerKind.InitialWakeup, accepted.TriggerKind);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
    }

    [Fact]
    public void OwnerSpeakerGate_PassesKwsFocusIntervalToSpeakerEvidence()
    {
        var wakeup = new FakeWakeupService();
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            PendingResult(),
            VerificationResult(SpeakerVerificationStatus.Match, 0.8f, SpeakerVerificationReason.MajorityMatch),
        ]);
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 3000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 200,
            speakerEvidence: speakerEvidence);
        gate.Start();

        gate.OnAudio(new float[1000]);
        wakeup.EmitWakeup(startTimeMs: 200, durationMs: 300);
        WakeupAudioDecision pending = gate.OnAudio(new float[500]);
        Assert.Equal(WakeupTriggerKind.None, pending.TriggerKind);
        WakeupAudioDecision accepted = gate.OnAudio(new float[500]);

        Assert.Equal(WakeupTriggerKind.InitialWakeup, accepted.TriggerKind);
        Assert.Equal(0, speakerEvidence.LastQuery.CandidateStartSample);
        Assert.Equal(200, speakerEvidence.LastQuery.FocusStartSample);
        Assert.Equal(500, speakerEvidence.LastQuery.FocusEndSampleExclusive);
        Assert.True(speakerEvidence.LastQuery.StrongUpstreamEvidence);
    }

    [Fact]
    public void OwnerSpeakerGate_WeakKwsScoreDoesNotMarkStrongUpstreamEvidence()
    {
        var wakeup = new FakeWakeupService();
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            VerificationResult(SpeakerVerificationStatus.Match, 0.8f, SpeakerVerificationReason.MajorityMatch),
        ]);
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 1000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: speakerEvidence,
            wakeupStrongScoreMargin: 15);
        gate.Start();
        gate.OnAudio(new float[500]);

        wakeup.EmitWakeup(score: 30, threshold: 20, durationMs: 500);
        gate.OnAudio(new float[500]);

        Assert.False(speakerEvidence.LastQuery.StrongUpstreamEvidence);
    }

    [Fact]
    public void OwnerSpeakerGate_PendingEvidenceTimesOutAfterCoverageAndRestartsDetector()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var wakeup = new FakeWakeupService();
        var pending = VerificationResult(SpeakerVerificationStatus.Pending, 0, SpeakerVerificationReason.WaitingForEvidence) with
        {
            RequiredEvidenceEndSampleExclusive = 500,
        };
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 2000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 0,
            timeProvider: time,
            speakerEvidence: new FakeSpeakerEvidenceProvider([pending]),
            speakerVerificationTimeout: TimeSpan.FromMilliseconds(100));
        gate.Start();
        gate.OnAudio(new float[500]);
        int startsBeforeHit = wakeup.StartCount;
        wakeup.EmitWakeup(durationMs: 500);

        WakeupAudioDecision pendingDecision = gate.OnAudio(new float[100]);
        time.Advance(TimeSpan.FromMilliseconds(101));
        WakeupAudioDecision rejected = gate.OnAudio(new float[100]);

        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, pendingDecision.Route);
        Assert.Null(rejected.Wakeup);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.True(wakeup.StartCount > startsBeforeHit);
    }

    [Fact]
    public void OwnerSpeakerGate_PostTriggerEvidenceExtendsCandidateAndDenialIsTyped()
    {
        var wakeup = new FakeWakeupService();
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            PendingResult(),
            VerificationResult(
                SpeakerVerificationStatus.InsufficientEvidence, 0.3f, SpeakerVerificationReason.AmbiguousEvidence),
        ]);
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 2000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: speakerEvidence,
            postTriggerEvidenceSamples: 300);
        gate.Start();
        gate.OnAudio(new float[500]);
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500);

        // The keyword ends at 500; with 300 samples of post-trigger context the candidate is not
        // complete until 800, so the pending verdict keeps waiting for audio rather than timing out.
        WakeupAudioDecision waiting = gate.OnAudio(new float[100]);
        Assert.Equal(WakeupAudioRoute.WakeDetectorOnly, waiting.Route);
        Assert.Equal(600, speakerEvidence.LastQuery.CandidateEndSampleExclusive);
        Assert.False(speakerEvidence.LastQuery.InputComplete);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);

        WakeupAudioDecision denied = gate.OnAudio(new float[200]);
        Assert.Null(denied.Wakeup);
        Assert.Equal(800, speakerEvidence.LastQuery.CandidateEndSampleExclusive);
        Assert.True(speakerEvidence.LastQuery.InputComplete);
        Assert.Equal(WakeupAuthorizationOutcome.InsufficientEvidence, gate.LastDeniedOutcome);
        Assert.Equal(1, gate.DeniedWakeups);
        Assert.Equal(0, gate.AuthorizedWakeups);
        Assert.Contains(new SpeakerCandidateId(1), speakerEvidence.Cancelled);
    }

    [Fact]
    public void OwnerSpeakerGate_UnavailableProviderDeniesWithoutKeywordOnlyFallback()
    {
        var wakeup = new FakeWakeupService();
        var speakerEvidence = new FakeSpeakerEvidenceProvider([
            VerificationResult(SpeakerVerificationStatus.Unavailable, 0f, SpeakerVerificationReason.ProviderError),
        ]);
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0,
            speakerEvidence: speakerEvidence);
        gate.Start();
        gate.OnAudio(new float[8000]);
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 500, score: 100, threshold: 20);

        WakeupAudioDecision denied = gate.OnAudio(new float[8000]);

        // Even a very strong keyword score cannot substitute for identity evidence.
        Assert.Null(denied.Wakeup);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.Equal(WakeupAuthorizationOutcome.Unavailable, gate.LastDeniedOutcome);
        Assert.True(speakerEvidence.LastQuery.StrongUpstreamEvidence);
    }

    [Fact]
    public void Wakeup_ReplaysAudioFromWakewordStartAndEntersContinuousMode()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 3200,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0);
        gate.Start();

        gate.OnAudio(Enumerable.Range(0, 3200).Select(i => (float)i).ToArray());
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 100);

        WakeupAudioDecision decision = gate.OnAudio([999f]);

        Assert.Equal(WakeupAudioRoute.Conversation, decision.Route);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
        Assert.NotNull(decision.ReplayAudio);
        Assert.Equal(3201, decision.ReplayAudio!.Length);
        Assert.Equal(0f, decision.ReplayAudio[0]);
        Assert.True(decision.ModeChanged);
    }

    [Fact]
    public void WakeupReplayLeadIn_StartsBeforeWakewordStart()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 3200,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 20);
        gate.Start();

        gate.OnAudio(Enumerable.Range(0, 3200).Select(i => (float)i).ToArray());
        wakeup.EmitWakeup(startTimeMs: 100, durationMs: 100);

        WakeupAudioDecision decision = gate.OnAudio([999f]);

        Assert.NotNull(decision.ReplayAudio);
        Assert.Equal(1921, decision.ReplayAudio!.Length);
        Assert.Equal(1280f, decision.ReplayAudio[0]);
    }

    [Fact]
    public void WakeupReplayLeadIn_CoversLateCtcStartEstimate()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 400);
        gate.Start();

        gate.OnAudio(Enumerable.Range(0, 16000).Select(i => (float)i).ToArray());
        wakeup.EmitWakeup(startTimeMs: 650, durationMs: 700);

        WakeupAudioDecision decision = gate.OnAudio([999f]);

        Assert.NotNull(decision.ReplayAudio);
        Assert.Equal(12001, decision.ReplayAudio!.Length);
        Assert.Equal(4000f, decision.ReplayAudio[0]);
    }

    [Fact]
    public void ContinuousMode_IdleTimeoutReturnsToWakeWord()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120,
            timeProvider: time);
        gate.Start();
        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);

        time.Advance(TimeSpan.FromSeconds(59));
        WakeupTimeoutDecision early = gate.CheckIdleTimeout();
        Assert.False(early.ModeChanged);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);

        time.Advance(TimeSpan.FromSeconds(2));
        WakeupTimeoutDecision timedOut = gate.CheckIdleTimeout();

        Assert.True(timedOut.ModeChanged);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.True(wakeup.StartCount >= 2);
    }

    [Fact]
    public void ContinuousMode_UtcClockRollbackStillUsesElapsedTimeForIdleTimeout()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120,
            timeProvider: time);
        gate.Start();
        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);

        time.Advance(TimeSpan.FromSeconds(59));
        time.RewindUtc(TimeSpan.FromMinutes(10));
        time.Advance(TimeSpan.FromSeconds(2));

        Assert.True(gate.CheckIdleTimeout().ModeChanged);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
    }

    [Fact]
    public void StopDuringWakeDetectorFeed_DoesNotAllowFeedToRestartDetector()
    {
        var wakeup = new FakeWakeupService
        {
            WakeupOnNextFeed = new WakeupEventArgs("小爱同学", 90, 20, durationMs: 0, startTimeMs: 0),
        };
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120);
        gate.Start();
        int startsBeforeFeed = wakeup.StartCount;
        wakeup.BeforeWakeupOnNextFeed = gate.Stop;

        gate.OnAudio([0.1f]);

        Assert.False(wakeup.IsListening);
        Assert.Equal(startsBeforeFeed, wakeup.StartCount);
    }

    [Fact]
    public void ContinuousMode_FeedsWakeDetectorSideChannelWhileRoutingConversation()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120);
        gate.Start();

        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);

        WakeupAudioDecision decision = gate.OnAudio([0.3f]);

        Assert.Equal(WakeupAudioRoute.Conversation, decision.Route);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
        Assert.Equal(2, wakeup.FedAudio.Count);
        Assert.Equal([0.3f], wakeup.FedAudio[1]);
    }

    [Fact]
    public void ContinuousMode_WakeupEventIsReportedAsInSessionWakeup()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0);
        gate.Start();
        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);

        gate.OnAudio([0.3f]);
        wakeup.EmitWakeup();
        WakeupAudioDecision decision = gate.OnAudio([0.4f]);

        Assert.Equal(WakeupAudioRoute.Conversation, decision.Route);
        Assert.Equal(WakeupTriggerKind.InSessionWakeup, decision.TriggerKind);
        Assert.NotNull(decision.Wakeup);
        Assert.False(decision.ModeChanged);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
    }

    [Fact]
    public void ContinuousMode_InSessionWakeupReplayStartsAtWakewordStart()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 1000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 1000,
            channels: 1,
            replayLeadInMs: 200);
        gate.Start();
        gate.OnAudio([0f]);
        wakeup.EmitWakeup();
        gate.OnAudio([1f]);

        gate.OnAudio(Enumerable.Range(0, 1000).Select(i => (float)i).ToArray());
        wakeup.EmitWakeup(startTimeMs: 200, durationMs: 300);
        WakeupAudioDecision decision = gate.OnAudio([1000f, 1001f]);

        Assert.Equal(WakeupTriggerKind.InSessionWakeup, decision.TriggerKind);
        Assert.NotNull(decision.ReplayAudio);
        Assert.Equal(802, decision.ReplayAudio!.Length);
        Assert.Equal(200f, decision.ReplayAudio[0]);
        Assert.Equal(1001f, decision.ReplayAudio[^1]);
    }

    [Fact]
    public void ReturnToWakeWord_LeavesContinuousModeAndRestartsWakeDetector()
    {
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120);
        gate.Start();
        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);
        int startCount = wakeup.StartCount;
        int stopCount = wakeup.StopCount;

        WakeupTimeoutDecision decision = gate.ReturnToWakeWord();

        Assert.True(decision.ModeChanged);
        Assert.Equal(AssistantWakeMode.WaitingForWakeWord, gate.Mode);
        Assert.True(wakeup.StartCount > startCount);
        Assert.True(wakeup.StopCount > stopCount);
    }

    [Fact]
    public void MarkUserActivity_ExtendsContinuousModeTimeout()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 16000,
            idleTimeout: TimeSpan.FromMinutes(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 120,
            timeProvider: time);
        gate.Start();
        gate.OnAudio([0.1f]);
        wakeup.EmitWakeup();
        gate.OnAudio([0.2f]);

        time.Advance(TimeSpan.FromSeconds(50));
        gate.MarkUserActivity();
        time.Advance(TimeSpan.FromSeconds(50));

        WakeupTimeoutDecision decision = gate.CheckIdleTimeout();

        Assert.False(decision.ModeChanged);
        Assert.Equal(AssistantWakeMode.ContinuousConversation, gate.Mode);
    }

    [Fact]
    public void WakeupTiming_IsRelativeToMostRecentWakeDetectorStart()
    {
        var time = new TestTimeProvider(DateTimeOffset.Parse("2026-06-10T00:00:00Z"));
        var wakeup = new FakeWakeupService();
        using var gate = CreateGate(
            wakeup,
            preRollSamples: 3200,
            idleTimeout: TimeSpan.FromSeconds(1),
            sampleRate: 16000,
            channels: 1,
            replayLeadInMs: 0,
            timeProvider: time);
        gate.Start();

        gate.OnAudio(new float[3200]);
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 0);
        gate.OnAudio([1f]);
        time.Advance(TimeSpan.FromSeconds(2));
        gate.CheckIdleTimeout();

        gate.OnAudio(Enumerable.Range(0, 3200).Select(i => (float)i).ToArray());
        wakeup.EmitWakeup(startTimeMs: 0, durationMs: 100);
        WakeupAudioDecision decision = gate.OnAudio([999f]);

        Assert.NotNull(decision.ReplayAudio);
        Assert.Equal(3201, decision.ReplayAudio!.Length);
        Assert.Equal(0f, decision.ReplayAudio[0]);
    }

    private sealed class FakeWakeupService : IWakeupService
    {
        public event EventHandler<WakeupEventArgs>? OnWakeup;
        public List<float[]> FedAudio { get; } = [];
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public bool IsListening { get; private set; }
        public WakeupEventArgs? WakeupOnNextFeed { get; set; }
        public Action? BeforeWakeupOnNextFeed { get; set; }

        public void Start()
        {
            StartCount++;
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
            if (WakeupOnNextFeed is { } wakeup)
            {
                WakeupOnNextFeed = null;
                Action? beforeWakeup = BeforeWakeupOnNextFeed;
                BeforeWakeupOnNextFeed = null;
                beforeWakeup?.Invoke();
                OnWakeup?.Invoke(this, wakeup);
            }
        }

        public void EmitWakeup(
            long startTimeMs = 0,
            long durationMs = 0,
            int score = 90,
            int threshold = 20)
            => OnWakeup?.Invoke(
                this,
                new WakeupEventArgs("小爱同学", score, threshold, durationMs, startTimeMs));

        public void Dispose() { }
    }

    /// <summary>
    /// Tests of wake mechanics default to a speaker evidence provider that authorizes the current
    /// candidate at once. A null provider is the Confirmation.Enable-off keyword-only path.
    /// </summary>
    private static WakeupSessionGate CreateGate(
        IWakeupService wakeup,
        int preRollSamples,
        TimeSpan idleTimeout,
        int sampleRate,
        int channels,
        int replayLeadInMs,
        ISpeakerEvidenceProvider? speakerEvidence = null,
        TimeProvider? timeProvider = null,
        int wakeupStrongScoreMargin = 15,
        TimeSpan? speakerVerificationTimeout = null,
        int postTriggerEvidenceSamples = 0)
    {
        speakerEvidence ??= new FakeSpeakerEvidenceProvider([
            VerificationResult(SpeakerVerificationStatus.Match, 0.9f, SpeakerVerificationReason.HighConfidenceMatch),
        ]);
        if (speakerVerificationTimeout is { } timeout && speakerEvidence is FakeSpeakerEvidenceProvider fake)
            fake.InferenceTimeout = timeout;
        return new WakeupSessionGate(
            wakeup,
            preRollSamples,
            idleTimeout,
            sampleRate,
            channels,
            replayLeadInMs,
            speakerEvidence,
            timeProvider,
            wakeupStrongScoreMargin,
            postTriggerEvidenceSamples);
    }

    private static SpeakerVerificationResult PendingResult()
        => VerificationResult(SpeakerVerificationStatus.Pending, 0, SpeakerVerificationReason.WaitingForEvidence);

    private static SpeakerVerificationResult VerificationResult(
        SpeakerVerificationStatus status,
        float score,
        SpeakerVerificationReason reason)
        => new(status, score, 0.5f, 1, 0, 0, reason);

    private sealed class FakeSpeakerEvidenceProvider : ISpeakerEvidenceProvider
    {
        private readonly Queue<SpeakerVerificationResult> _results;
        private SpeakerVerificationResult _last = PendingResult();
        private long _candidates;

        public SpeakerEvidenceQuery LastQuery { get; private set; }

        public int EvaluateCount { get; private set; }

        public TimeSpan InferenceTimeout { get; set; } = TimeSpan.FromMilliseconds(200);

        public List<SpeakerCandidateId> Cancelled { get; } = [];

        public FakeSpeakerEvidenceProvider(IEnumerable<SpeakerVerificationResult> results)
        {
            _results = new Queue<SpeakerVerificationResult>(results);
        }

        public SpeakerCandidateId NewCandidateId() => new(Interlocked.Increment(ref _candidates));

        public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query)
        {
            LastQuery = query;
            EvaluateCount++;
            if (_results.Count > 0)
                _last = _results.Dequeue();
            return _last with
            {
                StreamEpoch = query.StreamEpoch,
                EvidenceStartSample = query.CandidateStartSample,
                EvidenceEndSampleExclusive = query.CandidateEndSampleExclusive,
                CandidateId = query.CandidateId,
            };
        }

        public Task<SpeakerVerificationResult> WaitForDecisionAsync(
            SpeakerEvidenceQuery query,
            TimeSpan timeout,
            CancellationToken cancellationToken = default)
            => Task.FromResult(Evaluate(in query));

        public void Cancel(SpeakerCandidateId candidateId)
        {
            if (!candidateId.IsNone)
                Cancelled.Add(candidateId);
        }

        public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => InferenceTimeout;
    }

    private sealed class TestTimeProvider : TimeProvider
    {
        private DateTimeOffset _utcNow;
        private long _timestamp;

        public TestTimeProvider(DateTimeOffset utcNow)
        {
            _utcNow = utcNow;
        }

        public override DateTimeOffset GetUtcNow() => _utcNow;

        public override long GetTimestamp() => _timestamp;

        public override long TimestampFrequency => TimeSpan.TicksPerSecond;

        public void Advance(TimeSpan delta)
        {
            _utcNow += delta;
            _timestamp += delta.Ticks;
        }

        public void RewindUtc(TimeSpan delta) => _utcNow -= delta;
    }
}

internal static class WakeupGateTestAudioExtensions
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<WakeupSessionGate, Timeline> Timelines = new();

    public static WakeupAudioDecision OnAudio(this WakeupSessionGate gate, float[] audio)
    {
        Timeline timeline = Timelines.GetValue(gate, static value => new Timeline(value));
        long start = timeline.Position;
        timeline.Position += audio.Length;
        long sequence = ++timeline.Sequence;
        var range = new CaptureRange(1, sequence, sequence, start, timeline.Position, sequence);
        var frame = new ProcessedCaptureFrame(
            range, timeline.SampleRate, timeline.Channels, audio, 0, false);
        return gate.OnAudio(in frame);
    }

    private sealed class Timeline
    {
        public Timeline(WakeupSessionGate gate)
        {
            const System.Reflection.BindingFlags Flags =
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            SampleRate = (uint)(int)(typeof(WakeupSessionGate).GetField("_sampleRate", Flags)?.GetValue(gate)
                ?? throw new InvalidOperationException("Wakeup sample-rate field not found."));
            Channels = (uint)(int)(typeof(WakeupSessionGate).GetField("_channels", Flags)?.GetValue(gate)
                ?? throw new InvalidOperationException("Wakeup channel field not found."));
        }

        public uint SampleRate { get; }
        public uint Channels { get; }
        public long Position { get; set; }
        public long Sequence { get; set; }
    }
}
