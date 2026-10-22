using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Policies;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DTD.RenderAware;
using VoiceBox.DTD.RenderAware.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RenderAwarePlaybackBargeInDetectorTests
{
    [Fact]
    public void FirstModelWindowWarmsUpAndCallbackPartitioningIsIndependent()
    {
        using var first = CreateDetector(out FakeDtd firstDtd, out _);
        using var partitioned = CreateDetector(out FakeDtd partitionedDtd, out _);

        PushFrames(first, 2, 320, 1, 16000);
        PushFrames(partitioned, 8, 80, 1, 16000);

        Assert.Equal(firstDtd.ProcessCount, partitionedDtd.ProcessCount);
        Assert.True(firstDtd.ProcessCount > 0);
        Assert.Equal(first.GetSnapshot().HasObservation, partitioned.GetSnapshot().HasObservation);
        Assert.Equal(first.GetSnapshot().LastEndSampleExclusive, partitioned.GetSnapshot().LastEndSampleExclusive);
    }

    [Fact]
    public void StartupGuardFramesCalibrateEchoProfileWithoutEnteringDecisionHistory()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        dtd.Probability = 0.20f;
        dtd.SmoothedProbability = 0.20f;
        dtd.IsNearEndSpeech = false;

        PushFrames(
            detector,
            21,
            320,
            1,
            16000,
            afterAecScale: 0.35f,
            decisionEnabled: false,
            echoLearningEnabled: true);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 20,
            minRms: 0);
        Assert.True(snapshot.EchoProfileReliability >= 0.75);
        Assert.True(snapshot.TrustedEchoObservationCount >= 16);
        Assert.True(snapshot.PredictedEchoPostAecRms > 0);
        Assert.Equal(0, decision.FrameCount);
    }

    [Fact]
    public void RuntimeEchoProfileIsIndependentOfCallbackPartitioning()
    {
        using var first = CreateDetector(out FakeDtd firstDtd, out _);
        using var partitioned = CreateDetector(out FakeDtd partitionedDtd, out _);
        ConfigurePureEcho(firstDtd);
        ConfigurePureEcho(partitionedDtd);

        PushFrames(first, 20, 320, 1, 16000, afterAecScale: 0.35f, decisionEnabled: false);
        PushFrames(partitioned, 25, 256, 1, 16000, afterAecScale: 0.35f, decisionEnabled: false);

        RenderAwareBargeInSnapshot expected = first.GetSnapshot();
        RenderAwareBargeInSnapshot actual = partitioned.GetSnapshot();
        Assert.Equal(expected.TrustedEchoObservationCount, actual.TrustedEchoObservationCount);
        Assert.Equal(expected.TrustedEchoDurationMs, actual.TrustedEchoDurationMs, 6);
        Assert.Equal(expected.EchoProfileReliability, actual.EchoProfileReliability, 6);
        Assert.Equal(expected.PredictedEchoPostAecRms, actual.PredictedEchoPostAecRms, 6);
    }

    [Theory]
    [InlineData(160)]
    [InlineData(256)]
    [InlineData(320)]
    public void CallbackSizesProduceTheSameHopAndSourceTimeline(int callbackSamples)
    {
        using var reference = CreateDetector(out FakeDtd referenceDtd, out _);
        using var candidate = CreateDetector(out FakeDtd candidateDtd, out _);

        PushFrames(reference, 8, 320, 1, 16000);
        PushFrames(candidate, 2560 / callbackSamples, callbackSamples, 1, 16000);

        Assert.Equal(referenceDtd.ProcessCount, candidateDtd.ProcessCount);
        Assert.Equal(reference.GetSnapshot().LastStartSample, candidate.GetSnapshot().LastStartSample);
        Assert.Equal(reference.GetSnapshot().LastEndSampleExclusive, candidate.GetSnapshot().LastEndSampleExclusive);
    }

    [Fact]
    public void CorrelationFallbackSupersedesWarmupHintAndRemainsStableAfterPathReset()
    {
        var options = new VoiceAssistantBargeInOptions();
        options.Dtd.DecisionHistoryMs = 1000;
        var dtd = new FakeDtd();
        var estimator = new SequenceDelayEstimator();
        using var detector = new RenderAwarePlaybackBargeInDetector(
            options.Dtd,
            options.RenderAlignment,
            dtd,
            estimator);

        var primaryHint = new RenderDelayHint(12, 1, IsStable: true);
        for (int index = 0; index < 10; index++)
        {
            detector.Push(
                CreateFrame(320, 1, 16000, startSample: index * 320L),
                primaryHint,
                decisionEnabled: true);
        }

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        Assert.True(snapshot.AlignmentReady);
        Assert.InRange(snapshot.EstimatedDelayMs, 175, 185);
        Assert.InRange(snapshot.LastLocalDelayMs, 175, 185);
        Assert.InRange(snapshot.LastSelectedDelayMs, 175, 185);
    }

    [Fact]
    public void StableCorrelationPathChangeSupersedesRecentRefinedHoldover()
    {
        var options = new VoiceAssistantBargeInOptions();
        options.Dtd.DecisionHistoryMs = 1000;
        options.RenderAlignment.PathChangeThresholdMs = 20;
        var dtd = new FakeDtd();
        using var detector = new RenderAwarePlaybackBargeInDetector(
            options.Dtd,
            options.RenderAlignment,
            dtd,
            new RefinedThenFallbackDelayEstimator());

        PushFrames(detector, 10, 320, 1, 16000);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        Assert.True(snapshot.AlignmentReady);
        Assert.InRange(snapshot.EstimatedDelayMs, 39, 41);
        Assert.True(snapshot.AlignmentGeneration > 0);
        Assert.True(dtd.ResetCount > 0);
    }

    [Fact]
    public void Resamples48kStereoAndMapsObservationBackToInterleavedSourceTimeline()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);

        PushFrames(detector, 3, 960, 2, 48000);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        Assert.True(dtd.ProcessCount > 0);
        Assert.True(snapshot.HasObservation);
        Assert.Equal(1, snapshot.StreamEpoch);
        Assert.True(snapshot.LastEndSampleExclusive > snapshot.LastStartSample);
        Assert.Equal(2 * 960 * 3, snapshot.LastEndSampleExclusive);
    }

    [Fact]
    public void TimelineGapSuppressesInferenceThenValidStreamResumesAfterWarmup()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);

        long start = 0;
        PushContinuous(detector, 2, 320, 1, 16000, ref start);
        int beforeGap = dtd.ProcessCount;
        int resetsBeforeGap = dtd.ResetCount;
        DtdProcessBatch gap = detector.Push(
            CreateFrame(320, 1, 16000, startSample: start, renderStatus: RenderTimelineStatus.Gap),
            primaryAecDelay: null,
            decisionEnabled: true);
        start += 320;

        Assert.True(gap.StateReset);
        Assert.Equal(RenderTimelineStatus.Gap, gap.RenderStatus);
        Assert.True(dtd.ResetCount > resetsBeforeGap);
        Assert.Equal(beforeGap, dtd.ProcessCount);

        DtdProcessBatch firstValid = detector.Push(
            CreateFrame(320, 1, 16000, startSample: start),
            primaryAecDelay: null,
            decisionEnabled: true);
        start += 320;
        Assert.False(firstValid.HasObservation);

        PushContinuous(detector, 2, 320, 1, 16000, ref start);
        Assert.True(dtd.ProcessCount > beforeGap);
    }

    [Fact]
    public void IndependentPostAecSourceCannotPromoteAModelNegativeFrame()
    {
        var policy = new RenderAwareDtdEvidencePolicy(decisionThreshold: 0.55f);
        var profile = new RuntimeEchoProfileSnapshot(
            Reliability: 1,
            TrustedEchoDurationMs: 1000,
            RelevantEchoDurationMs: 1000,
            PredictedCaptureRms: 0.04,
            PredictedPostAecRms: 0.02,
            PredictedResidualRatio: 0.20,
            PureEchoProbabilityUpper: 0.20f,
            PureEchoSmoothedProbabilityUpper: 0.20f);
        var evidence = new RenderAwareDtdFrameEvidence(
            DecisionEnabled: true,
            EchoLearningEnabled: false,
            IsNearEndSpeech: false,
            Probability: 0.20f,
            SmoothedProbability: 0.20f,
            CaptureRms: 0.12,
            RenderRms: 0.08,
            RenderCorrelation: 0.05,
            AlignmentConfidence: 1,
            HasPostAecEvidence: true,
            PostAecRms: 0.09,
            PostAecResidualRatio: 0.75,
            PostAecRenderCorrelation: 0.04);

        RenderAwareDtdEvidenceDecision decision = policy.Evaluate(in evidence, in profile);

        Assert.False(decision.PreservedNearEnd);
        Assert.False(decision.Accepted);
        Assert.False(decision.EchoLike);
    }

    [Fact]
    public void RenderCoherentEchoCannotUseIndependentSourcePromotion()
    {
        var policy = new RenderAwareDtdEvidencePolicy(decisionThreshold: 0.55f);
        var profile = new RuntimeEchoProfileSnapshot(
            Reliability: 1,
            TrustedEchoDurationMs: 1000,
            RelevantEchoDurationMs: 1000,
            PredictedCaptureRms: 0.04,
            PredictedPostAecRms: 0.02,
            PredictedResidualRatio: 0.20,
            PureEchoProbabilityUpper: 0.20f,
            PureEchoSmoothedProbabilityUpper: 0.20f);
        var evidence = new RenderAwareDtdFrameEvidence(
            DecisionEnabled: true,
            EchoLearningEnabled: true,
            IsNearEndSpeech: false,
            Probability: 0.20f,
            SmoothedProbability: 0.20f,
            CaptureRms: 0.12,
            RenderRms: 0.08,
            RenderCorrelation: 0.80,
            AlignmentConfidence: 1,
            HasPostAecEvidence: true,
            PostAecRms: 0.09,
            PostAecResidualRatio: 0.75,
            PostAecRenderCorrelation: 0.70);

        RenderAwareDtdEvidenceDecision decision = policy.Evaluate(in evidence, in profile);

        Assert.False(decision.PreservedNearEnd);
        Assert.False(decision.Accepted);
    }

    [Fact]
    public void DelayedRenderWindowCrossingShortGapResetsModelAndRewarms()
    {
        var options = new VoiceAssistantBargeInOptions();
        options.Dtd.DecisionHistoryMs = 1000;
        var dtd = new FakeDtd();
        using var detector = new RenderAwarePlaybackBargeInDetector(
            options.Dtd,
            options.RenderAlignment,
            dtd,
            new FixedDelayEstimator(640));
        long start = 0;

        PushContinuous(detector, 4, 320, 1, 16000, ref start);
        int processBeforeGap = dtd.ProcessCount;
        int resetBeforeGap = dtd.ResetCount;
        PushContinuous(detector, 1, 320, 1, 16000, ref start, RenderTimelineStatus.Gap);
        PushContinuous(detector, 1, 320, 1, 16000, ref start);
        int processBeforeDelayedGap = dtd.ProcessCount;

        PushContinuous(detector, 1, 320, 1, 16000, ref start);
        Assert.Equal(processBeforeDelayedGap, dtd.ProcessCount);
        Assert.True(dtd.ResetCount > resetBeforeGap);

        PushContinuous(detector, 3, 320, 1, 16000, ref start);
        Assert.True(dtd.ProcessCount > Math.Max(processBeforeGap, processBeforeDelayedGap));
    }

    [Fact]
    public void EpochDiscontinuityAndFormatChangeAtomicallyResetAllState()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out FakeDelayEstimator estimator);

        PushFrames(detector, 3, 320, 1, 16000);
        Assert.True(detector.GetSnapshot().HasObservation);
        int resetsBefore = dtd.ResetCount;
        int delayResetsBefore = estimator.ResetCount;

        DtdProcessBatch epochBatch = detector.Push(
            CreateFrame(320, 1, 16000, epoch: 2, startSample: 0),
            primaryAecDelay: null,
            decisionEnabled: true);

        Assert.True(epochBatch.StateReset);
        Assert.False(detector.GetSnapshot().HasObservation);
        Assert.Equal(0, detector.GetSnapshot().TrustedEchoObservationCount);
        Assert.Equal(0, detector.GetSnapshot().EchoProfileReliability);
        Assert.True(dtd.ResetCount > resetsBefore);
        Assert.True(estimator.ResetCount > delayResetsBefore);

        detector.Push(
            CreateFrame(960, 2, 48000, epoch: 3, startSample: 0),
            primaryAecDelay: null,
            decisionEnabled: true);
        Assert.Equal(3, detector.GetSnapshot().StreamEpoch);
    }

    [Fact]
    public void RenderTimelineLengthMismatchIsRejectedAndResetsState()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        PushFrames(detector, 3, 320, 1, 16000);
        int resetCount = dtd.ResetCount;

        DuplexAudioFrame frame = CreateFrame(320, 1, 16000);
        frame = frame with
        {
            RenderTimeline = new RenderSignalTimeline(1, 0, 319, RenderTimelineStatus.Valid),
        };

        DtdProcessBatch batch = detector.Push(in frame, null, true);
        Assert.True(batch.StateReset);
        Assert.Equal(RenderTimelineStatus.Valid, batch.RenderStatus);
        Assert.True(dtd.ResetCount > resetCount);
    }

    [Fact]
    public void InferenceFailureIsTerminalAndNeverFallsBack()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        dtd.ThrowOnProcess = true;

        Assert.Throws<InvalidOperationException>(() => PushFrames(detector, 2, 320, 1, 16000));
        Assert.True(detector.GetSnapshot().IsFaulted);
        Assert.Throws<InvalidOperationException>(() => detector.Push(
            CreateFrame(320, 1, 16000, startSample: 640), null, true));
    }

    [Fact]
    public void TimelineAlignedPostAecResidualVetoesPureEchoButAllowsNearEndEnergy()
    {
        using var pureEcho = CreateDetector(out _, out _);
        PushFrames(pureEcho, 12, 320, 1, 16000, afterAecScale: 0.02f);
        BargeInDecision echoDecision = pureEcho.GetRecentDecision(
            TimeSpan.FromMilliseconds(1000),
            minDurationMs: 180,
            minRms: 0.03f);

        Assert.Equal(BargeInAction.Continue, echoDecision.Action);
        Assert.True(echoDecision.EchoLikeFrameCount > 0);

        using var nearEnd = CreateDetector(out _, out _);
        PushFrames(nearEnd, 12, 320, 1, 16000, afterAecScale: 0.4f);
        BargeInDecision nearEndDecision = nearEnd.GetRecentDecision(
            TimeSpan.FromMilliseconds(1000),
            minDurationMs: 180,
            minRms: 0.03f);

        Assert.Equal(BargeInAction.Interrupt, nearEndDecision.Action);
        Assert.Equal(0, nearEndDecision.EchoLikeFrameCount);
    }

    [Fact]
    public void ReliableProfileVetoesEchoOutsideTheBootstrapResidualThreshold()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            21,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.60f,
            decisionEnabled: false);
        Assert.True(detector.GetSnapshot().EchoProfileReliability >= 0.75);

        dtd.Probability = 0.60f;
        dtd.SmoothedProbability = 0.60f;
        dtd.IsNearEndSpeech = true;
        PushContinuous(
            detector,
            10,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.75f,
            echoLearningEnabled: false);

        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 180,
            minRms: 0.03f);
        Assert.Equal(BargeInAction.Continue, decision.Action);
        Assert.True(decision.EchoLikeFrameCount >= 9);
        Assert.True(detector.GetSnapshot().LastObservationEchoLike);
    }

    [Fact]
    public void ReliableProfileDoesNotVetoTheAmbiguousResidualBand()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            21,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.60f,
            decisionEnabled: false);
        Assert.True(detector.GetSnapshot().EchoProfileReliability >= 0.75);

        dtd.Probability = 0.60f;
        dtd.SmoothedProbability = 0.60f;
        dtd.IsNearEndSpeech = true;
        PushContinuous(
            detector,
            10,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.40f,
            echoLearningEnabled: false);

        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 180,
            minRms: 0.03f);
        Assert.Equal(BargeInAction.Interrupt, decision.Action);
        Assert.Equal(0, decision.EchoLikeFrameCount);
    }

    [Fact]
    public void CoherentPostAecEnvelopeBreakStartsAcousticPathChangeGuard()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            21,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.02f,
            decisionEnabled: false);
        long trustedBefore = detector.GetSnapshot().TrustedEchoObservationCount;

        dtd.Probability = 0.90f;
        dtd.SmoothedProbability = 0.90f;
        dtd.IsNearEndSpeech = true;
        PushContinuous(
            detector,
            10,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.10f,
            echoLearningEnabled: true);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 180,
            minRms: 0.03f);
        Assert.Equal(BargeInAction.Continue, decision.Action);
        Assert.True(snapshot.AcousticPathChangeGuardActive);
        Assert.Equal(1, snapshot.AcousticPathChangeCount);
        Assert.True(snapshot.LastObservationPathChangeGuarded);
        Assert.False(snapshot.LastObservationPreservedNearEnd);
        Assert.True(snapshot.TrustedEchoObservationCount > trustedBefore);
    }

    [Fact]
    public void IndependentNearEndEnergyStillUsesPreservedNearEndEvidence()
    {
        var policy = new RenderAwareDtdEvidencePolicy(decisionThreshold: 0.55f);
        var profile = new RuntimeEchoProfileSnapshot(
            Reliability: 1,
            TrustedEchoDurationMs: 1000,
            RelevantEchoDurationMs: 1000,
            PredictedCaptureRms: 0.04,
            PredictedPostAecRms: 0.02,
            PredictedResidualRatio: 0.20,
            PureEchoProbabilityUpper: 0.20f,
            PureEchoSmoothedProbabilityUpper: 0.20f);
        var evidence = new RenderAwareDtdFrameEvidence(
            DecisionEnabled: true,
            EchoLearningEnabled: false,
            IsNearEndSpeech: true,
            Probability: 0.90f,
            SmoothedProbability: 0.90f,
            CaptureRms: 0.12,
            RenderRms: 0.08,
            RenderCorrelation: 0.08,
            AlignmentConfidence: 1,
            HasPostAecEvidence: true,
            PostAecRms: 0.09,
            PostAecResidualRatio: 0.75,
            PostAecRenderCorrelation: 0.05);

        RenderAwareDtdEvidenceDecision decision = policy.Evaluate(in evidence, in profile);

        Assert.True(decision.Accepted);
        Assert.True(decision.PreservedNearEnd);
        Assert.False(decision.AcousticPathChange);
    }

    [Fact]
    public void PlaybackDuckingPreservesNearEndEvidenceAndDoesNotStartPathChangeGuard()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        dtd.Probability = 0.95f;
        dtd.SmoothedProbability = 0.95f;
        dtd.IsNearEndSpeech = true;
        long start = 0;
        PushContinuous(detector, 5, 320, 1, 16000, ref start,
            afterAecScale: 0.8f, playbackGain: 1f, renderCoherent: false);
        BargeInDecision before = detector.GetRecentDecision(TimeSpan.FromSeconds(1), 180, 0.03f);
        Assert.Equal(BargeInAction.Continue, before.Action);
        Assert.True(before.AcceptedFrameCount > 0);

        PushContinuous(detector, 7, 320, 1, 16000, ref start,
            afterAecScale: 0.8f, playbackGain: 0.15f, renderCoherent: false);

        BargeInDecision after = detector.GetRecentDecision(TimeSpan.FromSeconds(1), 180, 0.03f);
        Assert.Equal(BargeInAction.Interrupt, after.Action);
        Assert.True(after.AcceptedFrameCount > before.AcceptedFrameCount);
        Assert.Equal(0, detector.GetSnapshot().AcousticPathChangeCount);
    }

    [Fact]
    public void SignificantPlaybackGainIncreaseGuardsExactlyOneTransitionWindow()
    {
        var options = new VoiceAssistantBargeInOptions();
        options.Dtd.AcousticPathChangeGuardMs = 500;
        using var detector = CreateDetector(out FakeDtd dtd, out _, options);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            3,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.02f,
            playbackGain: 0.15f);

        dtd.Probability = 0.90f;
        dtd.SmoothedProbability = 0.90f;
        dtd.IsNearEndSpeech = true;
        PushContinuous(
            detector,
            10,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.10f,
            playbackGain: 1f);

        RenderAwareBargeInSnapshot guarded = detector.GetSnapshot();
        BargeInDecision guardedDecision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 180,
            minRms: 0.03f);
        Assert.Equal(BargeInAction.Continue, guardedDecision.Action);
        Assert.True(guarded.AcousticPathChangeGuardActive);
        Assert.Equal(1, guarded.AcousticPathChangeCount);
        Assert.Equal("playback-gain-change", guarded.LastAcousticPathChangeReason);
        Assert.InRange(guarded.AcousticPathChangeGuardRemainingMs, 299, 301);

        PushContinuous(
            detector,
            15,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.10f,
            playbackGain: 1f);

        RenderAwareBargeInSnapshot completed = detector.GetSnapshot();
        Assert.False(completed.AcousticPathChangeGuardActive);
        Assert.Equal(1, completed.AcousticPathChangeCount);
    }

    [Fact]
    public void NearEndFrameCannotPolluteProfileBeforeItsDecision()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            21,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.02f,
            decisionEnabled: false);
        long trustedBefore = detector.GetSnapshot().TrustedEchoObservationCount;

        dtd.Probability = 0.90f;
        dtd.SmoothedProbability = 0.90f;
        dtd.IsNearEndSpeech = true;
        PushContinuous(
            detector,
            2,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.10f,
            renderCoherent: false);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 20,
            minRms: 0);
        Assert.True(snapshot.LastObservationPreservedNearEnd);
        Assert.InRange(snapshot.TrustedEchoObservationCount, trustedBefore, trustedBefore + 1);
        Assert.Equal(1, decision.AcceptedFrameCount);
    }

    [Fact]
    public void EchoLearningCanBeDisabledDuringCandidateOrValidation()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);

        PushFrames(
            detector,
            21,
            320,
            1,
            16000,
            afterAecScale: 0.35f,
            decisionEnabled: false,
            echoLearningEnabled: false);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        Assert.Equal(0, snapshot.TrustedEchoObservationCount);
        Assert.Equal(0, snapshot.EchoProfileReliability);
    }

    [Fact]
    public void RenderGapAtomicallyClearsRuntimeEchoProfile()
    {
        using var detector = CreateDetector(out FakeDtd dtd, out _);
        ConfigurePureEcho(dtd);
        long start = 0;
        PushContinuous(
            detector,
            21,
            320,
            1,
            16000,
            ref start,
            afterAecScale: 0.35f,
            decisionEnabled: false);
        Assert.True(detector.GetSnapshot().TrustedEchoObservationCount > 0);

        PushContinuous(
            detector,
            1,
            320,
            1,
            16000,
            ref start,
            RenderTimelineStatus.Gap,
            afterAecScale: 0.35f,
            decisionEnabled: false);

        RenderAwareBargeInSnapshot snapshot = detector.GetSnapshot();
        Assert.Equal(0, snapshot.TrustedEchoObservationCount);
        Assert.Equal(0, snapshot.EchoProfileReliability);
    }

    [Fact]
    public void PostAecEvidenceUsesTheCurrentModelHopTimeline()
    {
        using var detector = CreateDetector(out _, out _);

        detector.Push(
            CreateFrame(320, 1, 16000, startSample: 0, afterAecScale: 1f),
            primaryAecDelay: null,
            decisionEnabled: true);
        detector.Push(
            CreateFrame(320, 1, 16000, startSample: 320, afterAecScale: 0.02f),
            primaryAecDelay: null,
            decisionEnabled: true);
        detector.Push(
            CreateFrame(320, 1, 16000, startSample: 640, afterAecScale: 0.02f),
            primaryAecDelay: null,
            decisionEnabled: true);

        BargeInDecision decision = detector.GetRecentDecision(
            TimeSpan.FromSeconds(1),
            minDurationMs: 20,
            minRms: 0.03f);

        Assert.Equal(1, decision.FrameCount);
        Assert.Equal(1, decision.EchoLikeFrameCount);
        Assert.Equal(320, decision.StartSample);
        Assert.Equal(640, decision.EndSampleExclusive);
    }

    [Fact]
    public async Task ConcurrentDisposeIsSafeAndSubsequentUseIsRejected()
    {
        using var detector = CreateDetector(out _, out _);
        Task[] pushes = Enumerable.Range(0, 8)
            .Select(index => Task.Run(() =>
            {
                try
                {
                    detector.Push(CreateFrame(320, 1, 16000, startSample: index * 320), null, false);
                }
                catch (ObjectDisposedException)
                {
                }
            }))
            .ToArray();

        Task dispose = Task.Run(detector.Dispose);
        await Task.WhenAll(pushes.Append(dispose));
        Assert.Throws<ObjectDisposedException>(() => detector.GetSnapshot());
    }

    private static RenderAwarePlaybackBargeInDetector CreateDetector(
        out FakeDtd dtd,
        out FakeDelayEstimator estimator,
        VoiceAssistantBargeInOptions? options = null)
    {
        dtd = new FakeDtd();
        estimator = new FakeDelayEstimator();
        options ??= new VoiceAssistantBargeInOptions();
        options.Dtd.DecisionHistoryMs = 1000;
        return new RenderAwarePlaybackBargeInDetector(
            options.Dtd,
            options.RenderAlignment,
            dtd,
            estimator);
    }

    private static void PushFrames(
        RenderAwarePlaybackBargeInDetector detector,
        int count,
        int frameCount,
        uint channels,
        uint sampleRate,
        RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
        float afterAecScale = 1f,
        bool decisionEnabled = true,
        bool echoLearningEnabled = true,
        float playbackGain = 1f,
        bool renderCoherent = true)
    {
        long start = 0;
        for (int i = 0; i < count; i++)
        {
            detector.Push(
                CreateFrame(frameCount, channels, sampleRate, startSample: start, renderStatus: renderStatus, afterAecScale: afterAecScale, playbackGain: playbackGain, renderCoherent: renderCoherent),
                primaryAecDelay: null,
                decisionEnabled,
                echoLearningEnabled);
            start += frameCount * (long)channels;
        }
    }

    private static void PushContinuous(
        RenderAwarePlaybackBargeInDetector detector,
        int count,
        int frameCount,
        uint channels,
        uint sampleRate,
        ref long startSample,
        RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
        float afterAecScale = 1f,
        bool decisionEnabled = true,
        bool echoLearningEnabled = true,
        float playbackGain = 1f,
        bool renderCoherent = true)
    {
        for (int i = 0; i < count; i++)
        {
            detector.Push(
                CreateFrame(
                    frameCount,
                    channels,
                    sampleRate,
                    startSample: startSample,
                    renderStatus: renderStatus,
                    afterAecScale: afterAecScale,
                    playbackGain: playbackGain,
                    renderCoherent: renderCoherent),
                primaryAecDelay: null,
                decisionEnabled,
                echoLearningEnabled);
            startSample += frameCount * (long)channels;
        }
    }

    private static void ConfigurePureEcho(FakeDtd dtd)
    {
        dtd.Probability = 0.20f;
        dtd.SmoothedProbability = 0.20f;
        dtd.IsNearEndSpeech = false;
    }

    private static DuplexAudioFrame CreateFrame(
        int frameCount,
        uint channels,
        uint sampleRate,
        long epoch = 1,
        long startSample = 0,
        RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
        float afterAecScale = 1f,
        float playbackGain = 1f,
        bool renderCoherent = true)
    {
        int length = checked(frameCount * (int)channels);
        float[] microphone = new float[length];
        float[] render = new float[length];
        float[] afterAec = new float[length];
        for (int i = 0; i < length; i++)
        {
            microphone[i] = renderCoherent
                ? 0.2f
                : 0.2f * MathF.Sin(2f * MathF.PI * 317f * (startSample + i) / sampleRate);
            render[i] = renderCoherent
                ? 0.05f
                : 0.05f * MathF.Sin(2f * MathF.PI * 971f * (startSample + i) / sampleRate);
            afterAec[i] = microphone[i] * afterAecScale;
        }

        var range = new CaptureRange(epoch, startSample / Math.Max(1, length), startSample / Math.Max(1, length), startSample, startSample + length, 1);
        var captureTimeline = CaptureSignalTimeline.FromWholeBuffer(in range, length);
        var renderTimeline = new RenderSignalTimeline(epoch, startSample, startSample + length, renderStatus);
        return new DuplexAudioFrame(
            SequenceNumber: range.StartSequenceNumber,
            TimestampTicks: 1,
            SampleRate: sampleRate,
            Channels: channels,
            FrameCount: (uint)frameCount,
            CaptureProcessed: true,
            RenderReference: render,
            CaptureRaw: microphone,
            CaptureBeforeAec: microphone,
            CaptureAfterAec: afterAec,
            StreamEpoch: epoch,
            StartSample: startSample,
            HasDiscontinuity: false,
            Timeline: new DuplexCaptureTimeline(captureTimeline, captureTimeline, captureTimeline),
            RenderTimeline: renderTimeline,
            PlaybackGain: playbackGain);
    }

    private sealed class FakeDtd : IRenderAwareDtdService
    {
        private int _processSinceReset;

        public int ProcessCount { get; private set; }
        public int ResetCount { get; private set; }
        public bool ThrowOnProcess { get; set; }
        public float Probability { get; set; } = 0.9f;
        public float SmoothedProbability { get; set; } = 0.9f;
        public bool IsNearEndSpeech { get; set; } = true;
        public int HopLength => 320;
        public int WindowLength => 640;
        public uint SampleRate => 16000;
        public uint ChannelCount => 1;

        public RenderAwareDtdObservation? Process(ReadOnlySpan<float> microphone, ReadOnlySpan<float> render, uint channels, uint sampleRate)
        {
            ProcessCount++;
            _processSinceReset++;
            if (ThrowOnProcess)
                throw new InvalidOperationException("synthetic model failure");
            if (_processSinceReset == 1)
                return null;
            return new RenderAwareDtdObservation(
                Probability,
                SmoothedProbability,
                IsNearEndSpeech,
                FrameIndex: ProcessCount - 1,
                Timestamp: TimeSpan.FromMilliseconds((ProcessCount - 1) * 20),
                FrameDuration: TimeSpan.FromMilliseconds(20));
        }

        public void Reset()
        {
            ResetCount++;
            _processSinceReset = 0;
        }
        public void Dispose() { }
    }

    private sealed class FakeDelayEstimator : IRenderDelayEstimator
    {
        public int ResetCount { get; private set; }

        public RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
            => new(true, 0, 1, IsRefined: true);

        public void Reset() => ResetCount++;
        public void Dispose() { }
    }

    private sealed class SequenceDelayEstimator : IRenderDelayEstimator
    {
        private int _calls;

        public RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
        {
            _calls++;
            return _calls < 4
                ? default
                : new RenderDelayEstimate(
                    HasEstimate: true,
                    DelaySamples: 2880,
                    Confidence: 0.9,
                    IsCorrelationFallback: true);
        }

        public void Reset() => _calls = 0;

        public void Dispose() { }
    }

    private sealed class FixedDelayEstimator(double delaySamples) : IRenderDelayEstimator
    {
        public RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
            => new(true, delaySamples, 1, IsRefined: true);

        public void Reset() { }

        public void Dispose() { }
    }

    private sealed class RefinedThenFallbackDelayEstimator : IRenderDelayEstimator
    {
        private int _calls;

        public RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
        {
            _calls++;
            return _calls <= 2
                ? new RenderDelayEstimate(true, 0, 0.95, IsRefined: true)
                : new RenderDelayEstimate(
                    true,
                    640,
                    0.90,
                    IsCorrelationFallback: true);
        }

        public void Reset() => _calls = 0;

        public void Dispose() { }
    }
}
