using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Modules.Speaker.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Direct contracts for the state owners owned by <c>ActiveNearEndCoordinator</c>.
/// The coordinator still drives them under its single data-plane lock; these tests pin the
/// decision edges that the integration tests only exercise indirectly.
/// </summary>
public sealed class NearEndStateOwnershipTests
{
    [Fact]
    public void CaptureSession_ForceEndDrainIsIgnoredOnceFinalizationIsClaimed()
    {
        var session = NewDrainingSession(targetSample: 100, fedThrough: 100);

        Assert.True(session.TryClaimEndFinalization(out _, out _));
        session.ForceEndDrain("late-force");

        Assert.True(session.EndFinalizationClaimed);
        Assert.False(session.EndDrainForced);
        Assert.Equal("none", session.EndDrainReason);
    }

    [Fact]
    public void CaptureSession_ForceEndDrainIsIgnoredWhenNoDrainIsPending()
    {
        var session = new NearEndCaptureSession();

        session.ForceEndDrain("no-drain");

        Assert.False(session.EndDrainForced);
        Assert.Equal("none", session.EndDrainReason);
    }

    [Fact]
    public void CaptureSession_FinalizationReportsFedThroughAndMissingSamples()
    {
        var session = NewDrainingSession(targetSample: 800, fedThrough: 500);
        session.ForceEndDrain("timeout");

        Assert.True(session.TryClaimEndFinalization(
            out TimeSpan speechDuration,
            out ActiveEndDrainOutcome outcome));

        Assert.Equal(TimeSpan.FromMilliseconds(40), speechDuration);
        Assert.Equal("timeout", outcome.Reason);
        Assert.Equal(800, outcome.TargetSample);
        Assert.Equal(500, outcome.FedThroughSample);
        Assert.Equal(300, outcome.MissingSamples);
    }

    [Fact]
    public void CaptureSession_FinalizationClampsFedThroughAndNeverReportsNegativeMissing()
    {
        var session = NewDrainingSession(targetSample: 800, fedThrough: 950);

        Assert.True(session.TryClaimEndFinalization(out _, out ActiveEndDrainOutcome outcome));

        Assert.Equal(800, outcome.FedThroughSample);
        Assert.Equal(0, outcome.MissingSamples);
    }

    [Fact]
    public void CaptureSession_FinalizationWaitsForInFlightFeedsAndTheTargetWatermark()
    {
        var pending = NewDrainingSession(targetSample: 800, fedThrough: 500);
        Assert.False(pending.TryClaimEndFinalization(out _, out _));

        var inFlight = NewDrainingSession(targetSample: 800, fedThrough: 800);
        inFlight.DirectFeedsInFlight = 1;
        Assert.False(inFlight.TryClaimEndFinalization(out _, out _));

        inFlight.DirectFeedsInFlight = 0;
        Assert.True(inFlight.TryClaimEndFinalization(out _, out _));
        Assert.False(inFlight.TryClaimEndFinalization(out _, out _));
    }

    [Fact]
    public void CaptureSession_ClearEndDrainRestoresEveryDrainField()
    {
        var session = NewDrainingSession(targetSample: 800, fedThrough: 800);
        session.ForceEndDrain("forced");
        Assert.True(session.TryClaimEndFinalization(out _, out _));

        session.ClearEndDrain();

        Assert.False(session.EndDrainPending);
        Assert.False(session.EndDrainForced);
        Assert.False(session.EndFinalizationClaimed);
        Assert.Equal("none", session.EndDrainReason);
        Assert.Equal(0, session.EndDrainStreamEpoch);
        Assert.Equal(0, session.EndDrainTargetSample);
        Assert.Equal(0, session.EndDrainDeadlineSample);
        Assert.Equal(TimeSpan.Zero, session.EndDrainSpeechDuration);
    }

    [Theory]
    [InlineData(5L, 0L, 5L)]
    [InlineData(5L, -3L, 2L)]
    [InlineData(long.MaxValue - 10, 10L, long.MaxValue)]
    [InlineData(long.MaxValue - 10, 11L, long.MaxValue)]
    public void CaptureSession_SaturatingAddOnlyClampsOnPositiveOverflow(
        long left,
        long right,
        long expected)
        => Assert.Equal(expected, NearEndCaptureSession.SaturatingAdd(left, right));

    [Fact]
    public void CaptureSession_VolatileActiveTurnReadObservesTheAssignedTurn()
    {
        var session = new NearEndCaptureSession { ActiveTurnId = 42 };

        Assert.Equal(42, session.ReadActiveTurnIdVolatile());
        Assert.Equal(42, session.ActiveTurnId);
    }

    [Fact]
    public void EndpointPolicy_ClearCandidateResetsEveryCandidateField()
    {
        var policy = new NearEndEndpointPolicy(new NearEndAudioReservoir(1024));
        policy.BeginCandidate(mainAudioGeneration: 7, mainAudioStreamEpoch: 3);
        policy.ExtendCandidate(streamEpoch: 3, startSample: 100, endSample: 200);

        policy.ClearCandidate();

        Assert.False(policy.CandidateActive);
        Assert.False(policy.CandidateLocated);
        Assert.False(policy.CandidateLocationAttempted);
        Assert.Equal(0, policy.CandidateGeneration);
        Assert.Equal(0, policy.CandidateStreamEpoch);
        Assert.Equal(0, policy.CandidateStartSample);
        Assert.Equal(0, policy.CandidateEndSample);
    }

    [Fact]
    public void EndpointPolicy_ClearUnownedResetsEveryUnownedField()
    {
        var policy = new NearEndEndpointPolicy(new NearEndAudioReservoir(1024));
        policy.RecordUnowned(generation: 4, streamEpoch: 9, endSample: 512);
        policy.UnownedProcessed = true;

        policy.ClearUnowned();

        Assert.False(policy.UnownedObserved);
        Assert.False(policy.UnownedProcessed);
        Assert.Equal(0, policy.UnownedGeneration);
        Assert.Equal(0, policy.UnownedStreamEpoch);
        Assert.Equal(0, policy.UnownedEndSample);
    }

    [Fact]
    public void EndpointPolicy_ExtendCandidateKeepsTheEarliestStartAndLatestEnd()
    {
        var policy = new NearEndEndpointPolicy(new NearEndAudioReservoir(1024));
        policy.BeginCandidate(mainAudioGeneration: 1, mainAudioStreamEpoch: 2);

        policy.ExtendCandidate(streamEpoch: 2, startSample: 500, endSample: 600);
        Assert.True(policy.CandidateLocated);
        Assert.Equal(500, policy.CandidateStartSample);
        Assert.Equal(600, policy.CandidateEndSample);

        // An earlier located chunk must widen the interval backwards, and a later one forwards.
        policy.ExtendCandidate(streamEpoch: 2, startSample: 300, endSample: 550);
        Assert.Equal(300, policy.CandidateStartSample);
        Assert.Equal(600, policy.CandidateEndSample);

        policy.ExtendCandidate(streamEpoch: 2, startSample: 700, endSample: 900);
        Assert.Equal(300, policy.CandidateStartSample);
        Assert.Equal(900, policy.CandidateEndSample);
    }

    [Fact]
    public void EndpointPolicy_TryLocateRecordsTheAttemptEvenWhenTheChunkIsNotFound()
    {
        var policy = new NearEndEndpointPolicy(new NearEndAudioReservoir(1024));

        bool located = policy.TryLocate(
            new float[] { 0.5f, 0.25f },
            out _,
            out _,
            out _);

        Assert.False(located);
        Assert.True(policy.CandidateLocationAttempted);
    }

    [Fact]
    public void SpeakerCandidate_BeginAdvancesTheGenerationAndPinsTheWindow()
    {
        var state = new NearEndSpeakerCandidateState();
        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200) with { Action = BargeInAction.Continue };

        state.Begin(in decision, new SpeakerCandidateId(9), ownerTurnId: 3, anchorSample: 120, endSampleExclusive: 260, deadlineSample: 900, strongUpstreamEvidence: true);

        Assert.True(state.Active);
        Assert.False(state.Authorized);
        Assert.True(state.IdentityRequired);
        Assert.Equal(new SpeakerCandidateId(9), state.CandidateId);
        Assert.Equal(3, state.OwnerTurnId);
        Assert.Equal(1, state.Generation);
        Assert.Equal(5, state.StreamEpoch);
        Assert.Equal(120, state.MainStartSample);
        Assert.Equal(260, state.EndSample);
        Assert.Equal(900, state.DeadlineSample);
        Assert.Equal(-1, state.InferenceReadyTimestamp);
        Assert.True(state.StrongUpstreamEvidence);
        // The candidate opened on valid near-end speech, before any DTD Interrupt.
        Assert.False(state.DtdInterruptObserved);

        state.Clear();
        state.Begin(in decision, new SpeakerCandidateId(10), 3, 0, 0, 0, strongUpstreamEvidence: false);
        Assert.Equal(2, state.Generation);
        Assert.True(state.IsCurrent(2));
        Assert.False(state.IsCurrent(1));
    }

    [Fact]
    public void SpeakerCandidate_DtdInterruptOnlyAddsAcousticEvidenceAndNeverAuthorizes()
    {
        var state = new NearEndSpeakerCandidateState();
        BargeInDecision opening = Decision(streamEpoch: 5, start: 100, end: 200) with { Action = BargeInAction.Continue };
        state.Begin(in opening, new SpeakerCandidateId(1), 3, 100, 200, 400, strongUpstreamEvidence: false);

        BargeInDecision interrupt = Decision(streamEpoch: 5, start: 100, end: 320);
        state.ObserveDecision(in interrupt, strongNearEnd: true);

        Assert.True(state.DtdInterruptObserved);
        Assert.True(state.StrongUpstreamEvidence);
        Assert.Equal(320, state.Decision.EndSampleExclusive);
        Assert.False(state.Authorized);

        // A decision from another stream epoch is ignored.
        BargeInDecision otherEpoch = Decision(streamEpoch: 6, start: 0, end: 50);
        state.ObserveDecision(in otherEpoch, strongNearEnd: true);
        Assert.Equal(5, state.Decision.StreamEpoch);
    }

    [Fact]
    public void SpeakerCandidate_ClearResetsEveryFieldButKeepsTheGeneration()
    {
        var state = new NearEndSpeakerCandidateState();
        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200);
        state.Begin(in decision, new SpeakerCandidateId(4), 2, 120, 260, 900, strongUpstreamEvidence: true);
        state.Authorize();
        state.InferenceReadyTimestamp = 1234;
        state.EndpointObserved = true;

        state.Clear();

        Assert.False(state.Active);
        Assert.False(state.Authorized);
        Assert.False(state.IdentityRequired);
        Assert.Equal(default, state.CandidateId);
        Assert.Equal(0, state.OwnerTurnId);
        Assert.Equal(default, state.Decision);
        Assert.Equal(0, state.StreamEpoch);
        Assert.Equal(-1, state.MainStartSample);
        Assert.Equal(0, state.EndSample);
        Assert.Equal(0, state.DeadlineSample);
        Assert.Equal(-1, state.InferenceReadyTimestamp);
        Assert.False(state.StrongUpstreamEvidence);
        Assert.False(state.DtdInterruptObserved);
        Assert.False(state.EndpointObserved);
        Assert.Equal(1, state.Generation);
        Assert.False(state.IsCurrent(1));
    }

    [Fact]
    public void SpeakerCandidate_PreAuthorizedTimelineIsTheKwsDisabledPathWithoutIdentity()
    {
        var state = new NearEndSpeakerCandidateState();
        BargeInDecision decision = Decision(streamEpoch: 7, start: 300, end: 480);

        state.BeginPreAuthorizedTimeline(in decision, ownerTurnId: 5);

        Assert.True(state.Active);
        Assert.True(state.Authorized);
        Assert.False(state.IdentityRequired);
        Assert.Equal(default, state.CandidateId);
        Assert.Equal(5, state.OwnerTurnId);
        Assert.Equal(7, state.StreamEpoch);
        Assert.Equal(-1, state.MainStartSample);
        Assert.Equal(480, state.EndSample);
        Assert.True(state.DtdInterruptObserved);
        Assert.True(state.IsCurrent(state.Generation));
    }

    private static NearEndCaptureSession NewDrainingSession(long targetSample, long fedThrough)
    {
        var session = new NearEndCaptureSession
        {
            ActiveTurnId = 1,
            DirectFeed = true,
            DirectFeedNextSample = fedThrough,
        };
        session.BeginEndDrain(
            streamEpoch: 3,
            endSampleExclusive: targetSample,
            speechDuration: TimeSpan.FromMilliseconds(40),
            timeoutSamples: 160);
        return session;
    }

    private static BargeInDecision Decision(long streamEpoch, long start, long end)
        => new(
            Action: BargeInAction.Interrupt,
            FrameCount: 10,
            AcceptedFrameCount: 8,
            RejectedFrameCount: 2,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 200,
            AcceptedDurationMs: 160,
            RequiredDurationMs: 100,
            ScoreDurationMs: 120,
            PeakScoreDurationMs: 120,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.05,
            MinRequiredRms: 0.05,
            MaxRequiredRms: 0.05,
            Reason: "test",
            StreamEpoch: streamEpoch,
            StartSample: start,
            EndSampleExclusive: end);
}
