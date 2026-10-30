using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Independent behavioral contracts for the near-end state owners. The data-plane integration
/// tests only exercise them through ports; these tests pin the state transitions and boundaries
/// each owner is responsible for.
/// </summary>
public sealed class NearEndStateOwnershipTests
{
    // -----------------------------------------------------------------------------------
    // NearEndCaptureStateMachine
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Capture_ForceEndDrainIsIgnoredOnceFinalizationIsClaimed()
    {
        NearEndCaptureStateMachine capture = NewDrainingCapture(targetSample: 100, fedThrough: 50);
        capture.ForceEndDrain("timeout");
        Assert.True(capture.TryClaimEndFinalization(out _, out _));

        capture.ForceEndDrain("late-force");

        Assert.True(capture.EndFinalizationClaimed);
        Assert.True(capture.EndDrainForced);
        Assert.Equal("timeout", capture.EndDrainReason);
    }

    [Fact]
    public void Capture_ForceEndDrainIsIgnoredWhenNoDrainIsPending()
    {
        var capture = new NearEndCaptureStateMachine();

        capture.ForceEndDrain("no-drain");

        Assert.False(capture.EndDrainForced);
        Assert.Equal("none", capture.EndDrainReason);
    }

    [Fact]
    public void Capture_FinalizationReportsFedThroughAndMissingSamples()
    {
        NearEndCaptureStateMachine capture = NewDrainingCapture(targetSample: 800, fedThrough: 500);
        capture.ForceEndDrain("timeout");

        Assert.True(capture.TryClaimEndFinalization(
            out TimeSpan speechDuration,
            out ActiveEndDrainOutcome outcome));

        Assert.Equal(TimeSpan.FromMilliseconds(40), speechDuration);
        Assert.Equal("timeout", outcome.Reason);
        Assert.Equal(800, outcome.TargetSample);
        Assert.Equal(500, outcome.FedThroughSample);
        Assert.Equal(300, outcome.MissingSamples);
    }

    [Fact]
    public void Capture_FinalizationClampsFedThroughAndNeverReportsNegativeMissing()
    {
        NearEndCaptureStateMachine capture = NewDrainingCapture(targetSample: 800, fedThrough: 950);

        Assert.True(capture.TryClaimEndFinalization(out _, out ActiveEndDrainOutcome outcome));

        Assert.Equal(800, outcome.FedThroughSample);
        Assert.Equal(0, outcome.MissingSamples);
    }

    [Fact]
    public void Capture_FinalizationWaitsForInFlightFeedsAndTheTargetWatermark()
    {
        NearEndCaptureStateMachine pending = NewDrainingCapture(targetSample: 800, fedThrough: 500);
        Assert.False(pending.TryClaimEndFinalization(out _, out _));

        var inFlight = new NearEndCaptureStateMachine();
        inFlight.Start(turnId: 1, firstFeedSample: 700, speechDurationMs: 40, "test", 0);
        inFlight.BeginDrain(streamEpoch: 3, endSampleExclusive: 800, false, timeoutSamples: 160);
        NearEndCaptureStateMachine.FeedDecision reserved = inFlight.ReserveFeed(
            new MainAudioObservation(1, 3, 700, 800, false),
            currentGeneration: 1,
            currentStreamEpoch: 3);
        Assert.True(reserved.Reservation.IsValid);
        Assert.False(inFlight.TryClaimEndFinalization(out _, out _));

        inFlight.CompleteFeed(1, reserved.Reservation, failed: false);
        Assert.True(inFlight.TryClaimEndFinalization(out _, out _));
        Assert.False(inFlight.TryClaimEndFinalization(out _, out _));
    }

    [Fact]
    public void Capture_ClearEndDrainRestoresEveryDrainFieldAndReturnsToDirectFeed()
    {
        NearEndCaptureStateMachine capture = NewDrainingCapture(targetSample: 800, fedThrough: 800);
        capture.ForceEndDrain("forced");
        Assert.True(capture.TryClaimEndFinalization(out _, out _));

        capture.ClearEndDrain();

        Assert.False(capture.EndDrainPending);
        Assert.False(capture.EndDrainForced);
        Assert.False(capture.EndFinalizationClaimed);
        Assert.Equal("none", capture.EndDrainReason);
        Assert.Equal(0, capture.EndDrainStreamEpoch);
        Assert.Equal(0, capture.EndDrainTargetSample);
        Assert.Equal(0, capture.EndDrainDeadlineSample);
        Assert.Equal(NearEndCaptureStateMachine.Status.DirectFeed, capture.State);
    }

    [Theory]
    [InlineData(5L, 0L, 5L)]
    [InlineData(5L, -3L, 2L)]
    [InlineData(long.MaxValue - 10, 10L, long.MaxValue)]
    [InlineData(long.MaxValue - 10, 11L, long.MaxValue)]
    public void Capture_SaturatingAddOnlyClampsOnPositiveOverflow(
        long left,
        long right,
        long expected)
        => Assert.Equal(expected, NearEndCaptureStateMachine.SaturatingAdd(left, right));

    [Fact]
    public void Capture_VolatileActiveTurnReadObservesTheStartedTurn()
    {
        var capture = new NearEndCaptureStateMachine();
        capture.Start(turnId: 42, firstFeedSample: 0, speechDurationMs: 0, "test", 0);

        Assert.Equal(42, capture.ReadActiveTurnIdVolatile());
        Assert.Equal(42, capture.ActiveTurnId);
        Assert.True(capture.DirectFeed);
    }

    [Fact]
    public void Capture_StopResetsTheTurnButKeepsInFlightFeedAccounting()
    {
        var capture = new NearEndCaptureStateMachine();
        capture.Start(turnId: 1, firstFeedSample: 0, speechDurationMs: 0, "test", 0);
        NearEndCaptureStateMachine.FeedDecision reserved = capture.ReserveFeed(
            new MainAudioObservation(1, 1, 0, 4, false),
            currentGeneration: 1,
            currentStreamEpoch: 1);
        Assert.True(reserved.Reservation.IsValid);
        Assert.Equal(1, capture.DirectFeedsInFlight);

        capture.Stop();

        Assert.Equal(NearEndCaptureStateMachine.Status.Idle, capture.State);
        Assert.Equal(0, capture.ActiveTurnId);
        Assert.Equal(1, capture.DirectFeedsInFlight);
        Assert.Null(Record.Exception(() =>
        {
            capture.CompleteFeed(1, reserved.Reservation, failed: false);
            Assert.Equal(0, capture.DirectFeedsInFlight);
        }));
    }

    [Fact]
    public void Capture_ObserveAudioFrameIgnoresAnotherTurnOrIdleCapture()
    {
        var capture = new NearEndCaptureStateMachine();
        Assert.False(capture.ObserveAudioFrame(
            1, 1, 10, false, 1, latestSpeechActive: false, endSilenceMs: 1, timeoutSamples: 10).Claimed);
        Assert.Equal(0, capture.SpeechDurationMs);

        capture.Start(turnId: 1, firstFeedSample: 0, speechDurationMs: 0, "test", 0);
        Assert.False(capture.ObserveAudioFrame(
            2, 1, 10, false, 1, latestSpeechActive: false, endSilenceMs: 1, timeoutSamples: 10).Claimed);
        Assert.Equal(0, capture.SpeechDurationMs);
    }

    [Fact]
    public void Capture_ObserveAudioFrameForcesDrainOnEpochChangeOrDeadline()
    {
        NearEndCaptureStateMachine epochChanged = NewDrainingCapture(targetSample: 10, fedThrough: 10);
        _ = epochChanged.ObserveAudioFrame(
            1, streamEpoch: 2, endSampleExclusive: 1, hasDiscontinuity: false,
            frameMs: 1, latestSpeechActive: false, endSilenceMs: 1, timeoutSamples: 10);
        Assert.True(epochChanged.EndDrainForced);
        Assert.Equal("capture-discontinuity", epochChanged.EndDrainReason);

        var timedOut = new NearEndCaptureStateMachine();
        timedOut.Start(turnId: 1, firstFeedSample: 10, speechDurationMs: 40, "test", 0);
        timedOut.BeginDrain(streamEpoch: 3, endSampleExclusive: 10, hasDiscontinuity: false, timeoutSamples: 0);
        _ = timedOut.ObserveAudioFrame(
            1, streamEpoch: 3, endSampleExclusive: 10, hasDiscontinuity: false,
            frameMs: 1, latestSpeechActive: false, endSilenceMs: 1, timeoutSamples: 10);
        Assert.True(timedOut.EndDrainForced);
        Assert.Equal("drain-timeout", timedOut.EndDrainReason);
    }

    [Fact]
    public void Capture_ObserveAudioFrameBeginsDrainAfterSilenceThreshold()
    {
        var capture = new NearEndCaptureStateMachine();
        capture.Start(turnId: 1, firstFeedSample: 0, speechDurationMs: 0, "test", 0);

        NearEndCaptureStateMachine.Finalization first = capture.ObserveAudioFrame(
            1, 1, 5, false, frameMs: 2, latestSpeechActive: true, endSilenceMs: 4, timeoutSamples: 10);
        Assert.False(first.Claimed);
        Assert.Equal(2, capture.SpeechDurationMs);

        NearEndCaptureStateMachine.Finalization second = capture.ObserveAudioFrame(
            1, 1, 10, false, frameMs: 4, latestSpeechActive: false, endSilenceMs: 4, timeoutSamples: 10);
        Assert.True(capture.EndDrainPending);
        Assert.Equal(10, capture.EndDrainTargetSample);
        // The drain is pending, but finalization must wait for the fed watermark to reach it.
        Assert.False(second.Claimed);
    }

    [Fact]
    public void Capture_ReserveFeedClipsAtTheDrainTargetAndReleasesTheWatermarkOnFailure()
    {
        var capture = new NearEndCaptureStateMachine();
        capture.Start(turnId: 1, firstFeedSample: 0, speechDurationMs: 0, "test", 0);
        capture.BeginDrain(streamEpoch: 1, endSampleExclusive: 10, false, timeoutSamples: 100);

        NearEndCaptureStateMachine.FeedDecision reserved = capture.ReserveFeed(
            new MainAudioObservation(1, 1, 8, 20, false),
            currentGeneration: 1,
            currentStreamEpoch: 1);

        Assert.True(reserved.Handled);
        Assert.Equal(8, reserved.Reservation.StartSample);
        Assert.Equal(10, reserved.Reservation.EndSampleExclusive);
        Assert.Equal(10, capture.DirectFeedNextSample);
        Assert.Equal("aligned", capture.EndDrainReason);

        capture.CompleteFeed(1, reserved.Reservation, failed: true);

        Assert.Equal(8, capture.DirectFeedNextSample);
        Assert.True(capture.EndDrainForced);
        Assert.Equal("asr-feed-failed", capture.EndDrainReason);
    }

    // -----------------------------------------------------------------------------------
    // NearEndEndpointTracker
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Endpoint_ClearCandidateResetsEveryCandidateField()
    {
        var tracker = new NearEndEndpointTracker(new NearEndTimelineBuffer(1024));
        tracker.BeginCandidate();
        tracker.ExtendCandidate(streamEpoch: 3, startSample: 100, endSample: 200);

        tracker.ClearCandidate();

        Assert.False(tracker.CandidateActive);
        Assert.False(tracker.CandidateLocated);
        Assert.False(tracker.CandidateLocationAttempted);
        Assert.Equal(0, tracker.CandidateGeneration);
        Assert.Equal(0, tracker.CandidateStreamEpoch);
        Assert.Equal(0, tracker.CandidateStartSample);
        Assert.Equal(0, tracker.CandidateEndSample);
    }

    [Fact]
    public void Endpoint_ClearUnownedResetsEveryUnownedField()
    {
        var tracker = new NearEndEndpointTracker(new NearEndTimelineBuffer(1024));
        tracker.RecordUnowned(generation: 4, streamEpoch: 9, endSample: 512);
        tracker.MarkUnownedProcessed();

        tracker.ClearUnowned();

        Assert.False(tracker.UnownedObserved);
        Assert.False(tracker.UnownedProcessed);
        Assert.Equal(0, tracker.UnownedGeneration);
        Assert.Equal(0, tracker.UnownedStreamEpoch);
        Assert.Equal(0, tracker.UnownedEndSample);
    }

    [Fact]
    public void Endpoint_ExtendCandidateKeepsTheEarliestStartAndLatestEnd()
    {
        var tracker = new NearEndEndpointTracker(new NearEndTimelineBuffer(1024));
        tracker.BeginCandidate();

        tracker.ExtendCandidate(streamEpoch: 2, startSample: 500, endSample: 600);
        Assert.True(tracker.CandidateLocated);
        Assert.Equal(500, tracker.CandidateStartSample);
        Assert.Equal(600, tracker.CandidateEndSample);

        tracker.ExtendCandidate(streamEpoch: 2, startSample: 300, endSample: 550);
        Assert.Equal(300, tracker.CandidateStartSample);
        Assert.Equal(600, tracker.CandidateEndSample);

        tracker.ExtendCandidate(streamEpoch: 2, startSample: 700, endSample: 900);
        Assert.Equal(300, tracker.CandidateStartSample);
        Assert.Equal(900, tracker.CandidateEndSample);
    }

    [Fact]
    public void Endpoint_TryLocateRecordsTheAttemptEvenWhenTheChunkIsNotFound()
    {
        var tracker = new NearEndEndpointTracker(new NearEndTimelineBuffer(1024));

        bool located = tracker.TryLocate(
            new float[] { 0.5f, 0.25f },
            out _,
            out _,
            out _);

        Assert.False(located);
        Assert.True(tracker.CandidateLocationAttempted);
    }

    [Fact]
    public void Endpoint_CaptureObservationTransitionsAreExplicit()
    {
        var tracker = new NearEndEndpointTracker(new NearEndTimelineBuffer(1024));

        tracker.ObserveCaptureStarted(endpointVadCandidate: true);
        Assert.True(tracker.CaptureObserved);
        Assert.True(tracker.CaptureSpeechActive);

        tracker.MarkCaptureSpeechEnded();
        Assert.True(tracker.CaptureObserved);
        Assert.False(tracker.CaptureSpeechActive);

        tracker.MarkCaptureSpeechActive();
        Assert.True(tracker.CaptureSpeechActive);

        tracker.ClearCaptureObservation();
        Assert.False(tracker.CaptureObserved);
        Assert.False(tracker.CaptureSpeechActive);
    }

    // -----------------------------------------------------------------------------------
    // NearEndCandidateStateMachine
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Candidate_BeginAdvancesTheGenerationAndPinsTheWindow()
    {
        var candidate = new NearEndCandidateStateMachine();
        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200)
            with { Action = BargeInAction.Continue };

        candidate.BeginSpeaker(
            in decision,
            new SpeakerCandidateId(9),
            ownerTurnId: 3,
            anchorSample: 120,
            endSampleExclusive: 260,
            deadlineSample: 900,
            strongUpstreamEvidence: true);

        Assert.True(candidate.Active);
        Assert.False(candidate.Authorized);
        Assert.True(candidate.IdentityRequired);
        Assert.Equal(new SpeakerCandidateId(9), candidate.CandidateId);
        Assert.Equal(3, candidate.OwnerTurnId);
        Assert.Equal(1, candidate.Generation);
        Assert.Equal(5, candidate.StreamEpoch);
        Assert.Equal(120, candidate.MainStartSample);
        Assert.Equal(260, candidate.EndSample);
        Assert.Equal(900, candidate.DeadlineSample);
        Assert.Equal(-1, candidate.InferenceReadyTimestamp);
        Assert.True(candidate.StrongUpstreamEvidence);
        Assert.False(candidate.DtdInterruptObserved);

        candidate.Clear();
        candidate.BeginSpeaker(in decision, new SpeakerCandidateId(10), 3, 0, 0, 0, false);
        Assert.Equal(2, candidate.Generation);
        Assert.True(candidate.IsCurrent(2));
        Assert.False(candidate.IsCurrent(1));
    }

    [Fact]
    public void Candidate_DtdInterruptOnlyAddsAcousticEvidenceAndNeverAuthorizes()
    {
        var candidate = new NearEndCandidateStateMachine();
        BargeInDecision opening = Decision(streamEpoch: 5, start: 100, end: 200)
            with { Action = BargeInAction.Continue };
        candidate.BeginSpeaker(in opening, new SpeakerCandidateId(1), 3, 100, 200, 400, false);

        BargeInDecision interrupt = Decision(streamEpoch: 5, start: 100, end: 320);
        candidate.ObserveDecision(in interrupt, strongNearEnd: true);

        Assert.True(candidate.DtdInterruptObserved);
        Assert.True(candidate.StrongUpstreamEvidence);
        Assert.Equal(320, candidate.Decision.EndSampleExclusive);
        Assert.False(candidate.Authorized);

        BargeInDecision otherEpoch = Decision(streamEpoch: 6, start: 0, end: 50);
        candidate.ObserveDecision(in otherEpoch, strongNearEnd: true);
        Assert.Equal(5, candidate.Decision.StreamEpoch);
    }

    [Fact]
    public void Candidate_ClearResetsEveryFieldButKeepsTheGeneration()
    {
        var candidate = new NearEndCandidateStateMachine();
        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200);
        candidate.BeginSpeaker(in decision, new SpeakerCandidateId(4), 2, 120, 260, 900, true);
        candidate.Authorize();
        var pending = new SpeakerVerificationResult(
            SpeakerVerificationStatus.Pending, 0, 0, 5, 120, 260,
            SpeakerVerificationReason.WaitingForInference);
        candidate.ObserveInferenceWait(in pending, 1234);
        candidate.MarkEndpointObserved(1_000);

        candidate.Clear();

        Assert.False(candidate.Active);
        Assert.False(candidate.Authorized);
        Assert.False(candidate.IdentityRequired);
        Assert.Equal(default, candidate.CandidateId);
        Assert.Equal(0, candidate.OwnerTurnId);
        Assert.Equal(default, candidate.Decision);
        Assert.Equal(0, candidate.StreamEpoch);
        Assert.Equal(-1, candidate.MainStartSample);
        Assert.Equal(0, candidate.EndSample);
        Assert.Equal(0, candidate.DeadlineSample);
        Assert.Equal(-1, candidate.InferenceReadyTimestamp);
        Assert.False(candidate.StrongUpstreamEvidence);
        Assert.False(candidate.DtdInterruptObserved);
        Assert.False(candidate.EndpointObserved);
        Assert.Equal(1, candidate.Generation);
        Assert.False(candidate.IsCurrent(1));
    }

    [Fact]
    public void Candidate_AuthorizedTimelineIsTheKwsDisabledPathWithoutIdentity()
    {
        var candidate = new NearEndCandidateStateMachine();
        BargeInDecision decision = Decision(streamEpoch: 7, start: 300, end: 480);

        candidate.BeginAuthorizedTimeline(in decision, ownerTurnId: 5);

        Assert.True(candidate.Active);
        Assert.True(candidate.Authorized);
        Assert.False(candidate.IdentityRequired);
        Assert.Equal(default, candidate.CandidateId);
        Assert.Equal(5, candidate.OwnerTurnId);
        Assert.Equal(7, candidate.StreamEpoch);
        Assert.Equal(-1, candidate.MainStartSample);
        Assert.Equal(480, candidate.EndSample);
        Assert.True(candidate.DtdInterruptObserved);
        Assert.True(candidate.IsCurrent(candidate.Generation));
    }

    [Fact]
    public void Candidate_EndpointObservationFreezesTheWindowAtTheDeadline()
    {
        var candidate = new NearEndCandidateStateMachine();
        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200);
        candidate.BeginSpeaker(in decision, new SpeakerCandidateId(1), 3, 100, 200, 400, false);

        candidate.MarkEndpointObserved(450);

        Assert.True(candidate.EndpointObserved);
        Assert.Equal(400, candidate.EndSample);

        candidate.MarkEndpointObserved(300);
        Assert.Equal(400, candidate.EndSample);
    }

    [Fact]
    public void Candidate_AuthorizeOnlyMovesAnAwaitingSpeakerCandidate()
    {
        var candidate = new NearEndCandidateStateMachine();
        candidate.Authorize();
        Assert.False(candidate.Active);

        BargeInDecision decision = Decision(streamEpoch: 5, start: 100, end: 200);
        candidate.BeginSpeaker(in decision, new SpeakerCandidateId(1), 3, 100, 200, 400, false);
        candidate.Authorize();

        Assert.True(candidate.Authorized);
        Assert.False(candidate.IdentityRequired == false);
    }

    // -----------------------------------------------------------------------------------
    // NearEndConfirmationJournal
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Confirmation_QueueAcquireCommitIsATransaction()
    {
        var journal = new NearEndConfirmationJournal();
        NearEndConfirmationLease lease = Lease(operationId: 7);

        Assert.True(journal.TryQueue(in lease));
        Assert.False(journal.TryQueue(in lease));
        Assert.True(journal.HasPending);

        Assert.False(journal.TryAcquire(8, out _));
        Assert.True(journal.TryAcquire(7, out NearEndConfirmationLease acquired));
        Assert.True(journal.HasTaken);
        Assert.False(journal.TryAcquire(7, out _));

        Assert.True(journal.TryCommit(in acquired));
        Assert.False(journal.HasQueued);
        Assert.False(journal.TryCommit(in acquired));
    }

    [Fact]
    public void Confirmation_StaleAbortCannotClearANewerLease()
    {
        var journal = new NearEndConfirmationJournal();
        NearEndConfirmationLease first = Lease(operationId: 1);
        Assert.True(journal.TryQueue(in first));
        Assert.True(journal.TryAcquire(1, out NearEndConfirmationLease taken));
        Assert.True(journal.TryAbort(in taken));

        NearEndConfirmationLease second = Lease(operationId: 2);
        Assert.True(journal.TryQueue(in second));
        Assert.True(journal.TryAcquire(2, out NearEndConfirmationLease current));

        NearEndConfirmationLease stale = Lease(operationId: 1);
        Assert.False(journal.TryAbort(in stale));
        Assert.True(journal.HasTaken);
        Assert.True(journal.TryAbort(in current));
        Assert.False(journal.HasQueued);
    }

    [Fact]
    public void Confirmation_ReplayRequestFollowsPendingThenTaken()
    {
        var journal = new NearEndConfirmationJournal();
        NearEndConfirmationLease lease = Lease(operationId: 3);
        Assert.True(journal.TryQueue(in lease));

        Assert.True(journal.TryGetReplayRequest(out NearEndConfirmationLease pending));
        Assert.Equal(3, pending.OperationId);
        Assert.True(journal.IsQueuedOperation(3));

        Assert.True(journal.TryAcquire(3, out _));
        Assert.True(journal.TryGetReplayRequest(out NearEndConfirmationLease taken));
        Assert.Equal(3, taken.OperationId);
        Assert.True(journal.IsQueuedOperation(3));

        Assert.True(journal.TryDiscardPending(3) == false);
        Assert.True(journal.IsQueuedOperation(3));
    }

    // -----------------------------------------------------------------------------------
    // NearEndTimelineBuffer and NearEndReplayBuffer
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Timeline_DiscontinuityAdvancesGenerationAndInvalidatesOlderTails()
    {
        var timeline = new NearEndTimelineBuffer(64);
        long start = 0;
        AddTimeline(timeline, [1f, 2f], streamEpoch: 1, ref start);
        long generation = timeline.Generation;

        AddTimeline(timeline, [3f, 4f], streamEpoch: 1, ref start, discontinuity: true);

        Assert.Equal(generation + 1, timeline.Generation);
        Assert.Equal(timeline.Generation, timeline.InvalidatedGeneration);
        Assert.Equal(4, timeline.EndSample);
        Assert.Equal(1, timeline.StreamEpoch);
    }

    [Fact]
    public void Timeline_ClearStartsANewGenerationWithoutInvalidatingTails()
    {
        var timeline = new NearEndTimelineBuffer(64);
        long start = 0;
        AddTimeline(timeline, [1f, 2f], streamEpoch: 1, ref start);
        long generation = timeline.Generation;

        timeline.Clear();

        Assert.Equal(generation + 1, timeline.Generation);
        Assert.Equal(0, timeline.EndSample);
        Assert.Equal(0, timeline.StreamEpoch);
        Assert.Equal(0, timeline.InvalidatedGeneration);
    }

    [Fact]
    public void Replay_BoundedQueueReportsOverflowAndTheDrainLeaseIsSingleConsumer()
    {
        var replay = new NearEndReplayBuffer();
        var frame = new NearEndConfirmationTailFrame(1, 1, 0, 1, false, false, 1);

        for (int index = 0; index < 256; index++)
            Assert.True(replay.TryWrite(in frame));
        Assert.False(replay.TryWrite(in frame));
        Assert.True(replay.HasFrames);

        Assert.True(replay.TryEnterDrain());
        Assert.False(replay.TryEnterDrain());
        Assert.True(replay.TryRead(out _));
        replay.ExitDrain();
        Assert.True(replay.TryEnterDrain());

        replay.Clear();
        Assert.False(replay.HasFrames);
        Assert.Equal(0, replay.NextSample);
    }

    [Fact]
    public void Replay_NextSampleOnlyAdvancesAtTheImmutableBoundary()
    {
        var replay = new NearEndReplayBuffer();
        replay.ResetNextSample(40);
        replay.AdvanceNextSample(30);
        Assert.Equal(40, replay.NextSample);
        replay.AdvanceNextSample(60);
        Assert.Equal(60, replay.NextSample);
    }

    // -----------------------------------------------------------------------------------
    // PlaybackDuckingPolicy
    // -----------------------------------------------------------------------------------

    [Fact]
    public void Ducking_DucksRecentSpeechAndRestoresAfterTheReleaseWindow()
    {
        var policy = new PlaybackDuckingPolicy();
        BargeInDecision decision = EvidenceDecision(end: 100);

        PlaybackDuckingPolicy.CommandBatch duck = policy.Update(
            Frame(startSample: 100, sampleCount: 20, streamEpoch: 1),
            in decision,
            candidateActive: false,
            confirmationPending: false,
            allowsDucking: true,
            ownerTurnId: 5,
            duckGain: 0.2f,
            releaseMs: 200);

        Assert.Equal(PlaybackDuckingPolicy.Kind.Duck, duck.Second.Kind);
        Assert.Equal(5, duck.Second.OwnerTurnId);
        Assert.Equal(0.2f, duck.Second.Gain);
        Assert.True(policy.IsDucked);

        PlaybackDuckingPolicy.CommandBatch release = policy.Update(
            Frame(startSample: 10_000, sampleCount: 20, streamEpoch: 1),
            in decision,
            candidateActive: false,
            confirmationPending: false,
            allowsDucking: true,
            ownerTurnId: 5,
            duckGain: 0.2f,
            releaseMs: 200);

        Assert.Equal(PlaybackDuckingPolicy.Kind.Restore, release.Second.Kind);
        Assert.False(policy.IsDucked);
    }

    [Fact]
    public void Ducking_OwnerChangeRestoresTheOldOwnerAndCanDuckTheNewOneInTheSameFrame()
    {
        var policy = new PlaybackDuckingPolicy();
        BargeInDecision decision = EvidenceDecision(end: 100);
        _ = policy.Update(
            Frame(startSample: 100, sampleCount: 20, streamEpoch: 1),
            in decision, false, false, true, ownerTurnId: 5, 0.2f, 200);

        PlaybackDuckingPolicy.CommandBatch changed = policy.Update(
            Frame(startSample: 120, sampleCount: 20, streamEpoch: 1),
            in decision, false, false, true, ownerTurnId: 6, 0.2f, 200);

        Assert.Equal(PlaybackDuckingPolicy.Kind.Restore, changed.First.Kind);
        Assert.Equal(5, changed.First.OwnerTurnId);
        Assert.Equal(PlaybackDuckingPolicy.Kind.Duck, changed.Second.Kind);
        Assert.Equal(6, changed.Second.OwnerTurnId);
        Assert.True(policy.IsDucked);
    }

    [Fact]
    public void Ducking_IsDisabledForIdentityConfirmedCandidatesAndUnusableTimelines()
    {
        var policy = new PlaybackDuckingPolicy();
        BargeInDecision decision = EvidenceDecision(end: 100);

        PlaybackDuckingPolicy.CommandBatch disabled = policy.Update(
            Frame(100, 20, 1), in decision, false, false, allowsDucking: false, 5, 0.2f, 200);
        Assert.True(disabled.IsEmpty);

        BargeInDecision discontinuous = decision with { HasDiscontinuity = true };
        PlaybackDuckingPolicy.CommandBatch unusable = policy.Update(
            Frame(100, 20, 1), in discontinuous, false, false, true, 5, 0.2f, 200);
        Assert.True(unusable.IsEmpty);
    }

    [Fact]
    public void Ducking_RollbackOnlyClearsTheMatchingDuck()
    {
        var policy = new PlaybackDuckingPolicy();
        BargeInDecision decision = EvidenceDecision(end: 100);
        PlaybackDuckingPolicy.CommandBatch duck = policy.Update(
            Frame(100, 20, 1), in decision, false, false, true, 5, 0.2f, 200);
        Assert.True(policy.IsDucked);

        policy.RollbackDuck(duck.Second.Sequence - 1);
        Assert.True(policy.IsDucked);

        policy.RollbackDuck(duck.Second.Sequence);
        Assert.False(policy.IsDucked);
    }

    // -----------------------------------------------------------------------------------
    // NearEndEffectExecutor and tail duration
    // -----------------------------------------------------------------------------------

    [Fact]
    public void EffectExecutor_AppliesRestoresForDifferentOwnersIndependently()
    {
        var playback = new RecordingPlaybackPort();
        var executor = new NearEndEffectExecutor(
            playback,
            new NullSignalSink(),
            new NullRecognitionSink(),
            speakerEvidence: null);

        // A newer restore for owner B may arrive before the older restore for the owner that is
        // actually ducked. Cross-owner restores are independent and must both be applied.
        executor.Execute(new NearEndEffect.RestorePlayback(OwnerTurnId: 6, StreamEpoch: 1, Sequence: 3));
        executor.Execute(new NearEndEffect.RestorePlayback(OwnerTurnId: 5, StreamEpoch: 1, Sequence: 1));

        Assert.Equal(
            [(6L, "restore"), (5L, "restore")],
            playback.Calls);
    }

    [Fact]
    public void EffectExecutor_DropsOnlyStaleCommandsForTheSameOwner()
    {
        var playback = new RecordingPlaybackPort();
        var executor = new NearEndEffectExecutor(
            playback,
            new NullSignalSink(),
            new NullRecognitionSink(),
            speakerEvidence: null);

        executor.Execute(new NearEndEffect.DuckPlayback(5, 1, 0.2f, 1f, Sequence: 10));
        executor.Execute(new NearEndEffect.RestorePlayback(5, 1, Sequence: 11));
        // A delayed duplicate of the older duck must not re-duck the owner after its restore.
        executor.Execute(new NearEndEffect.DuckPlayback(5, 1, 0.2f, 1f, Sequence: 10));

        Assert.Equal([(5L, "duck"), (5L, "restore")], playback.Calls);
    }

    [Fact]
    public void TailDuration_UsesSamplesPerChannelForStereo()
    {
        DuplexAudioFrame mono = Frame(startSample: 0, sampleCount: 320, streamEpoch: 1);
        Assert.Equal(20.0, NearEndDataPlane.ComputeTailDurationMs(in mono, tailStartSample: 0), 6);

        DuplexAudioFrame stereo = Frame(
            startSample: 0,
            sampleCount: 320,
            streamEpoch: 1,
            channels: 2);
        Assert.Equal(20.0, NearEndDataPlane.ComputeTailDurationMs(in stereo, tailStartSample: 0), 6);
        Assert.Equal(10.0, NearEndDataPlane.ComputeTailDurationMs(in stereo, tailStartSample: 320), 6);
    }

    // -----------------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------------

    private static NearEndCaptureStateMachine NewDrainingCapture(long targetSample, long fedThrough)
    {
        var capture = new NearEndCaptureStateMachine();
        capture.Start(
            turnId: 1,
            firstFeedSample: fedThrough,
            speechDurationMs: 40,
            preRollSource: "test",
            preRollSamples: 0);
        capture.BeginDrain(
            streamEpoch: 3,
            endSampleExclusive: targetSample,
            hasDiscontinuity: false,
            timeoutSamples: 160);
        return capture;
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

    private static BargeInDecision EvidenceDecision(long end)
        => new(
            Action: BargeInAction.Continue,
            FrameCount: 8,
            AcceptedFrameCount: 8,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 160,
            AcceptedDurationMs: 160,
            RequiredDurationMs: 100,
            ScoreDurationMs: 160,
            PeakScoreDurationMs: 160,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.05,
            MinRequiredRms: 0.05,
            MaxRequiredRms: 0.05,
            Reason: "test",
            StreamEpoch: 1,
            StartSample: 0,
            EndSampleExclusive: end,
            LatestFrameAccepted: true);

    private static DuplexAudioFrame Frame(
        long startSample,
        int sampleCount,
        long streamEpoch,
        float playbackGain = 1f,
        int channels = 1)
    {
        float[] audio = new float[sampleCount * channels];
        return new DuplexAudioFrame(
            SequenceNumber: 1,
            TimestampTicks: 1,
            SampleRate: 16000,
            Channels: (uint)channels,
            FrameCount: (uint)sampleCount,
            CaptureProcessed: true,
            RenderReference: audio,
            CaptureRaw: audio,
            CaptureBeforeAec: audio,
            CaptureAfterAec: audio,
            StreamEpoch: streamEpoch,
            StartSample: startSample,
            HasDiscontinuity: false,
            PlaybackGain: playbackGain);
    }

    private sealed class RecordingPlaybackPort : IActiveNearEndPlaybackPort
    {
        public List<(long OwnerTurnId, string Action)> Calls { get; } = [];

        public bool IsPlaybackActive => true;

        public long? ActiveOwnerTurnId => null;

        public bool TryDuckPlayback(long ownerTurnId, float gain, float originalGain)
        {
            Calls.Add((ownerTurnId, "duck"));
            return true;
        }

        public bool TryRestorePlayback(long ownerTurnId)
        {
            Calls.Add((ownerTurnId, "restore"));
            return true;
        }
    }

    private sealed class NullSignalSink : IActiveNearEndSignalSink
    {
        public long NextOperationId() => 1;

        public bool TryEnqueueConfirmation(long operationId) => true;

        public void EnqueueSpeechEnded(long turnId, TimeSpan duration)
        {
        }

        public void ReportDataPlaneFault(AssistantFailure failure)
        {
        }
    }

    private sealed class NullRecognitionSink : INearEndRecognitionAudioSink
    {
        public bool TryFeed(long turnId, ReadOnlySpan<float> audio) => true;

        public bool HasAttachedSession(long turnId) => true;

        public void Reset(long turnId)
        {
        }
    }

    private static NearEndConfirmationLease Lease(long operationId)
        => new(
            OperationId: operationId,
            CandidateId: new SpeakerCandidateId(9),
            OwnerTurnId: 4,
            StreamEpoch: 1,
            Selection: ActiveNearEndPreRollSelection.Empty("test"),
            InitialAudio: [],
            Decision: default,
            EndpointVadCandidate: false,
            IdentityRequired: false,
            IdentityAuthorized: false,
            SpeakerResult: default);

    private static void AddTimeline(
        NearEndTimelineBuffer timeline,
        float[] audio,
        long streamEpoch,
        ref long start,
        bool discontinuity = false)
    {
        var range = new CaptureRange(
            StreamEpoch: streamEpoch,
            StartSequenceNumber: 1,
            EndSequenceNumber: 1,
            StartSample: start,
            EndSampleExclusive: start + audio.Length,
            TimestampTicks: 1);
        timeline.Add(audio, in range, discontinuity);
        start += audio.Length;
    }
}
