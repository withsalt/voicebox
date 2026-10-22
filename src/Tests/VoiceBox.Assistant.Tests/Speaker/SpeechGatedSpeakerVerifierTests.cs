using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Speech-gated prefetch + first-window decision contracts. Window = 8000 samples (500ms), hop =
/// 1600 (100ms); every frame is 1600 samples so one hop-aligned window completes per frame after
/// the first five. Scores are scripted through the audio amplitude (see ScriptedSpeakerService).
/// </summary>
public sealed class SpeechGatedSpeakerVerifierTests : IDisposable
{
    private const int Hop = 1600;
    private const int Window = 8000;
    private readonly ScriptedSpeakerService _speaker = new();
    private readonly List<SpeechGatedSpeakerVerifier> _verifiers = [];

    [Fact]
    public async Task Silence_SchedulesNoInference()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 10, ownerScore: 0f);
        await Task.Delay(50);

        Assert.Equal(0, _speaker.ExtractCount);
        Assert.Equal(0, verifier.Prefetch.ScheduledCount);
        Assert.False(verifier.Prefetch.IsSpeechActive);
    }

    [Fact]
    public async Task SpeechGate_PrefetchesFirstCompleteWindowBeforeAnyQuery()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);

        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        Assert.True(verifier.Prefetch.IsSpeechActive);
        Assert.True(_speaker.ExtractCount >= 1);
        Assert.True(verifier.Cache.TryGet(1, 0, Window, out SpeakerWindowEvidence evidence));
        Assert.Equal(SpeakerWindowEvidenceKind.Scored, evidence.Kind);
        Assert.True(evidence.TryGetScore(SpeakerPurposeBank.CleanNearEnd, out _));
        Assert.True(evidence.TryGetScore(SpeakerPurposeBank.DoubleTalkNearEnd, out _));
    }

    [Fact]
    public async Task FormalQuery_SchedulesOnlyTheNextMissingWindow()
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.PrefetchMaxWindows = 1;
        SpeechGatedSpeakerVerifier verifier = Create(options);
        verifier.Prefetch.SetEnabled(false);
        _speaker.ExtractEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _speaker.ExtractBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        FeedFrames(verifier, count: 7, ownerScore: 0.6f);
        try
        {
            SpeakerVerificationResult pending = verifier.Evaluate(
                BargeInQuery(candidateId: 101, end: Window + 2 * Hop));
            await _speaker.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.Equal(SpeakerVerificationStatus.Pending, pending.Status);
            Assert.Equal(0, verifier.Scheduler.QueueDepth);
            Assert.Equal(1, verifier.Diagnostics.Snapshot().InferenceScheduled);
        }
        finally { _speaker.ExtractBlock.TrySetResult(); }
    }

    [Fact]
    public async Task Diagnostics_DistinguishCacheAndInferenceActivity()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        SpeakerVerificationResult result = verifier.Evaluate(
            WakeupQuery(candidateId: 102, focusStart: 2000, focusEnd: 6000, captureEnd: Window));
        Assert.Equal(SpeakerVerificationStatus.Match, result.Status);

        SpeakerVerificationDiagnosticsSnapshot diagnostics = verifier.Diagnostics.Snapshot();
        Assert.True(diagnostics.Queries >= 1);
        Assert.True(diagnostics.CacheHits >= 1);
        Assert.True(diagnostics.InferenceCompleted >= 1);
        Assert.True(diagnostics.AppendedFrames >= 5);
    }

    [Fact]
    public async Task FirstWindowHighConfidence_MatchesImmediatelyFromCache()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        int extractsBeforeQuery = _speaker.ExtractCount;

        SpeakerVerificationResult result = verifier.Evaluate(WakeupQuery(candidateId: 1, focusStart: 2000, focusEnd: 6000, captureEnd: Window));

        Assert.Equal(SpeakerVerificationStatus.Match, result.Status);
        Assert.Equal(SpeakerVerificationReason.HighConfidenceMatch, result.Reason);
        Assert.True(result.HighConfidenceMatch);
        Assert.Equal(1, result.ObservedWindows);
        Assert.Equal(extractsBeforeQuery, _speaker.ExtractCount);
        Assert.Equal(new SpeakerCandidateId(1), result.CandidateId);
    }

    [Fact]
    public async Task NormalScore_NeedsMajorityAndDecidesPerHopWithoutBatching()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.6f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        SpeakerVerificationResult afterOne = verifier.Evaluate(BargeInQuery(candidateId: 2, end: Window));
        Assert.Equal(SpeakerVerificationStatus.Pending, afterOne.Status);
        Assert.Equal(1, afterOne.MatchedWindows);

        FeedFrames(verifier, count: 1, ownerScore: 0.6f, startFrame: 5);
        await WaitUntilAsync(() => verifier.Cache.Count >= 2);
        SpeakerVerificationResult afterTwo = verifier.Evaluate(BargeInQuery(candidateId: 2, end: Window + Hop));
        Assert.Equal(SpeakerVerificationStatus.Match, afterTwo.Status);
        Assert.Equal(SpeakerVerificationReason.MajorityMatch, afterTwo.Reason);
        Assert.Equal(2, afterTwo.MatchedWindows);
    }

    [Fact]
    public async Task StrongUpstream_LowersVoteToOneButStillNeedsCurrentMatch()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.6f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        SpeakerVerificationResult strong = verifier.Evaluate(
            WakeupQuery(candidateId: 3, focusStart: 1000, focusEnd: 6000, captureEnd: Window, strongUpstream: true));
        Assert.Equal(SpeakerVerificationStatus.Match, strong.Status);
        Assert.Equal(SpeakerVerificationReason.StrongUpstreamMatch, strong.Reason);
        Assert.True(strong.UsedStrongUpstreamEvidence);

        SpeechGatedSpeakerVerifier impostor = Create();
        FeedFrames(impostor, count: 5, ownerScore: 0.2f);
        await WaitUntilAsync(() => impostor.Cache.Count >= 1);
        SpeakerVerificationResult denied = impostor.Evaluate(
            WakeupQuery(candidateId: 4, focusStart: 1000, focusEnd: 6000, captureEnd: Window, strongUpstream: true));
        Assert.NotEqual(SpeakerVerificationStatus.Match, denied.Status);
    }

    [Fact]
    public async Task StrongUpstreamOneVote_CanBeDisabledByPolicy()
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.Confirmation.AllowStrongUpstreamOneVote = false;
        SpeechGatedSpeakerVerifier verifier = Create(options);
        FeedFrames(verifier, count: 5, ownerScore: 0.6f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        SpeakerVerificationResult result = verifier.Evaluate(
            WakeupQuery(candidateId: 5, focusStart: 1000, focusEnd: 6000, captureEnd: Window, strongUpstream: true));
        Assert.Equal(SpeakerVerificationStatus.Pending, result.Status);
    }

    [Fact]
    public async Task Impostor_IsMismatchAfterEnoughRejectedWindows()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 6, ownerScore: 0.1f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 2);

        SpeakerVerificationResult result = verifier.Evaluate(BargeInQuery(candidateId: 6, end: Window + Hop));
        Assert.Equal(SpeakerVerificationStatus.Mismatch, result.Status);
        Assert.Equal(SpeakerVerificationReason.MajorityMismatch, result.Reason);
        Assert.Equal(2, result.RejectedWindows);
    }

    [Theory]
    [InlineData(6)]
    [InlineData(7)]
    public async Task SingleTalk_RejectedWindows_RemainPendingUntilEndpoint(int frameCount)
    {
        SpeechGatedSpeakerVerifier verifier = Create(singleTalk: true);
        FeedFrames(verifier, count: frameCount, ownerScore: 0.1f);
        await WaitUntilAsync(() => verifier.Cache.Count >= frameCount - 4);

        SpeakerEvidenceQuery query = SingleTalkQuery(candidateId: 22, start: 0, end: frameCount * Hop);
        SpeakerVerificationResult pending = verifier.Evaluate(in query);

        Assert.Equal(SpeakerVerificationStatus.Pending, pending.Status);
        Assert.Equal(frameCount - 4, pending.RejectedWindows);
        Assert.Equal(0, pending.MatchedWindows);

        SpeakerVerificationResult complete = verifier.Evaluate(query with { InputComplete = true });
        Assert.Equal(SpeakerVerificationStatus.Mismatch, complete.Status);
        Assert.Equal(SpeakerVerificationReason.MajorityMismatch, complete.Reason);
    }

    [Fact]
    public async Task SingleTalk_LowScorePrefix_CanMatchLaterWindowsBeforeEndpoint()
    {
        SpeechGatedSpeakerVerifier verifier = Create(singleTalk: true);
        FeedFrames(verifier, count: 6, ownerScore: 0.1f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 2);

        SpeakerEvidenceQuery query = SingleTalkQuery(candidateId: 23, start: 0, end: 6 * Hop);
        Assert.Equal(SpeakerVerificationStatus.Pending, verifier.Evaluate(in query).Status);

        FeedFrames(verifier, count: 7, ownerScore: 0.6f, startFrame: 6);
        SpeakerVerificationResult recovered = await verifier.WaitForDecisionAsync(
            query with { CandidateEndSampleExclusive = 13 * Hop },
            TimeSpan.FromSeconds(2));

        Assert.Equal(SpeakerVerificationStatus.Match, recovered.Status);
        Assert.Equal(SpeakerVerificationReason.MajorityMatch, recovered.Reason);
        Assert.True(recovered.MatchedWindows >= 2);
        Assert.Equal(0, recovered.RejectedWindows);
        Assert.Equal(query.CandidateId, recovered.CandidateId);
    }

    [Fact]
    public async Task SingleTalk_CompletedInput_WaitsForIdentityAudioBeforeRejecting()
    {
        SpeechGatedSpeakerVerifier verifier = Create(singleTalk: true);
        FeedFrames(verifier, count: 6, ownerScore: 0.1f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 2);

        // The endpoint has arrived before the identity tap's remaining owner speech.
        SpeakerEvidenceQuery query = SingleTalkQuery(
            candidateId: 24, start: 0, end: 13 * Hop, inputComplete: true);
        SpeakerVerificationResult pending = verifier.Evaluate(in query);
        Assert.Equal(SpeakerVerificationStatus.Pending, pending.Status);
        Assert.Equal(2, pending.RejectedWindows);

        FeedFrames(verifier, count: 7, ownerScore: 0.6f, startFrame: 6);
        SpeakerVerificationResult complete = await verifier.WaitForDecisionAsync(query, TimeSpan.FromSeconds(2));
        Assert.Equal(SpeakerVerificationStatus.Match, complete.Status);
        Assert.Equal(SpeakerVerificationReason.MajorityMatch, complete.Reason);
    }

    [Fact]
    public async Task SingleTalk_HighConfidenceOwner_MatchesBeforeEndpoint()
    {
        SpeechGatedSpeakerVerifier verifier = Create(singleTalk: true);
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        SpeakerVerificationResult result = verifier.Evaluate(
            SingleTalkQuery(candidateId: 25, start: 0, end: Window));

        Assert.Equal(SpeakerVerificationStatus.Match, result.Status);
        Assert.Equal(SpeakerVerificationReason.HighConfidenceMatch, result.Reason);
        Assert.Equal(1, result.ObservedWindows);
    }

    [Fact]
    public async Task AmbiguousWindows_EndAsInsufficientEvidenceNotMismatch()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 7, ownerScore: 0.47f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 3);

        SpeakerVerificationResult result = verifier.Evaluate(
            BargeInQuery(candidateId: 7, end: Window + 2 * Hop, inputComplete: true));
        Assert.Equal(SpeakerVerificationStatus.InsufficientEvidence, result.Status);
        Assert.Equal(3, result.AmbiguousWindows);
        Assert.Equal(0, result.RejectedWindows);
    }

    [Fact]
    public async Task SilentCandidate_IsInsufficientSpeechWhenComplete()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 6, ownerScore: 0f);
        await Task.Delay(30);

        SpeakerVerificationResult pending = verifier.Evaluate(BargeInQuery(candidateId: 8, end: Window));
        Assert.Equal(SpeakerVerificationStatus.Pending, pending.Status);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        SpeakerVerificationResult complete = verifier.Evaluate(BargeInQuery(candidateId: 8, end: Window, inputComplete: true));
        Assert.Equal(SpeakerVerificationStatus.InsufficientEvidence, complete.Status);
        Assert.Equal(SpeakerVerificationReason.InsufficientSpeech, complete.Reason);
    }

    [Fact]
    public async Task TrailingSilenceInsideCandidate_DoesNotDisplaceTheUtteranceWindows()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        // The user stops talking but the candidate (and the callback) keeps running for 600ms.
        FeedFrames(verifier, count: 6, ownerScore: 0f, startFrame: 5);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);

        // The newest complete windows are pure silence; they are not evidence and must not push the
        // scored speech windows out of the recent selection.
        SpeakerVerificationResult result = await verifier.WaitForDecisionAsync(
            BargeInQuery(candidateId: 20, end: 11 * Hop, inputComplete: true),
            TimeSpan.FromSeconds(2));

        Assert.Equal(SpeakerVerificationStatus.Match, result.Status);
        Assert.True(result.MatchedWindows >= 2);
        Assert.Equal(0, result.RejectedWindows);
    }

    [Fact]
    public async Task EmbeddingIsExtractedOncePerWindow_AndBothBanksScored()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 7, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 3);
        int extracts = _speaker.ExtractCount;

        verifier.Evaluate(BargeInQuery(candidateId: 9, end: Window + 2 * Hop));
        verifier.Evaluate(WakeupQuery(candidateId: 10, focusStart: 1000, focusEnd: 6000, captureEnd: Window + 2 * Hop));
        await Task.Delay(30);

        Assert.Equal(3, extracts);
        Assert.Equal(extracts, _speaker.ExtractCount);
    }

    [Fact]
    public async Task TimelineGap_FailsQueriesAcrossTheGapButNotAfterIt()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        // Frames 6.. arrive after a dropped frame.
        FeedFrames(verifier, count: 5, ownerScore: 0.9f, startFrame: 6);
        await WaitUntilAsync(() => verifier.Cache.Count >= 2);

        SpeakerVerificationResult crossing = verifier.Evaluate(BargeInQuery(candidateId: 11, start: 0, end: 11 * Hop));
        Assert.Equal(SpeakerVerificationStatus.Unavailable, crossing.Status);
        Assert.Equal(SpeakerVerificationReason.TimelineGap, crossing.Reason);

        SpeakerVerificationResult after = verifier.Evaluate(BargeInQuery(candidateId: 12, start: 6 * Hop, end: 11 * Hop));
        Assert.Equal(SpeakerVerificationStatus.Match, after.Status);
    }

    [Fact]
    public async Task ProviderException_MarksOnlyThatWindowUnavailable()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        _speaker.ExtractException = new InvalidOperationException("model crashed");
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        _speaker.ExtractException = null;
        FeedFrames(verifier, count: 2, ownerScore: 0.9f, startFrame: 5);
        await WaitUntilAsync(() => verifier.Cache.Count >= 3);

        Assert.True(verifier.Cache.TryGet(1, 0, Window, out SpeakerWindowEvidence failed));
        Assert.Equal(SpeakerWindowEvidenceKind.Failed, failed.Kind);
        Assert.Equal(SpeakerVerificationReason.ProviderError, failed.Reason);
        SpeakerVerificationResult result = verifier.Evaluate(BargeInQuery(candidateId: 13, end: Window + 2 * Hop));
        Assert.Equal(SpeakerVerificationStatus.Match, result.Status);
    }

    [Fact]
    public async Task WaitForDecision_TimesOutAsUnavailableWhileInferenceIsBlocked()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        _speaker.ExtractEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _speaker.ExtractBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await _speaker.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

        SpeakerVerificationResult result = await verifier.WaitForDecisionAsync(
            BargeInQuery(candidateId: 14, end: Window),
            TimeSpan.FromMilliseconds(50));
        Assert.Equal(SpeakerVerificationStatus.Unavailable, result.Status);
        Assert.Equal(SpeakerVerificationReason.InferenceTimeout, result.Reason);
        _speaker.ExtractBlock.TrySetResult();
    }

    [Fact]
    public async Task Cancel_DropsQueuedCandidateJobsButKeepsCachedEmbeddings()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        _speaker.ExtractEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _speaker.ExtractBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);
        await _speaker.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        verifier.Prefetch.SetEnabled(false);
        FeedFrames(verifier, count: 3, ownerScore: 0.9f, startFrame: 5);

        SpeakerVerificationResult pending = verifier.Evaluate(BargeInQuery(candidateId: 15, end: Window + 3 * Hop));
        Assert.Equal(SpeakerVerificationStatus.Pending, pending.Status);
        int queued = verifier.Scheduler.QueueDepth;
        Assert.True(queued >= 1);

        verifier.Cancel(new SpeakerCandidateId(15));
        Assert.True(verifier.Scheduler.QueueDepth < queued);
        _speaker.ExtractBlock.TrySetResult();
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
        Assert.True(verifier.Cache.TryGet(1, 0, Window, out _));
    }

    [Fact]
    public async Task PrefetchOverload_RejectsPrefetchButNeverStarvesCandidates()
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.InferenceQueueCapacity = 1;
        options.PrefetchMaxWindows = 4;
        SpeechGatedSpeakerVerifier verifier = Create(options);
        _speaker.ExtractEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _speaker.ExtractBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        FeedFrames(verifier, count: 9, ownerScore: 0.9f);
        await _speaker.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.True(verifier.Prefetch.RejectedCount >= 1);
        Assert.Equal(0, verifier.Scheduler.OverloadCount);
        _speaker.ExtractBlock.TrySetResult();
        await WaitUntilAsync(() => verifier.Cache.Count >= 1);
    }

    [Fact]
    public async Task SingleTalkInferenceOverload_DuringOpenCandidate_RemainsPending()
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.InferenceQueueCapacity = 1;
        SpeechGatedSpeakerVerifier verifier = Create(options, singleTalk: true);
        _speaker.ExtractEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _speaker.ExtractBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        // Occupy the worker with one prefetch and the queue with another window. The formal
        // single-talk query is intentionally evaluated before the utterance endpoint.
        FeedFrames(verifier, count: 6, ownerScore: 0.9f);
        await _speaker.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));

        verifier.Prefetch.SetEnabled(false);
        FeedFrames(verifier, count: 5, ownerScore: 0.9f, startFrame: 6);
        SpeakerEvidenceQuery query = SingleTalkQuery(
            candidateId: 21,
            start: 6 * Hop,
            end: 11 * Hop,
            inputComplete: false);

        SpeakerVerificationResult result = verifier.Evaluate(in query);
        Assert.Equal(SpeakerVerificationStatus.Pending, result.Status);
        Assert.NotEqual(SpeakerVerificationReason.InferenceTimeout, result.Reason);
        Assert.NotEqual(SpeakerVerificationReason.InferenceOverload, result.Reason);
        Assert.Equal(0, verifier.Scheduler.OverloadCount);

        _speaker.ExtractBlock.TrySetResult();
    }

    [Fact]
    public void Query_WithoutEnabledBank_IsUnavailableMissingPurposeBank()
    {
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.Confirmation.Enable = true;
        SpeechGatedSpeakerVerifier verifier = Create(options, includeDoubleTalk: false, singleTalk: true);
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);

        SpeakerVerificationResult result = verifier.Evaluate(BargeInQuery(candidateId: 16, end: Window));
        Assert.Equal(SpeakerVerificationStatus.Unavailable, result.Status);
        Assert.Equal(SpeakerVerificationReason.MissingPurposeBank, result.Reason);
    }

    [Fact]
    public void Query_OnStaleEpochOrExpiredAudio_IsUnavailable()
    {
        SpeechGatedSpeakerVerifier verifier = Create();
        FeedFrames(verifier, count: 5, ownerScore: 0.9f);

        SpeakerVerificationResult epoch = verifier.Evaluate(BargeInQuery(candidateId: 17, end: Window, streamEpoch: 2));
        Assert.Equal(SpeakerVerificationReason.StreamEpochMismatch, epoch.Reason);
        Assert.Equal(SpeakerVerificationStatus.Unavailable, epoch.Status);
    }

    private SpeechGatedSpeakerVerifier Create(
        VoiceAssistantSpeakerOptions? options = null,
        bool includeDoubleTalk = true,
        bool singleTalk = false)
    {
        options ??= SpeakerTestProfiles.Options("unused");
        SpeakerVerificationPolicy policy = SpeakerTestProfiles.Policy(_speaker, options, singleTalk: singleTalk);
        var verifier = new SpeechGatedSpeakerVerifier(
            _speaker,
            policy,
            SpeakerTestProfiles.Variant(SpeakerTestProfiles.Descriptor(), includeDoubleTalk),
            NullLogger.Instance);
        verifier.Start();
        _verifiers.Add(verifier);
        return verifier;
    }

    private static void FeedFrames(SpeechGatedSpeakerVerifier verifier, int count, float ownerScore, int startFrame = 0)
    {
        for (int i = 0; i < count; i++)
        {
            int frame = startFrame + i;
            verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(
                1,
                (long)frame * Hop,
                ScriptedSpeakerService.OwnerScoreAudio(ownerScore, Hop),
                sequence: frame));
        }
    }

    private static SpeakerEvidenceQuery BargeInQuery(
        long candidateId,
        long end,
        long start = 0,
        bool inputComplete = false,
        long streamEpoch = 1)
        => new(
            SpeakerVerificationPurpose.BargeInCommand,
            streamEpoch,
            start,
            end,
            QueryTimestampTicks: 1,
            InputComplete: inputComplete,
            CandidateId: new SpeakerCandidateId(candidateId));

    private static SpeakerEvidenceQuery SingleTalkQuery(
        long candidateId,
        long start,
        long end,
        bool inputComplete = false)
        => new(
            SpeakerVerificationPurpose.SingleTalk,
            1,
            start,
            end,
            QueryTimestampTicks: 1,
            InputComplete: inputComplete,
            CandidateId: new SpeakerCandidateId(candidateId));

    private static SpeakerEvidenceQuery WakeupQuery(
        long candidateId,
        long focusStart,
        long focusEnd,
        long captureEnd,
        bool strongUpstream = false)
        => new(
            SpeakerVerificationPurpose.Wakeup,
            1,
            0,
            captureEnd,
            QueryTimestampTicks: 1,
            FocusStartSample: focusStart,
            FocusEndSampleExclusive: focusEnd,
            StrongUpstreamEvidence: strongUpstream,
            CandidateId: new SpeakerCandidateId(candidateId));

    private static async Task WaitUntilAsync(Func<bool> condition, int timeoutMs = 3000)
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (!condition())
        {
            if (DateTime.UtcNow > deadline)
                throw new TimeoutException("Condition was not met in time.");
            await Task.Delay(5);
        }
    }

    public void Dispose()
    {
        foreach (SpeechGatedSpeakerVerifier verifier in _verifiers)
            verifier.Dispose();
    }
}
