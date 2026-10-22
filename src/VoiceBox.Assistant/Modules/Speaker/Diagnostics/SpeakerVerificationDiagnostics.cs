using VoiceBox.Assistant.Modules.Speaker.Models;
using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace VoiceBox.Assistant.Modules.Speaker.Diagnostics;

/// <summary>
/// Lock-free counters for the identity data plane. The snapshot is intentionally owned by the
/// Assistant and contains no provider objects or audio buffers. It is used by diagnostics and
/// tests to distinguish acoustic rejection from timeline, capacity, and scheduling failures.
/// </summary>
internal sealed class SpeakerVerificationDiagnostics
{
    private readonly ILogger _logger;
    private readonly object _candidateLock = new();
    private readonly Dictionary<SpeakerCandidateId, CandidateObservation> _candidates = [];
    private readonly long[] _reasons = new long[Enum.GetValues<SpeakerVerificationReason>().Length];
    private long _lastSummary;
    private long _queueWaitTicks;
    private long _queueWaitCount;
    private long _missingSamples;
    private long _qualityRejected;
    private long _finalCandidates;
    private long _queries;
    private long _matches;
    private long _pending;
    private long _mismatches;
    private long _insufficientEvidence;
    private long _unavailable;
    private long _cancelled;
    private long _cacheHits;
    private long _cacheMisses;
    private long _inferenceScheduled;
    private long _prefetchScheduled;
    private long _prefetchRejected;
    private long _inferenceOverload;
    private long _inferenceCompleted;
    private long _inferenceLatencyTicks;
    private long _inferenceLatencyMaxTicks;
    private long _timelineGaps;
    private long _audioExpired;
    private long _invalidFrames;
    private long _appendedFrames;
    private long _maxQueueDepth;

    public SpeakerVerificationDiagnostics(ILogger logger) => _logger = logger;

    public long ReasonCount(SpeakerVerificationReason reason) => Volatile.Read(ref _reasons[(int)reason]);
    public void ObserveDecision(in SpeakerEvidenceQuery query, in SpeakerVerificationResult result)
    {
        long started = Stopwatch.GetTimestamp();
        lock (_candidateLock)
        {
            if (_candidates.TryGetValue(query.CandidateId, out CandidateObservation previous))
            {
                started = previous.Started;
                // A caller may repeatedly read a final result. Count a candidate once.
                if (previous.Final) return;
            }
            if (_candidates.Count >= 256 && !_candidates.ContainsKey(query.CandidateId))
                _candidates.Remove(_candidates.MinBy(pair => pair.Value.Started).Key);
            _candidates[query.CandidateId] = new(started, result.Status, result.Reason, result.IsFinal);
        }
        if (result.IsFinal) Interlocked.Increment(ref _finalCandidates);
        // Pending is a high-frequency state. Keep it silent; only terminal outcomes are useful in
        // the normal log stream. Trace can still expose the full candidate transition history.
        if (result.IsFinal)
            _logger.LogDebug("Speaker candidate: purpose={Purpose}, candidate={CandidateId}, owner={OwnerTurnId}, epoch={Epoch}, range=[{Start},{End}), status={Status}, reason={Reason}, observed={Observed}, matched={Matched}/{Required}, invalid={Invalid}, score={Score}, elapsedMs={ElapsedMs:F1}.",
                query.Purpose, query.CandidateId.Value, query.OwnerTurnId, query.StreamEpoch,
                query.CandidateStartSample, query.CandidateEndSampleExclusive, result.Status, result.Reason,
                result.ObservedWindows, result.MatchedWindows, result.RequiredMatches, result.InvalidWindows,
                result.Score, Stopwatch.GetElapsedTime(started).TotalMilliseconds);

        long now = Stopwatch.GetTimestamp();
        long previousSummary = Volatile.Read(ref _lastSummary);
        if (Stopwatch.GetElapsedTime(previousSummary, now) >= TimeSpan.FromSeconds(1) &&
            Interlocked.CompareExchange(ref _lastSummary, now, previousSummary) == previousSummary)
            LogSummary();
    }

    public void Forget(SpeakerCandidateId id)
    {
        lock (_candidateLock) _candidates.Remove(id);
    }

    public void LogSummary()
    {
        if (_logger.IsEnabled(LogLevel.Trace))
            _logger.LogTrace("Speaker diagnostics: {Snapshot}", Snapshot());
    }

    public void RecordQueueWait(long ticks)
    {
        Interlocked.Add(ref _queueWaitTicks, ticks);
        Interlocked.Increment(ref _queueWaitCount);
    }

    public void RecordQualityRejected(in SpeakerWindowQuality quality)
    {
        Interlocked.Increment(ref _qualityRejected);

        //_logger.LogDebug(
        //    "Speaker window ineligible: finite={Finite}, rms={Rms:F4}, peak={Peak:F4}, speechRatio={SpeechRatio:P1}, clippingRatio={ClippingRatio:P3}, dcOffset={DcOffset:F4}, speechSamples={SpeechSamples}/{SampleCount}.",
        //    quality.AllFinite,
        //    quality.Rms,
        //    quality.Peak,
        //    quality.SpeechRatio,
        //    quality.ClippingRatio,
        //    quality.DcOffset,
        //    quality.SpeechSamples,
        //    quality.SampleCount);
    }

    public void RecordQuery() => Interlocked.Increment(ref _queries);

    public void RecordResult(SpeakerVerificationStatus status, SpeakerVerificationReason reason = SpeakerVerificationReason.None)
    {
        Interlocked.Increment(ref _reasons[(int)reason]);
        switch (status)
        {
            case SpeakerVerificationStatus.Match:
                Interlocked.Increment(ref _matches);
                break;
            case SpeakerVerificationStatus.Pending:
                Interlocked.Increment(ref _pending);
                break;
            case SpeakerVerificationStatus.Mismatch:
                Interlocked.Increment(ref _mismatches);
                break;
            case SpeakerVerificationStatus.InsufficientEvidence:
                Interlocked.Increment(ref _insufficientEvidence);
                break;
            case SpeakerVerificationStatus.Unavailable:
                Interlocked.Increment(ref _unavailable);
                break;
            case SpeakerVerificationStatus.Cancelled:
                Interlocked.Increment(ref _cancelled);
                break;
        }

        switch (reason)
        {
            case SpeakerVerificationReason.AudioExpired:
                RecordAudioExpired();
                break;
        }
    }

    public void RecordCacheHit() => Interlocked.Increment(ref _cacheHits);

    public void RecordCacheMiss() => Interlocked.Increment(ref _cacheMisses);

    public void RecordInferenceScheduled(bool prefetch)
    {
        Interlocked.Increment(ref _inferenceScheduled);
        if (prefetch)
            Interlocked.Increment(ref _prefetchScheduled);
    }

    public void RecordPrefetchRejected() => Interlocked.Increment(ref _prefetchRejected);

    public void RecordInferenceOverload() => Interlocked.Increment(ref _inferenceOverload);

    public void RecordInferenceCompleted(long elapsedTicks)
    {
        Interlocked.Increment(ref _inferenceCompleted);
        Interlocked.Add(ref _inferenceLatencyTicks, elapsedTicks);
        long observed;
        do
        {
            observed = Volatile.Read(ref _inferenceLatencyMaxTicks);
            if (elapsedTicks <= observed)
                break;
        }
        while (Interlocked.CompareExchange(ref _inferenceLatencyMaxTicks, elapsedTicks, observed) != observed);
    }

    public void RecordTimelineGap(long missingSamples = 0)
    {
        Interlocked.Increment(ref _timelineGaps);
        Interlocked.Add(ref _missingSamples, missingSamples);
    }

    public void RecordAudioExpired() => Interlocked.Increment(ref _audioExpired);

    public void RecordTimelineFailure(SpeakerTimelineFailure failure)
    {
        switch (failure)
        {
            case SpeakerTimelineFailure.AudioExpired:
                RecordAudioExpired();
                break;
        }
    }

    public void RecordInvalidFrame() => Interlocked.Increment(ref _invalidFrames);

    public void RecordAppendedFrame() => Interlocked.Increment(ref _appendedFrames);

    public void ObserveQueueDepth(int depth)
    {
        long value = Math.Max(0, depth);
        long observed;
        do
        {
            observed = Volatile.Read(ref _maxQueueDepth);
            if (value <= observed)
                return;
        }
        while (Interlocked.CompareExchange(ref _maxQueueDepth, value, observed) != observed);
    }

    public SpeakerVerificationDiagnosticsSnapshot Snapshot()
        => new(
            Queries: Volatile.Read(ref _queries),
            Matches: Volatile.Read(ref _matches),
            Pending: Volatile.Read(ref _pending),
            Mismatches: Volatile.Read(ref _mismatches),
            InsufficientEvidence: Volatile.Read(ref _insufficientEvidence),
            Unavailable: Volatile.Read(ref _unavailable),
            Cancelled: Volatile.Read(ref _cancelled),
            CacheHits: Volatile.Read(ref _cacheHits),
            CacheMisses: Volatile.Read(ref _cacheMisses),
            InferenceScheduled: Volatile.Read(ref _inferenceScheduled),
            PrefetchScheduled: Volatile.Read(ref _prefetchScheduled),
            PrefetchRejected: Volatile.Read(ref _prefetchRejected),
            InferenceOverload: Volatile.Read(ref _inferenceOverload),
            InferenceCompleted: Volatile.Read(ref _inferenceCompleted),
            InferenceLatencyTicks: Volatile.Read(ref _inferenceLatencyTicks),
            InferenceLatencyMaxTicks: Volatile.Read(ref _inferenceLatencyMaxTicks),
            TimelineGaps: Volatile.Read(ref _timelineGaps),
            AudioExpired: Volatile.Read(ref _audioExpired),
            InvalidFrames: Volatile.Read(ref _invalidFrames),
            AppendedFrames: Volatile.Read(ref _appendedFrames),
            MaxQueueDepth: Volatile.Read(ref _maxQueueDepth),
            QueueWaitMilliseconds: Volatile.Read(ref _queueWaitTicks) * 1000d / Stopwatch.Frequency / Math.Max(1, Volatile.Read(ref _queueWaitCount)),
            MissingSamples: Volatile.Read(ref _missingSamples),
            QualityRejected: Volatile.Read(ref _qualityRejected),
            FinalCandidates: Volatile.Read(ref _finalCandidates));

    private readonly record struct CandidateObservation(long Started, SpeakerVerificationStatus Status, SpeakerVerificationReason Reason, bool Final);
}
