using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Buffering;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Diagnostics;
using VoiceBox.Assistant.Modules.Speaker.Evidence;
using VoiceBox.Assistant.Modules.Speaker.Extensions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Controllers;

/// <summary>
/// Online owner verifier. Identity frames are copied into the timeline ring on the dispatch
/// thread; speech energy opens the prefetch gate so the first complete window is inferred before
/// any KWS hit or DTD interrupt exists. <see cref="Evaluate"/> decides from the cache the moment
/// the first window is scored: a high-confidence window, a strong-upstream one-vote, or the
/// normal 2-of-3 majority may match; missing windows are scheduled with candidate priority.
/// Time-based waiting is the caller's bounded deadline; this type never blocks on the model.
/// </summary>
internal sealed class SpeechGatedSpeakerVerifier : ISpeakerEvidenceProvider, IDisposable
{
    private readonly ISpeakerService _speaker;
    private readonly SpeakerVerificationPolicy _policy;
    private readonly OwnerSpeakerProfileVariant _variant;
    private readonly ILogger _logger;
    private readonly SpeakerAudioTimelineBuffer _timeline;
    private readonly SpeakerEvidenceCache _cache;
    private readonly SpeakerWindowQualityEvaluator _quality;
    private readonly SpeakerEvidenceWindowSelector _windows;
    private readonly SpeakerInferenceScheduler _scheduler;
    private readonly SpeakerSpeechPrefetchGate _prefetch;
    private readonly SpeakerVerificationDiagnostics _diagnostics;
    private readonly IReadOnlyDictionary<SpeakerPurposeBank, SpeakerPrototypeBank> _banks;
    private readonly object _priorLock = new();
    private long _lastOwnerMatchEpoch;
    private long _lastOwnerMatchEvidenceEnd;
    private long _nextCandidateId;
    private long _lastTrimStart = long.MinValue;
    private long _lastTrimEpoch;
    private int _disposed;

    public SpeechGatedSpeakerVerifier(
        ISpeakerService speaker,
        SpeakerVerificationPolicy policy,
        OwnerSpeakerProfileVariant variant,
        ILogger logger)
    {
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _variant = variant ?? throw new ArgumentNullException(nameof(variant));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _diagnostics = new SpeakerVerificationDiagnostics(logger);
        _banks = variant.Banks
            .Where(static pair => pair.Value.IsUsable)
            .ToDictionary(static pair => pair.Key, static pair => pair.Value);
        _timeline = new SpeakerAudioTimelineBuffer(
            checked((int)Math.Max(policy.RetentionSamples, policy.WindowSamples * 2L)),
            policy.SampleRate,
            _diagnostics);
        _cache = new SpeakerEvidenceCache();
        _quality = new SpeakerWindowQualityEvaluator(policy.MinFrameRms, policy.SampleRate);
        _windows = new SpeakerEvidenceWindowSelector(policy);
        _scheduler = new SpeakerInferenceScheduler(
            speaker,
            _timeline,
            _cache,
            _quality,
            _banks,
            policy.InferenceQueueCapacity,
            logger,
            _diagnostics);
        _prefetch = new SpeakerSpeechPrefetchGate(
            _timeline,
            _windows,
            _quality,
            _scheduler,
            policy.PrefetchMaxWindows,
            hangoverSamples: policy.HopSamples * 2);
    }

    public SpeakerVerificationPolicy Policy => _policy;

    public SpeakerAudioTimelineBuffer Timeline => _timeline;

    public SpeakerEvidenceCache Cache => _cache;

    public SpeakerInferenceScheduler Scheduler => _scheduler;

    public SpeakerSpeechPrefetchGate Prefetch => _prefetch;

    public SpeakerVerificationDiagnostics Diagnostics => _diagnostics;

    public bool IsStarted => _scheduler.IsStarted;

    public void Start()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _timeline.Clear();
        _cache.Clear();
        _prefetch.Reset();
        _lastTrimEpoch = 0;
        _lastTrimStart = long.MinValue;
        _scheduler.Start();
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await _scheduler.StopAsync().ConfigureAwait(false);
        _diagnostics.LogSummary();
        _cache.Clear();
        _prefetch.Reset();
    }

    /// <summary>Identity frame from the duplex dispatch thread: copy, trim, run the energy gate.</summary>
    public void OnSpeakerCapture(in SpeakerCaptureFrame frame)
    {
        if (Volatile.Read(ref _disposed) != 0 || !_scheduler.IsStarted)
            return;

        if (!_timeline.Append(in frame))
            return;

        if (_timeline.TryGetBounds(out long epoch, out _, out long retainedStart, out _) &&
            (epoch != _lastTrimEpoch || _lastTrimStart == long.MinValue ||
             retainedStart - _lastTrimStart >= _policy.HopSamples))
        {
            _lastTrimEpoch = epoch;
            _lastTrimStart = retainedStart;
            _cache.Trim(epoch, retainedStart);
        }

        _prefetch.Observe(in frame);
        // Audio arrival, including silence, can satisfy a waiter even when no inference ran.
        _cache.NotifyAudioAvailable();
    }

    public SpeakerCandidateId NewCandidateId() => new(Interlocked.Increment(ref _nextCandidateId));

    public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => _policy.Get(purpose).InferenceTimeout;

    public void Cancel(SpeakerCandidateId candidateId)
    {
        _scheduler.Cancel(candidateId);
        _diagnostics.Forget(candidateId);
    }

    public void SetPlaybackActive(bool active) => _prefetch.PrefetchPriority =
        active && _policy.Get(SpeakerVerificationPurpose.BargeInCommand).Enabled
            ? SpeakerInferencePriority.BargeInPrefetch : SpeakerInferencePriority.WakeupPrefetch;

    public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query)
    {
        _diagnostics.RecordQuery();
        SpeakerVerificationResult result = EvaluateCore(in query);
        _diagnostics.RecordResult(result.Status, result.Reason);
        _diagnostics.ObserveDecision(in query, in result);
        return result;
    }

    private SpeakerVerificationResult EvaluateCore(in SpeakerEvidenceQuery query)
    {
        SpeakerPurposePolicy policy = _policy.Get(query.Purpose);
        if (!policy.Enabled)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.NotReady, policy.RequiredMatches);
        if (Volatile.Read(ref _disposed) != 0 || !_scheduler.IsStarted)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.NotReady, policy.RequiredMatches);
        if (!_banks.TryGetValue(query.Purpose.Bank(), out SpeakerPrototypeBank? bank) || bank is null)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.MissingPurposeBank, policy.RequiredMatches);

        if (query.CandidateEndSampleExclusive <= query.CandidateStartSample)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.CandidateRangeInvalid, policy.RequiredMatches);
        if (query.HasFocusInterval &&
            (query.FocusStartSample < query.CandidateStartSample ||
             query.FocusEndSampleExclusive > query.CandidateEndSampleExclusive))
        {
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.CandidateRangeInvalid, policy.RequiredMatches);
        }

        if (!_timeline.TryGetBounds(out long epoch, out long segmentStart, out long retainedStart, out long retainedEnd))
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.NotReady, policy.RequiredMatches);
        if (query.StreamEpoch != epoch)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.StreamEpochMismatch, policy.RequiredMatches);
        if (query.CandidateStartSample < retainedStart)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.AudioExpired, policy.RequiredMatches);
        if (query.CandidateStartSample < segmentStart)
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.TimelineGap, policy.RequiredMatches);

        long availableEnd = Math.Min(query.CandidateEndSampleExclusive, retainedEnd);
        bool ringCaughtUp = retainedEnd >= query.CandidateEndSampleExclusive;
        bool candidateExhausted = query.InputComplete && ringCaughtUp;
        EvidenceWindowSelection selection = query.Purpose == SpeakerVerificationPurpose.Wakeup
            ? _windows.SelectWakeupWindows(in query, policy.WindowCount, segmentStart, retainedEnd)
            : _windows.SelectRecentWindows(in query, policy.WindowCount, segmentStart, retainedEnd);

        // Speech-time gate over the union of the candidate interval that has actually arrived.
        int speechSamples = 0;
        if (availableEnd > query.CandidateStartSample &&
            !_timeline.TryCountSpeechSamples(
                query.StreamEpoch,
                query.CandidateStartSample,
                availableEnd,
                _policy.MinFrameRms,
                out speechSamples,
                out SpeakerTimelineFailure speechFailure))
        {
            return SpeakerVerificationResult.Unavailable(
                policy.Threshold,
                in query,
                SpeakerInferenceScheduler.MapTimelineFailure(speechFailure),
                policy.RequiredMatches);
        }
        bool speechSufficient = speechSamples >= policy.MinimumSpeechSamples;

        SpeakerInferencePriority priority = CandidatePriority(query.Purpose);
        SpeakerPurposeBank bankKind = query.Purpose.Bank();
        int observed = 0, matched = 0, ambiguous = 0, rejected = 0, invalid = 0, failed = 0, pendingInference = 0;
        SpeakerInferenceJob? nextMissing = null;
        float bestScore = -1f;
        SpeakerWindowScore bestWindow = default;
        long evidenceEnd = selection.FirstWindowStart;
        long latestMatchedEnd = 0;
        bool latestWindowMatched = false;
        bool latestWindowHighConfidence = false;
        SpeakerVerificationReason latestFailure = SpeakerVerificationReason.None;
        bool overloaded = false;

        // Windows to evaluate, oldest first. Wakeup windows are selected from the whole candidate
        // by speech coverage rather than by the KWS span midpoint. KWS token spans can include a
        // long tail of blanks, so midpoint-only geometry frequently picked three quiet windows
        // even though a usable speech window existed nearby.
        Span<long> windowEnds = stackalloc long[Math.Max(1, policy.WindowCount)];
        int windowCount = query.Purpose == SpeakerVerificationPurpose.Wakeup
            ? SelectWakeupWindowEndsBySpeech(
                in query,
                segmentStart,
                retainedEnd,
                windowEnds)
            : SelectRecentSpeechWindowEnds(in query, in selection, segmentStart, windowEnds);
        if (windowCount > 0)
            selection = selection with { FirstWindowStart = windowEnds[0] - _policy.WindowSamples, WindowCount = windowCount };

        for (int i = 0; i < windowCount; i++)
        {
            long end = windowEnds[i];
            long start = end - _policy.WindowSamples;

            if (!_cache.TryGet(query.StreamEpoch, start, end, out SpeakerWindowEvidence evidence))
            {
                _diagnostics.RecordCacheMiss();
                nextMissing ??= new SpeakerInferenceJob(query.StreamEpoch, start, end, priority, query.CandidateId, IsPrefetch: false);
                pendingInference++;
                invalid++;
                continue;
            }

            _diagnostics.RecordCacheHit();

            switch (evidence.Kind)
            {
                case SpeakerWindowEvidenceKind.Scored when evidence.TryGetScore(bankKind, out SpeakerWindowScore score):
                    evidenceEnd = end;
                    latestWindowMatched = false;
                    latestWindowHighConfidence = false;
                    observed++;
                    if (score.Reported > bestScore)
                    {
                        bestScore = score.Reported;
                        bestWindow = score;
                    }
                    if (SpeakerProfileMatcher.IsMatch(in score, policy.Threshold))
                    {
                        matched++;
                        latestMatchedEnd = end;
                        latestWindowMatched = true;
                        latestWindowHighConfidence = SpeakerProfileMatcher.IsHighConfidence(
                            in score,
                            policy.Threshold,
                            policy.HighConfidenceThreshold);
                    }
                    else if (SpeakerProfileMatcher.IsRejected(in score, policy.Threshold, policy.IndeterminateScoreMargin))
                    {
                        rejected++;
                    }
                    else
                    {
                        ambiguous++;
                    }
                    break;
                case SpeakerWindowEvidenceKind.Scored:
                    // Embedding exists but this bank was not usable at scoring time.
                    failed++;
                    latestFailure = SpeakerVerificationReason.MissingPurposeBank;
                    break;
                case SpeakerWindowEvidenceKind.Ineligible:
                    // Silence or non-speech cannot prove another identity; it is simply not evidence.
                    invalid++;
                    break;
                default:
                    failed++;
                    invalid++;
                    latestFailure = evidence.Reason;
                    break;
            }
        }

        bool recentOwnerPrior = HasRecentOwnerPrior(in query, policy);
        bool strongUpstream = policy.AllowStrongUpstreamOneVote && query.StrongUpstreamEvidence;
        bool fusedMatch = latestWindowMatched && rejected == 0 && (strongUpstream || recentOwnerPrior);

        if (latestWindowHighConfidence && rejected == 0 && speechSufficientOrFocus(speechSufficient, in query))
        {
            return Match(in query, in selection, evidenceEnd, latestMatchedEnd, bestScore, in bestWindow,
                SpeakerVerificationReason.HighConfidenceMatch, observed, matched, 1, ambiguous, rejected, invalid,
                highConfidence: true);
        }
        if (fusedMatch && speechSufficientOrFocus(speechSufficient, in query))
        {
            return Match(in query, in selection, evidenceEnd, latestMatchedEnd, bestScore, in bestWindow,
                strongUpstream ? SpeakerVerificationReason.StrongUpstreamMatch : SpeakerVerificationReason.RecentOwnerPriorMatch,
                observed, matched, 1, ambiguous, rejected, invalid,
                usedStrongUpstream: strongUpstream, usedRecentPrior: !strongUpstream && recentOwnerPrior);
        }
        if (matched >= policy.RequiredMatches && speechSufficient)
        {
            return Match(in query, in selection, evidenceEnd, latestMatchedEnd, bestScore, in bestWindow,
                SpeakerVerificationReason.MajorityMatch, observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid);
        }
        if (_policy.PrefixHighConfidenceEnabled &&
            query.Purpose != SpeakerVerificationPurpose.SingleTalk &&
            rejected == 0 &&
            TryPrefixHighConfidence(in query, policy, bankKind, priority, out SpeakerWindowScore prefix))
        {
            return Match(in query, in selection, evidenceEnd, latestMatchedEnd, prefix.Reported, in prefix,
                SpeakerVerificationReason.PrefixHighConfidenceMatch, observed, matched, 1, ambiguous, rejected, invalid,
                highConfidence: true);
        }

        // Realtime purposes may reject from the first negative windows. Single-talk keeps sliding
        // until the endpoint and identity audio are both complete: a weak or contaminated prefix
        // must not deny the later owner speech of the same utterance.
        if ((query.Purpose != SpeakerVerificationPurpose.SingleTalk || candidateExhausted) &&
            rejected >= policy.RequiredMatches && matched == 0)
        {
            return Final(SpeakerVerificationStatus.Mismatch, in query, in selection, evidenceEnd, bestScore, in bestWindow,
                SpeakerVerificationReason.MajorityMismatch, observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid);
        }

        // Read every cached verdict first; an already decisive first window must not enqueue
        // windows 2/3. Only one missing window per candidate may be queued/running at a time.
        if (nextMissing is { } missing)
        {
            SpeakerScheduleResult schedule = _scheduler.TrySchedule(in missing);
            overloaded = schedule == SpeakerScheduleResult.Overloaded;
            if (schedule == SpeakerScheduleResult.Stopped)
                return SpeakerVerificationResult.Unavailable(policy.Threshold, in query, SpeakerVerificationReason.RuntimeStopping, policy.RequiredMatches);
        }

        if (overloaded && observed == 0)
        {
            // Single-talk is allowed to keep accumulating while ASR is running. Queue pressure is
            // transient for an open utterance; returning Unavailable here would let the VAD gate
            // deny the command before its endpoint. The completed-query path still fails closed.
            if (!query.InputComplete && query.Purpose == SpeakerVerificationPurpose.SingleTalk)
            {
                return SpeakerVerificationResult.Pending(
                    policy.Threshold,
                    in query,
                    SpeakerVerificationReason.WaitingForInference,
                    observed,
                    matched,
                    policy.RequiredMatches,
                    ambiguous,
                    rejected,
                    invalid,
                    selection.RequiredEvidenceEnd);
            }

            return SpeakerVerificationResult.Unavailable(
                policy.Threshold,
                in query,
                SpeakerVerificationReason.InferenceOverload,
                policy.RequiredMatches);
        }

        if (pendingInference > 0)
        {
            return SpeakerVerificationResult.Pending(policy.Threshold, in query, SpeakerVerificationReason.WaitingForInference,
                observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid, selection.RequiredEvidenceEnd);
        }

        bool geometryComplete = selection.WindowCount >= policy.WindowCount;
        if (!geometryComplete && !candidateExhausted)
        {
            return SpeakerVerificationResult.Pending(policy.Threshold, in query,
                speechSufficient ? SpeakerVerificationReason.WaitingForEvidence : SpeakerVerificationReason.CollectingCandidate,
                observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid, selection.RequiredEvidenceEnd);
        }
        if (!speechSufficient && !candidateExhausted)
        {
            return SpeakerVerificationResult.Pending(policy.Threshold, in query, SpeakerVerificationReason.CollectingCandidate,
                observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid, selection.RequiredEvidenceEnd);
        }
        if (geometryComplete && !candidateExhausted && query.Purpose != SpeakerVerificationPurpose.Wakeup)
        {
            // Recent-window purposes keep sliding while the candidate is still open.
            return SpeakerVerificationResult.Pending(policy.Threshold, in query, SpeakerVerificationReason.WaitingForEvidence,
                observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid, selection.RequiredEvidenceEnd);
        }

        // Terminal without a match.
        if (observed == 0 && failed > 0)
        {
            return SpeakerVerificationResult.Unavailable(policy.Threshold, in query,
                latestFailure == SpeakerVerificationReason.None ? SpeakerVerificationReason.ProviderError : latestFailure,
                policy.RequiredMatches);
        }
        if (rejected > matched && rejected > 0)
        {
            return Final(SpeakerVerificationStatus.Mismatch, in query, in selection, evidenceEnd, bestScore, in bestWindow,
                SpeakerVerificationReason.MajorityMismatch, observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid);
        }

        return SpeakerVerificationResult.InsufficientEvidence(policy.Threshold, in query,
            speechSufficient ? SpeakerVerificationReason.AmbiguousEvidence : SpeakerVerificationReason.InsufficientSpeech,
            observed, matched, policy.RequiredMatches, ambiguous, rejected, invalid);

        static bool speechSufficientOrFocus(bool sufficient, in SpeakerEvidenceQuery query)
            // A decisive current window is still a current-window match; the speech-time floor only
            // applies to the majority path so a short keyword focus is not starved.
            => sufficient || query.HasFocusInterval;
    }

    public async Task<SpeakerVerificationResult> WaitForDecisionAsync(
        SpeakerEvidenceQuery query,
        TimeSpan timeout,
        CancellationToken cancellationToken = default)
    {
        if (timeout < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout));

        long startedAt = Stopwatch.GetTimestamp();
        while (true)
        {
            Task update = _cache.UpdateSignal;
            SpeakerVerificationResult result = Evaluate(in query);
            if (result.IsFinal)
                return result;

            TimeSpan remaining = timeout - Stopwatch.GetElapsedTime(startedAt);
            if (remaining <= TimeSpan.Zero)
                return SpeakerVerificationResult.Unavailable(result.Threshold, in query, SpeakerVerificationReason.InferenceTimeout, result.RequiredMatches);

            try
            {
                await update.WaitAsync(remaining, cancellationToken).ConfigureAwait(false);
            }
            catch (TimeoutException)
            {
                return SpeakerVerificationResult.Unavailable(result.Threshold, in query, SpeakerVerificationReason.InferenceTimeout, result.RequiredMatches);
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _scheduler.Dispose();
    }

    /// <summary>
    /// Selects the best complete windows in a wakeup candidate by speech coverage. The KWS focus
    /// remains a tie-breaker, but it is not the primary geometry because CTC token spans may be
    /// wider than the spoken keyword and can otherwise center all windows on silence.
    /// </summary>
    private int SelectWakeupWindowEndsBySpeech(
        in SpeakerEvidenceQuery query,
        long segmentStart,
        long retainedEnd,
        Span<long> destination)
    {
        long availableEnd = Math.Min(query.CandidateEndSampleExclusive, retainedEnd);
        long firstStart = SpeakerEvidenceWindowSelector.AlignUp(
            query.CandidateStartSample,
            segmentStart,
            _policy.HopSamples);
        long firstEnd = checked(firstStart + _policy.WindowSamples);
        if (availableEnd < firstEnd || destination.IsEmpty)
            return 0;

        long focusCenter = query.HasFocusInterval
            ? checked(query.FocusStartSample +
                (query.FocusEndSampleExclusive - query.FocusStartSample) / 2)
            : query.CandidateStartSample;
        Span<WakeupWindowCandidate> best = stackalloc WakeupWindowCandidate[destination.Length];
        int bestCount = 0;

        long end = firstEnd;
        while (end <= availableEnd)
        {
            long start = end - _policy.WindowSamples;
            double speechRatio = GetWindowSpeechRatio(query.StreamEpoch, start, end);
            if (speechRatio >= 0)
            {
                var candidate = new WakeupWindowCandidate(
                    end,
                    speechRatio,
                    Distance(end, focusCenter));
                int insert = 0;
                while (insert < bestCount && IsBetter(best[insert], candidate))
                    insert++;

                if (insert < best.Length)
                {
                    int newCount = Math.Min(bestCount + 1, best.Length);
                    for (int i = newCount - 1; i > insert; i--)
                        best[i] = best[i - 1];
                    best[insert] = candidate;
                    bestCount = newCount;
                }
            }

            end = checked(end + _policy.HopSamples);
        }

        // Preserve chronological order for the existing vote/diagnostic semantics. Selection is
        // ranked by speech coverage above; inference and majority voting remain time ordered.
        for (int i = 0; i < bestCount; i++)
            destination[i] = best[i].End;
        for (int i = 1; i < bestCount; i++)
        {
            long value = destination[i];
            int j = i - 1;
            while (j >= 0 && destination[j] > value)
            {
                destination[j + 1] = destination[j];
                j--;
            }
            destination[j + 1] = value;
        }
        return bestCount;
    }

    private double GetWindowSpeechRatio(long streamEpoch, long start, long end)
    {
        if (_cache.TryGet(streamEpoch, start, end, out SpeakerWindowEvidence evidence))
            return evidence.Quality.SpeechRatio;

        if (!_timeline.TryCountSpeechSamples(
                streamEpoch,
                start,
                end,
                _policy.MinFrameRms,
                out int speechSamples,
                out _))
        {
            return -1;
        }

        return speechSamples / (double)(end - start);
    }

    private static bool IsBetter(in WakeupWindowCandidate left, in WakeupWindowCandidate right)
        => left.SpeechRatio > right.SpeechRatio ||
           (left.SpeechRatio == right.SpeechRatio && left.FocusDistance <= right.FocusDistance);

    private static long Distance(long value, long target)
        => value >= target ? value - target : target - value;

    private readonly record struct WakeupWindowCandidate(
        long End,
        double SpeechRatio,
        long FocusDistance);

    /// <summary>
    /// The newest complete windows of the candidate that are not already known to be silence,
    /// oldest first. Unknown windows are kept (they are scheduled by the caller); only cached
    /// <see cref="SpeakerWindowEvidenceKind.Ineligible"/> windows are skipped.
    /// </summary>
    private int SelectRecentSpeechWindowEnds(
        in SpeakerEvidenceQuery query,
        in EvidenceWindowSelection selection,
        long segmentStart,
        Span<long> destination)
    {
        long earliestStart = SpeakerEvidenceWindowSelector.AlignUp(query.CandidateStartSample, segmentStart, _policy.HopSamples);
        long earliestEnd = checked(earliestStart + _policy.WindowSamples);
        int picked = 0;
        for (int k = selection.RetainedCandidateWindowCount - 1; k >= 0 && picked < destination.Length; k--)
        {
            long end = checked(earliestEnd + (long)k * _policy.HopSamples);
            if (_cache.TryGet(query.StreamEpoch, end - _policy.WindowSamples, end, out SpeakerWindowEvidence evidence) &&
                evidence.Kind == SpeakerWindowEvidenceKind.Ineligible)
            {
                continue;
            }
            destination[picked++] = end;
        }

        destination.Slice(0, picked).Reverse();
        return picked;
    }

    private bool TryPrefixHighConfidence(
        in SpeakerEvidenceQuery query,
        SpeakerPurposePolicy policy,
        SpeakerPurposeBank bank,
        SpeakerInferencePriority priority,
        out SpeakerWindowScore score)
    {
        score = default;
        long speechStart = query.HasFocusInterval ? query.FocusStartSample : query.CandidateStartSample;
        long end = checked(speechStart + _policy.PrefixHighConfidenceSamples);
        if (end > query.CandidateEndSampleExclusive)
            return false;
        if (_timeline.Classify(query.StreamEpoch, speechStart, end) != SpeakerTimelineFailure.None)
            return false;

        if (!_cache.TryGet(query.StreamEpoch, speechStart, end, out SpeakerWindowEvidence evidence))
        {
            var job = new SpeakerInferenceJob(query.StreamEpoch, speechStart, end, priority, query.CandidateId, IsPrefetch: false, IsPrefix: true);
            _scheduler.TrySchedule(in job);
            return false;
        }

        return evidence.Kind == SpeakerWindowEvidenceKind.Scored &&
            evidence.TryGetScore(bank, out score) &&
            SpeakerProfileMatcher.IsHighConfidence(in score, policy.Threshold, policy.HighConfidenceThreshold);
    }

    private SpeakerVerificationResult Match(
        in SpeakerEvidenceQuery query,
        in EvidenceWindowSelection selection,
        long evidenceEnd,
        long latestMatchedEnd,
        float score,
        in SpeakerWindowScore window,
        SpeakerVerificationReason reason,
        int observed,
        int matched,
        int required,
        int ambiguous,
        int rejected,
        int invalid,
        bool highConfidence = false,
        bool usedStrongUpstream = false,
        bool usedRecentPrior = false)
    {
        lock (_priorLock)
        {
            _lastOwnerMatchEpoch = query.StreamEpoch;
            _lastOwnerMatchEvidenceEnd = latestMatchedEnd > 0 ? latestMatchedEnd : evidenceEnd;
        }

        _logger.LogTrace(
            "本人说话人验证通过: purpose={Purpose}, candidate={CandidateId}, reason={Reason}, score={Score:F4} (centroid={Centroid:F4}, best={Best:F4}, top2={Top2:F4}), observed={Observed}, matched={Matched}/{Required}, ambiguous={Ambiguous}, rejected={Rejected}, invalid={Invalid}, epoch={Epoch}, evidence=[{Start},{End}).",
            query.Purpose,
            query.CandidateId,
            reason,
            score,
            window.CentroidScore,
            window.BestPrototypeScore,
            window.Top2MeanScore,
            observed,
            matched,
            required,
            ambiguous,
            rejected,
            invalid,
            query.StreamEpoch,
            selection.FirstWindowStart,
            evidenceEnd);

        return new SpeakerVerificationResult(
            SpeakerVerificationStatus.Match,
            score,
            _policy.GetThreshold(query.Purpose),
            query.StreamEpoch,
            selection.FirstWindowStart,
            evidenceEnd,
            reason,
            observed,
            matched,
            required,
            highConfidence,
            ambiguous,
            rejected,
            invalid,
            usedStrongUpstream,
            usedRecentPrior,
            selection.RequiredEvidenceEnd,
            window.CentroidScore,
            window.BestPrototypeScore,
            window.Top2MeanScore,
            query.CandidateId);
    }

    private SpeakerVerificationResult Final(
        SpeakerVerificationStatus status,
        in SpeakerEvidenceQuery query,
        in EvidenceWindowSelection selection,
        long evidenceEnd,
        float score,
        in SpeakerWindowScore window,
        SpeakerVerificationReason reason,
        int observed,
        int matched,
        int required,
        int ambiguous,
        int rejected,
        int invalid)
        => new(
            status,
            score,
            _policy.GetThreshold(query.Purpose),
            query.StreamEpoch,
            selection.FirstWindowStart,
            evidenceEnd,
            reason,
            observed,
            matched,
            required,
            false,
            ambiguous,
            rejected,
            invalid,
            false,
            false,
            selection.RequiredEvidenceEnd,
            window.CentroidScore,
            window.BestPrototypeScore,
            window.Top2MeanScore,
            query.CandidateId);

    private bool HasRecentOwnerPrior(in SpeakerEvidenceQuery query, SpeakerPurposePolicy policy)
    {
        if (policy.RecentOwnerPriorSamples <= 0)
            return false;
        long epoch;
        long lastEnd;
        lock (_priorLock)
        {
            epoch = _lastOwnerMatchEpoch;
            lastEnd = _lastOwnerMatchEvidenceEnd;
        }
        if (epoch != query.StreamEpoch || lastEnd <= 0)
            return false;

        long referenceStart = query.HasFocusInterval ? query.FocusStartSample : query.CandidateStartSample;
        if (referenceStart <= lastEnd)
            return false;
        return referenceStart - lastEnd <= policy.RecentOwnerPriorSamples;
    }

    private static SpeakerInferencePriority CandidatePriority(SpeakerVerificationPurpose purpose)
        => purpose switch
        {
            SpeakerVerificationPurpose.BargeInCommand => SpeakerInferencePriority.BargeInCommandCandidate,
            SpeakerVerificationPurpose.Wakeup => SpeakerInferencePriority.WakeupCandidate,
            _ => SpeakerInferencePriority.SingleTalk,
        };
}
