using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Modules.Speaker.Buffering;
using VoiceBox.Assistant.Modules.Speaker.Diagnostics;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Evidence;

/// <summary>
/// Single provider worker fed by a bounded priority queue. Windows are deduplicated through the
/// evidence cache, prefetch jobs are rejected when the queue is full, and a formal candidate may
/// evict a queued lower-priority prefetch job. A provider exception marks only the failing window.
/// Stop drains the queue and joins the worker exactly once; the provider is never called while any
/// lock is held.
/// </summary>
internal sealed class SpeakerInferenceScheduler : IDisposable
{
    private readonly ISpeakerService _speaker;
    private readonly SpeakerAudioTimelineBuffer _timeline;
    private readonly SpeakerEvidenceCache _cache;
    private readonly SpeakerWindowQualityEvaluator _quality;
    private readonly ILogger _logger;
    private readonly int _capacity;
    private readonly object _lock = new();
    private readonly SortedSet<QueuedJob> _queue = new(QueuedJobComparer.Instance);
    private readonly SemaphoreSlim _available = new(0, 1);
    private readonly IReadOnlyDictionary<SpeakerPurposeBank, SpeakerPrototypeBank> _banks;
    private readonly SpeakerVerificationDiagnostics? _diagnostics;
    private CancellationTokenSource? _stopSource;
    private Task? _worker;
    private long _sequence;
    private bool _started;
    private bool _stopping;
    private long _overloadCount;
    private long _jobsRun;
    private readonly Dictionary<SpeakerCandidateId, (long Epoch, long Start, long End)> _candidateWindows = [];
    private readonly int _prefetchCapacity;
    private int _queuedPrefetchCount;

    public SpeakerInferenceScheduler(
        ISpeakerService speaker,
        SpeakerAudioTimelineBuffer timeline,
        SpeakerEvidenceCache cache,
        SpeakerWindowQualityEvaluator quality,
        IReadOnlyDictionary<SpeakerPurposeBank, SpeakerPrototypeBank> banks,
        int capacity,
        ILogger logger,
        SpeakerVerificationDiagnostics? diagnostics = null)
    {
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _timeline = timeline ?? throw new ArgumentNullException(nameof(timeline));
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        _quality = quality ?? throw new ArgumentNullException(nameof(quality));
        _banks = banks ?? throw new ArgumentNullException(nameof(banks));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _diagnostics = diagnostics;
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _prefetchCapacity = capacity == 1 ? 1 : Math.Min(capacity - Math.Min(2, capacity - 1), Math.Max(1, capacity * 3 / 5));
    }

    public long OverloadCount => Volatile.Read(ref _overloadCount);

    public long JobsRun => Volatile.Read(ref _jobsRun);

    public int QueueDepth
    {
        get { lock (_lock) return _queue.Count; }
    }

    public bool IsStarted
    {
        get { lock (_lock) return _started && !_stopping; }
    }

    public void Start()
    {
        lock (_lock)
        {
            if (_started)
                throw new InvalidOperationException("Speaker inference scheduler has already started.");
            _started = true;
            _stopping = false;
            _candidateWindows.Clear();
            _queuedPrefetchCount = 0;
            _stopSource = new CancellationTokenSource();
            CancellationToken token = _stopSource.Token;
            _worker = Task.Factory.StartNew(
                () => WorkerLoop(token),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
        }
    }

    public SpeakerScheduleResult TrySchedule(in SpeakerInferenceJob job)
    {
        lock (_lock)
        {
            if (!_started || _stopping)
                return SpeakerScheduleResult.Stopped;

            var key = (job.StreamEpoch, job.StartSample, job.EndSampleExclusive);
            if (!job.IsPrefetch && !job.CandidateId.IsNone &&
                _candidateWindows.TryGetValue(job.CandidateId, out var current) && current != key)
                return SpeakerScheduleResult.CandidateBusy;

            if (!_cache.TryReserve(job.StreamEpoch, job.StartSample, job.EndSampleExclusive))
            {
                // Promote in place: the original reservation and wakeup belong to the same job.
                // Adding again here used to extract the same window twice and exceed capacity.
                if (!job.IsPrefetch)
                {
                    QueuedJob? found = null;
                    foreach (QueuedJob queued in _queue)
                    {
                        if ((queued.Job.StreamEpoch, queued.Job.StartSample, queued.Job.EndSampleExclusive) == key)
                        {
                            found = queued;
                            break;
                        }
                    }
                    if (found is { } existing && existing.Job.Priority < job.Priority)
                    {
                        _queue.Remove(existing);
                        if (existing.Job.IsPrefetch) _queuedPrefetchCount--;
                        _queue.Add(existing with { Job = job });
                    }
                    if (!job.CandidateId.IsNone && _cache.IsInFlight(key.Item1, key.Item2, key.Item3))
                        _candidateWindows[job.CandidateId] = key;
                }
                return SpeakerScheduleResult.AlreadyKnown;
            }

            if (job.IsPrefetch && (_queuedPrefetchCount >= _prefetchCapacity || _queue.Count >= _prefetchCapacity))
            {
                _cache.Release(job.StreamEpoch, job.StartSample, job.EndSampleExclusive, notify: false);
                _diagnostics?.RecordPrefetchRejected();
                return SpeakerScheduleResult.PrefetchRejected;
            }
            if (_queue.Count >= _capacity && !TryEvictQueuedPrefetch(in job))
            {
                _cache.Release(job.StreamEpoch, job.StartSample, job.EndSampleExclusive, notify: false);
                Interlocked.Increment(ref _overloadCount);
                _diagnostics?.RecordInferenceOverload();
                return SpeakerScheduleResult.Overloaded;
            }

            _queue.Add(new QueuedJob(job, ++_sequence, Stopwatch.GetTimestamp()));
            if (job.IsPrefetch) _queuedPrefetchCount++;
            else if (!job.CandidateId.IsNone) _candidateWindows[job.CandidateId] = key;
            _diagnostics?.ObserveQueueDepth(_queue.Count);
            _diagnostics?.RecordInferenceScheduled(job.IsPrefetch);
        }
        // Coalesced wakeup: promotion, eviction and cancellation cannot leak semaphore permits.
        try { _available.Release(); } catch (SemaphoreFullException) { }
        return SpeakerScheduleResult.Queued;
    }

    /// <summary>Frees one queued prefetch reservation for a formal candidate.</summary>
    private bool TryEvictQueuedPrefetch(in SpeakerInferenceJob incoming)
    {
        QueuedJob? evicted = null;
        foreach (QueuedJob queued in _queue)
        {
            if (queued.Job.IsPrefetch && queued.Job.Priority < incoming.Priority)
            {
                evicted = queued;
                break;
            }
        }

        if (evicted is not { } prefetch)
            return false;

        _queue.Remove(prefetch);
        _queuedPrefetchCount--;
        _cache.Release(
            prefetch.Job.StreamEpoch,
            prefetch.Job.StartSample,
            prefetch.Job.EndSampleExclusive);
        return true;
    }

    /// <summary>Removes not-yet-started jobs of a candidate; cached embeddings are kept.</summary>
    public int Cancel(SpeakerCandidateId candidateId)
    {
        if (candidateId.IsNone)
            return 0;

        List<QueuedJob>? removed = null;
        lock (_lock)
        {
            _candidateWindows.Remove(candidateId);
            foreach (QueuedJob queued in _queue)
            {
                if (queued.Job.CandidateId == candidateId && !queued.Job.IsPrefetch &&
                    !_candidateWindows.Values.Contains((queued.Job.StreamEpoch, queued.Job.StartSample, queued.Job.EndSampleExclusive)))
                    (removed ??= []).Add(queued);
            }
            if (removed is null)
                return 0;
            foreach (QueuedJob queued in removed)
                _queue.Remove(queued);
            _queuedPrefetchCount -= removed.Count(static value => value.Job.IsPrefetch);
        }

        foreach (QueuedJob queued in removed)
            _cache.Release(queued.Job.StreamEpoch, queued.Job.StartSample, queued.Job.EndSampleExclusive);
        return removed.Count;
    }

    public async Task StopAsync()
    {
        Task? worker;
        CancellationTokenSource? stop;
        QueuedJob[] pending;
        lock (_lock)
        {
            if (!_started)
                return;
            _stopping = true;
            worker = _worker;
            stop = _stopSource;
            pending = _queue.ToArray();
            _queue.Clear();
            _queuedPrefetchCount = 0;
        }

        foreach (QueuedJob queued in pending)
            _cache.Release(queued.Job.StreamEpoch, queued.Job.StartSample, queued.Job.EndSampleExclusive);

        try { stop?.Cancel(); }
        catch (ObjectDisposedException) { }
        if (worker is not null)
        {
            try { await worker.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }

        lock (_lock)
        {
            _worker = null;
            _stopSource?.Dispose();
            _stopSource = null;
            _started = false;
            _stopping = false;
            _candidateWindows.Clear();
            _queuedPrefetchCount = 0;
        }
    }

    public void Dispose()
    {
        try { StopAsync().GetAwaiter().GetResult(); }
        catch { }
        _available.Dispose();
    }

    private void WorkerLoop(CancellationToken cancellationToken)
    {
        float[] window = new float[_timeline.Capacity];
        while (!cancellationToken.IsCancellationRequested)
        {
            QueuedJob? next = null;
            lock (_lock)
            {
                if (_stopping) return;
                if (_queue.Count > 0)
                {
                    next = _queue.Max;
                    _queue.Remove(next.Value);
                    if (next.Value.Job.IsPrefetch) _queuedPrefetchCount--;
                }
            }
            if (next is not { } queued)
            {
                try { _available.Wait(cancellationToken); }
                catch (OperationCanceledException) { return; }
                continue;
            }

            long started = Stopwatch.GetTimestamp();
            _diagnostics?.RecordQueueWait(started - queued.EnqueuedAt);
            try { RunJobCore(queued.Job, window); }
            finally
            {
                _diagnostics?.RecordInferenceCompleted(Stopwatch.GetTimestamp() - started);
                lock (_lock)
                {
                    var key = (queued.Job.StreamEpoch, queued.Job.StartSample, queued.Job.EndSampleExclusive);
                    foreach (var candidate in _candidateWindows.Where(pair => pair.Value == key).Select(pair => pair.Key).ToArray())
                        _candidateWindows.Remove(candidate);
                }
                _cache.Pulse();
            }
        }
    }

    private void RunJobCore(in SpeakerInferenceJob job, float[] scratch)
    {
        Interlocked.Increment(ref _jobsRun);
        int length = checked((int)(job.EndSampleExclusive - job.StartSample));
        if (length <= 0 || length > scratch.Length)
        {
            _cache.Store(Failed(job, SpeakerVerificationReason.CandidateRangeInvalid));
            return;
        }

        Span<float> audio = scratch.AsSpan(0, length);
        if (!_timeline.TrySnapshot(job.StreamEpoch, job.StartSample, job.EndSampleExclusive, audio, out SpeakerTimelineFailure failure))
        {
            _diagnostics?.RecordTimelineFailure(failure);
            _cache.Store(Failed(job, MapTimelineFailure(failure)));
            return;
        }

        SpeakerWindowQuality quality = _quality.Evaluate(audio);
        if (!_quality.IsEligible(in quality))
        {
            _diagnostics?.RecordQualityRejected(in quality);
            _cache.Store(new SpeakerWindowEvidence(
                job.StreamEpoch,
                job.StartSample,
                job.EndSampleExclusive,
                SpeakerWindowEvidenceKind.Ineligible,
                quality.AllFinite ? SpeakerVerificationReason.InsufficientSpeech : SpeakerVerificationReason.InvalidEmbedding,
                embedding: null,
                quality,
                job.IsPrefix));
            return;
        }

        float[] embedding;
        try
        {
            embedding = _speaker.ExtractEmbedding(audio);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Speaker embedding extraction failed for window [{Start},{End}).", job.StartSample, job.EndSampleExclusive);
            _cache.Store(Failed(job, SpeakerVerificationReason.ProviderError, quality));
            return;
        }

        if (embedding is null || embedding.Length != _speaker.EmbeddingSize || !IsFiniteNonZero(embedding))
        {
            _cache.Store(Failed(job, SpeakerVerificationReason.InvalidEmbedding, quality));
            return;
        }

        var evidence = new SpeakerWindowEvidence(
            job.StreamEpoch,
            job.StartSample,
            job.EndSampleExclusive,
            SpeakerWindowEvidenceKind.Scored,
            SpeakerVerificationReason.None,
            embedding,
            quality,
            job.IsPrefix);
        try
        {
            foreach ((SpeakerPurposeBank kind, SpeakerPrototypeBank bank) in _banks)
            {
                if (bank.IsUsable)
                    evidence.SetScore(kind, SpeakerProfileMatcher.Score(_speaker, bank, embedding));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Speaker bank comparison failed for window [{Start},{End}).", job.StartSample, job.EndSampleExclusive);
            _cache.Store(Failed(job, SpeakerVerificationReason.ProviderError, quality));
            return;
        }

        _cache.Store(evidence);
    }

    private static SpeakerWindowEvidence Failed(
        in SpeakerInferenceJob job,
        SpeakerVerificationReason reason,
        SpeakerWindowQuality quality = default)
        => new(
            job.StreamEpoch,
            job.StartSample,
            job.EndSampleExclusive,
            SpeakerWindowEvidenceKind.Failed,
            reason,
            embedding: null,
            quality,
            job.IsPrefix);

    private static bool IsFiniteNonZero(float[] embedding)
    {
        double sumSq = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            if (!float.IsFinite(embedding[i]))
                return false;
            sumSq += (double)embedding[i] * embedding[i];
        }
        return sumSq > 0;
    }

    internal static SpeakerVerificationReason MapTimelineFailure(SpeakerTimelineFailure failure)
        => failure switch
        {
            SpeakerTimelineFailure.StreamEpochMismatch => SpeakerVerificationReason.StreamEpochMismatch,
            SpeakerTimelineFailure.TimelineGap => SpeakerVerificationReason.TimelineGap,
            SpeakerTimelineFailure.AudioExpired => SpeakerVerificationReason.AudioExpired,
            SpeakerTimelineFailure.NotYetAvailable => SpeakerVerificationReason.WaitingForEvidence,
            SpeakerTimelineFailure.NoStream => SpeakerVerificationReason.NotReady,
            _ => SpeakerVerificationReason.CandidateRangeInvalid,
        };

    private readonly record struct QueuedJob(SpeakerInferenceJob Job, long Sequence, long EnqueuedAt);

    private sealed class QueuedJobComparer : IComparer<QueuedJob>
    {
        public static QueuedJobComparer Instance { get; } = new();

        /// <summary>Higher priority sorts last so <see cref="SortedSet{T}.Max"/> is the next job; ties keep FIFO.</summary>
        public int Compare(QueuedJob x, QueuedJob y)
        {
            int priority = ((int)x.Job.Priority).CompareTo((int)y.Job.Priority);
            if (priority != 0)
                return priority;
            return y.Sequence.CompareTo(x.Sequence);
        }
    }
}
