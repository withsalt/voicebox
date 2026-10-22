using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Buffering;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;

namespace VoiceBox.Assistant.Modules.Speaker.Evidence;

/// <summary>
/// Speech-gated prefetch: silence schedules nothing; as soon as identity-tap energy opens the gate,
/// every newly completed hop-aligned window is queued for inference so a later KWS hit or
/// playback-time candidate mostly finds its first window already scored. It never waits for a DTD
/// interrupt, a keyword or a formal query, and it never runs the model itself.
/// </summary>
internal sealed class SpeakerSpeechPrefetchGate
{
    private readonly SpeakerAudioTimelineBuffer _timeline;
    private readonly SpeakerEvidenceWindowSelector _windows;
    private readonly SpeakerWindowQualityEvaluator _quality;
    private readonly SpeakerInferenceScheduler _scheduler;
    private readonly int _maxWindows;
    private readonly long _hangoverSamples;
    private readonly object _lock = new();
    private long _speechActiveUntil = long.MinValue;
    private long _lastScheduledEnd = long.MinValue;
    private long _lastEpoch;
    private int _lastGapGeneration = int.MinValue;
    private long _scheduledCount;
    private long _rejectedCount;
    private bool _enabled = true;
    private int _prefetchPriority;

    public SpeakerSpeechPrefetchGate(
        SpeakerAudioTimelineBuffer timeline,
        SpeakerEvidenceWindowSelector windows,
        SpeakerWindowQualityEvaluator quality,
        SpeakerInferenceScheduler scheduler,
        int maxWindows,
        int hangoverSamples)
    {
        _timeline = timeline ?? throw new ArgumentNullException(nameof(timeline));
        _windows = windows ?? throw new ArgumentNullException(nameof(windows));
        _quality = quality ?? throw new ArgumentNullException(nameof(quality));
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        if (maxWindows <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxWindows));
        _maxWindows = maxWindows;
        _hangoverSamples = Math.Max(0, hangoverSamples);
    }

    public long ScheduledCount => Volatile.Read(ref _scheduledCount);

    public long RejectedCount => Volatile.Read(ref _rejectedCount);

    /// <summary>Whether the speech gate is currently open at the latest observed sample.</summary>
    public bool IsSpeechActive
    {
        get
        {
            lock (_lock)
                return _timeline.TryGetBounds(out _, out _, out _, out long end) && end <= _speechActiveUntil;
        }
    }

    /// <summary>Composition disables the gate entirely when no realtime purpose is effective.</summary>
    public void SetEnabled(bool enabled)
    {
        lock (_lock)
            _enabled = enabled;
    }

    /// <summary>Priority used for prefetch jobs; the playback owner raises it while speaking.</summary>
    public SpeakerInferencePriority PrefetchPriority
    {
        get => (SpeakerInferencePriority)Volatile.Read(ref _prefetchPriority);
        set => Volatile.Write(ref _prefetchPriority, (int)value);
    }

    /// <summary>Called after the frame has been appended to the ring. Allocation-free.</summary>
    public void Observe(in SpeakerCaptureFrame frame)
    {
        if (!frame.IsValid)
            return;

        bool speech = _quality.IsSpeechFrame(frame.Audio.Span);
        long frameEnd = frame.SourceRange.EndSampleExclusive;
        Span<long> ends = stackalloc long[8];
        int count;
        long epoch;
        SpeakerInferencePriority priority;
        lock (_lock)
        {
            if (!_enabled)
                return;

            int gapGeneration = _timeline.GapGeneration;
            if (frame.SourceRange.StreamEpoch != _lastEpoch || gapGeneration != _lastGapGeneration)
            {
                _lastEpoch = frame.SourceRange.StreamEpoch;
                _lastGapGeneration = gapGeneration;
                _lastScheduledEnd = long.MinValue;
                _speechActiveUntil = long.MinValue;
            }

            if (speech)
                _speechActiveUntil = checked(frameEnd + _hangoverSamples);
            if (frameEnd > _speechActiveUntil)
                return;

            if (!_timeline.TryGetBounds(out epoch, out long segmentStart, out _, out long retainedEnd))
                return;

            count = _windows.EnumerateCompleteWindowEnds(
                segmentStart,
                _lastScheduledEnd,
                retainedEnd,
                ends.Slice(0, Math.Min(ends.Length, _maxWindows)));
            if (count == 0)
                return;
            _lastScheduledEnd = ends[count - 1];
            priority = PrefetchPriority;
        }

        for (int i = 0; i < count; i++)
        {
            long end = ends[i];
            var job = new SpeakerInferenceJob(
                epoch,
                end - _windows.WindowSamples,
                end,
                priority,
                default,
                IsPrefetch: true);
            switch (_scheduler.TrySchedule(in job))
            {
                case SpeakerScheduleResult.Queued:
                    Interlocked.Increment(ref _scheduledCount);
                    break;
                case SpeakerScheduleResult.PrefetchRejected:
                    Interlocked.Increment(ref _rejectedCount);
                    break;
            }
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _speechActiveUntil = long.MinValue;
            _lastScheduledEnd = long.MinValue;
            _lastGapGeneration = int.MinValue;
            PrefetchPriority = SpeakerInferencePriority.WakeupPrefetch;
        }
    }
}
