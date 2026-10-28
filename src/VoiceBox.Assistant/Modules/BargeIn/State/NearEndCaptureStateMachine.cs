using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Owns the confirmed capture's direct ASR feed: the turn identity, the monotonic feed watermark,
/// in-flight feed accounting and the bounded end drain. Drain finalization is claimed at most once
/// and only after every reserved feed completed. All mutation happens under the near-end
/// data-plane lock.
/// </summary>
internal sealed class NearEndCaptureStateMachine
{
    public enum Status
    {
        Idle,
        DirectFeed,
        Draining,
    }

    public readonly record struct FeedDecision(
        long TurnId,
        bool Handled,
        NearEndFeedReservation Reservation,
        bool Finalized,
        TimeSpan SpeechDuration,
        ActiveEndDrainOutcome Outcome)
    {
        public static FeedDecision NotHandled => default;
    }

    public readonly record struct Finalization(
        bool Claimed,
        long TurnId,
        TimeSpan SpeechDuration,
        ActiveEndDrainOutcome Outcome)
    {
        public static Finalization None => default;
    }

    private long _activeTurnId;
    private long _nextSample;
    private long _directFeedSamples;
    private int _feedsInFlight;
    private bool _drainForced;
    private bool _finalizationClaimed;
    private string _drainReason = "none";
    private long _drainStreamEpoch;
    private long _drainTargetSample;
    private long _drainDeadlineSample;
    private TimeSpan _drainSpeechDuration;
    private double _speechDurationMs;
    private double _silenceMs;
    private string _preRollSource = "none";
    private int _preRollSamples;

    public Status State { get; private set; }

    public long ActiveTurnId => _activeTurnId;

    public bool DirectFeed => State != Status.Idle;

    public bool EndDrainPending => State == Status.Draining;

    public long DirectFeedNextSample => _nextSample;

    public long DirectFeedSamples => _directFeedSamples;

    public int DirectFeedsInFlight => _feedsInFlight;

    public double SpeechDurationMs => _speechDurationMs;

    public string PreRollSource => _preRollSource;

    public int PreRollSamples => _preRollSamples;

    public bool EndDrainForced => _drainForced;

    public bool EndFinalizationClaimed => _finalizationClaimed;

    public string EndDrainReason => _drainReason;

    public long EndDrainStreamEpoch => _drainStreamEpoch;

    public long EndDrainTargetSample => _drainTargetSample;

    public long EndDrainDeadlineSample => _drainDeadlineSample;

    /// <summary>
    /// Lock-free read used by the duplex callback to skip frames before taking the data-plane
    /// lock. It is the only unsynchronized access on this machine.
    /// </summary>
    public long ReadActiveTurnIdVolatile() => Volatile.Read(ref _activeTurnId);

    /// <summary>Starts the capture for a committed turn. In-flight feeds from a prior turn stay counted.</summary>
    public void Start(
        long turnId,
        long firstFeedSample,
        double speechDurationMs,
        string preRollSource,
        int preRollSamples)
    {
        _activeTurnId = turnId;
        ResetDrainFields();
        State = Status.DirectFeed;
        _nextSample = firstFeedSample;
        _directFeedSamples = 0;
        _speechDurationMs = speechDurationMs;
        _silenceMs = 0;
        _preRollSource = preRollSource;
        _preRollSamples = preRollSamples;
    }

    public void Stop()
    {
        _activeTurnId = 0;
        State = Status.Idle;
        _nextSample = 0;
        _directFeedSamples = 0;
        _speechDurationMs = 0;
        _silenceMs = 0;
        _preRollSource = "none";
        _preRollSamples = 0;
        ResetDrainFields();
    }

    /// <summary>Clears only the feed watermark, as a timeline reset does.</summary>
    public void ResetFeedWatermark() => _nextSample = 0;

    /// <summary>Pins the feed watermark while capture is idle, after a timeline discontinuity.</summary>
    public void SetFeedWatermark(long sample) => _nextSample = sample;

    /// <summary>Moves the watermark forward to owned main audio that was already observed.</summary>
    public void SetFeedWatermarkAtLeast(long sample) => _nextSample = Math.Max(_nextSample, sample);

    public bool IsActive(long turnId) => _activeTurnId == turnId && State != Status.Idle;

    public void AddSamples(long turnId, int samples)
    {
        if (samples > 0 && _activeTurnId == turnId && State != Status.Idle)
            _directFeedSamples += samples;
    }

    /// <summary>
    /// Reserves the still-unfed part of one observation. Returns <see cref="FeedDecision.Handled"/>
    /// false when the observation belongs to another turn or timeline epoch.
    /// </summary>
    public FeedDecision ReserveFeed(
        in MainAudioObservation observation,
        long currentGeneration,
        long currentStreamEpoch)
    {
        if (_activeTurnId == 0 ||
            State == Status.Idle ||
            observation.Generation != currentGeneration ||
            observation.StreamEpoch != currentStreamEpoch)
        {
            return FeedDecision.NotHandled;
        }

        long turnId = _activeTurnId;
        if (observation.HasDiscontinuity)
        {
            // 处理链报告时间轴断裂时，目标点之前的缺口已无法恢复。当前帧位于缺口之后，
            // 不能假装它与已喂音频连续；等待中的结束排空必须立即以 partial 收尾，非结束态
            // 则从下一帧重新建立直送起点。
            if (State == Status.Draining)
                ForceEndDrain("capture-discontinuity");
            else
                _nextSample = Math.Max(_nextSample, observation.EndSample);

            return Claim(handled: true, default);
        }

        if (observation.EndSample <= _nextSample)
            return Claim(handled: true, default);

        long feedStart = Math.Max(observation.StartSample, _nextSample);
        long feedEnd = observation.EndSample;
        if (State == Status.Draining)
        {
            if (observation.StreamEpoch != _drainStreamEpoch)
            {
                ForceEndDrain("stream-epoch-changed");
                feedEnd = feedStart;
            }
            else
            {
                feedEnd = Math.Min(feedEnd, _drainTargetSample);
            }
        }

        NearEndFeedReservation reservation = default;
        if (feedEnd > feedStart)
        {
            reservation = new NearEndFeedReservation(turnId, feedStart, feedEnd);
            _nextSample = feedEnd;
            _feedsInFlight++;
        }

        if (State == Status.Draining &&
            _nextSample >= _drainTargetSample &&
            !_drainForced)
        {
            _drainReason = "aligned";
        }

        return reservation.IsValid
            ? new FeedDecision(turnId, true, reservation, false, TimeSpan.Zero, default)
            : Claim(handled: true, reservation);
    }

    /// <summary>Releases a reserved feed and rolls the watermark back when the provider rejected it.</summary>
    public void CompleteFeed(long turnId, in NearEndFeedReservation reservation, bool failed)
    {
        _feedsInFlight--;
        if (failed && _activeTurnId == turnId && _nextSample == reservation.EndSampleExclusive)
        {
            _nextSample = reservation.StartSample;
            if (State == Status.Draining)
                ForceEndDrain("asr-feed-failed");
        }
    }

    /// <summary>Advances duration/silence facts for one direct-feed frame and starts the end drain.</summary>
    public Finalization ObserveAudioFrame(
        long turnId,
        long streamEpoch,
        long endSampleExclusive,
        bool hasDiscontinuity,
        double frameMs,
        bool latestSpeechActive,
        int endSilenceMs,
        long timeoutSamples)
    {
        if (_activeTurnId != turnId || State == Status.Idle)
            return Finalization.None;

        if (State == Status.Draining)
        {
            if (hasDiscontinuity || streamEpoch != _drainStreamEpoch)
                ForceEndDrain("capture-discontinuity");
            else if (endSampleExclusive >= _drainDeadlineSample)
                ForceEndDrain("drain-timeout");

            return ClaimFinalization(turnId);
        }

        _speechDurationMs += frameMs;
        // 结束检测必须看最新帧，而不是整个滑窗里是否还残留任意已接受帧。否则用户停说后，
        // 窗口内一个旧语音帧或偶发噪声尖峰就会持续把静音计时归零。
        if (latestSpeechActive)
            _silenceMs = 0;
        else
            _silenceMs += frameMs;

        if (_silenceMs >= endSilenceMs)
        {
            BeginDrain(streamEpoch, endSampleExclusive, hasDiscontinuity, timeoutSamples);
            return ClaimFinalization(turnId);
        }

        return Finalization.None;
    }

    public void BeginDrain(
        long streamEpoch,
        long endSampleExclusive,
        bool hasDiscontinuity,
        long timeoutSamples)
    {
        if (_activeTurnId == 0 || State != Status.DirectFeed)
            return;

        State = Status.Draining;
        _drainForced = false;
        _finalizationClaimed = false;
        _drainReason = "none";
        _drainSpeechDuration = TimeSpan.FromMilliseconds(_speechDurationMs);
        _drainStreamEpoch = streamEpoch;
        _drainTargetSample = endSampleExclusive;
        _drainDeadlineSample = SaturatingAdd(endSampleExclusive, timeoutSamples);

        if (hasDiscontinuity || streamEpoch <= 0 || endSampleExclusive < 0)
            ForceEndDrain("invalid-capture-timeline");
        else if (_nextSample >= _drainTargetSample)
            _drainReason = "aligned";
    }

    public void ForceEndDrain(string reason)
    {
        if (State != Status.Draining || _finalizationClaimed)
            return;

        _drainForced = true;
        _drainReason = reason;
    }

    /// <summary>
    /// Claims the single terminal finalization for this drain. It requires that no direct feed is
    /// still in flight and that the fed watermark reached the target unless the drain was forced.
    /// </summary>
    public bool TryClaimEndFinalization(
        out TimeSpan speechDuration,
        out ActiveEndDrainOutcome outcome)
    {
        speechDuration = TimeSpan.Zero;
        outcome = default;
        if (State != Status.Draining ||
            _finalizationClaimed ||
            _feedsInFlight != 0 ||
            (!_drainForced && _nextSample < _drainTargetSample))
        {
            return false;
        }

        _finalizationClaimed = true;
        speechDuration = _drainSpeechDuration;
        long fedThrough = Math.Min(_nextSample, _drainTargetSample);
        outcome = new ActiveEndDrainOutcome(
            _drainReason,
            _drainStreamEpoch,
            _drainTargetSample,
            fedThrough,
            Math.Max(0, _drainTargetSample - fedThrough));
        return true;
    }

    public void ClearEndDrain()
    {
        ResetDrainFields();
        State = _activeTurnId != 0 ? Status.DirectFeed : Status.Idle;
    }

    // Stryker disable equality: right==0 never overflows; left==(Max-right) yields MaxValue for both > and >=.
    public static long SaturatingAdd(long left, long right)
        => right > 0 && left > long.MaxValue - right ? long.MaxValue : left + right;
    // Stryker restore equality

    private Finalization ClaimFinalization(long turnId)
        => TryClaimEndFinalization(out TimeSpan speechDuration, out ActiveEndDrainOutcome outcome)
            ? new Finalization(true, turnId, speechDuration, outcome)
            : Finalization.None;

    private FeedDecision Claim(bool handled, NearEndFeedReservation reservation)
        => TryClaimEndFinalization(out TimeSpan speechDuration, out ActiveEndDrainOutcome outcome)
            ? new FeedDecision(_activeTurnId, handled, reservation, true, speechDuration, outcome)
            : new FeedDecision(_activeTurnId, handled, reservation, false, TimeSpan.Zero, default);

    private void ResetDrainFields()
    {
        _drainForced = false;
        _finalizationClaimed = false;
        _drainReason = "none";
        _drainStreamEpoch = 0;
        _drainTargetSample = 0;
        _drainDeadlineSample = 0;
        _drainSpeechDuration = TimeSpan.Zero;
    }
}
