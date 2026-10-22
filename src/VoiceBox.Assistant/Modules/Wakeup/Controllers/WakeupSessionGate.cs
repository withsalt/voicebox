using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Wakeup.Adapters;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Extensions;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Controllers;

/// <summary>
/// Wake-word session gate. It coordinates four owned parts — <see cref="WakeupModeState"/>,
/// <see cref="WakeupPreRollTimeline"/>, <see cref="WakeupSpeakerAuthorization"/> and
/// <see cref="WakeupDetectorAdapter"/> — behind one state lock, so a mode change plus its pre-roll
/// re-anchoring is a single atomic step. Speaker authorization is evaluated
/// against the same capture interval reported by KWS; it never appends a second, independently
/// scheduled copy of candidate audio.
/// </summary>
internal sealed class WakeupSessionGate : IDisposable, IWakeupDetectorActivationView
{
    private readonly WakeupModeState _modeState;
    private readonly WakeupPreRollTimeline _timeline;
    private readonly WakeupSpeakerAuthorization? _authorization;
    private readonly WakeupDetectorAdapter _detector;
    private readonly TimeSpan _idleTimeout;
    private readonly int _sampleRate;
    private readonly int _channels;
    private readonly ILogger _logger;
    private readonly object _lock = new();

    private WakeupEventArgs? _pendingWakeup;
    // Capture position at which the pending candidate was last evaluated and found Waiting. The
    // gate consults the authorization up to three times per frame (before append, after append,
    // after the detector feed); the same position is never evaluated twice.
    private long _waitingEvaluatedCaptureEnd = -1;
    private bool _started;
    private bool _disposed;
    private WakeupAuthorizationOutcome _lastDeniedOutcome;
    private long _deniedWakeups;
    private long _authorizedWakeups;

    public WakeupSessionGate(
        IWakeupService wakeup,
        int preRollSamples,
        TimeSpan idleTimeout,
        int sampleRate,
        int channels,
        int replayLeadInMs,
        ISpeakerEvidenceProvider? speakerEvidence,
        TimeProvider? timeProvider = null,
        int wakeupStrongScoreMargin = 15,
        int postTriggerEvidenceSamples = 0,
        ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(wakeup);
        if (preRollSamples < 0)
            throw new ArgumentOutOfRangeException(nameof(preRollSamples));
        if (idleTimeout < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(idleTimeout));
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels));
        if (replayLeadInMs < 0)
            throw new ArgumentOutOfRangeException(nameof(replayLeadInMs));
        if (wakeupStrongScoreMargin is < 0 or > 100)
            throw new ArgumentOutOfRangeException(nameof(wakeupStrongScoreMargin));

        TimeProvider clock = timeProvider ?? TimeProvider.System;
        _idleTimeout = idleTimeout;
        _sampleRate = sampleRate;
        _channels = channels;
        _logger = logger ?? NullLogger<WakeupSessionGate>.Instance;
        _modeState = new WakeupModeState(clock);
        _timeline = new WakeupPreRollTimeline(preRollSamples, sampleRate, channels, replayLeadInMs);
        _authorization = speakerEvidence is null
            ? null
            : new WakeupSpeakerAuthorization(
                speakerEvidence,
                clock,
                wakeupStrongScoreMargin,
                postTriggerEvidenceSamples);
        _detector = new WakeupDetectorAdapter(wakeup, this);
        _detector.Subscribe(OnWakeup);
    }

    public AssistantWakeMode Mode
    {
        get { lock (_lock) return _modeState.Mode; }
    }

    /// <summary>Outcome of the most recently denied keyword candidate, for diagnostics.</summary>
    public WakeupAuthorizationOutcome LastDeniedOutcome
    {
        get { lock (_lock) return _lastDeniedOutcome; }
    }

    public long DeniedWakeups => Volatile.Read(ref _deniedWakeups);

    public long AuthorizedWakeups => Volatile.Read(ref _authorizedWakeups);

    public void Start()
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _started = true;
            _modeState.ResetToWaitingForWakeWord();
            _pendingWakeup = null;
            _authorization?.Reset();
            _timeline.ClearForStart();
        }
        _detector.StartUnconditional();
    }

    public void Stop()
    {
        lock (_lock)
        {
            _started = false;
            _pendingWakeup = null;
            _authorization?.Reset();
            _timeline.ClearForStop();
        }
        _detector.StopUnconditional();
    }

    public WakeupAudioDecision OnAudio(in ProcessedCaptureFrame frame)
    {
        ReadOnlySpan<float> audio = frame.Audio.Span;
        if (audio.IsEmpty)
        {
            AssistantWakeMode mode = Mode;
            return new WakeupAudioDecision(
                WakeupAudioRoute.WakeDetectorOnly,
                null,
                null,
                WakeupTriggerKind.None,
                mode,
                mode,
                false);
        }
        if (frame.SampleRate != _sampleRate || frame.Channels != _channels ||
            frame.SourceRange.Length != audio.Length)
        {
            throw new InvalidOperationException("Wakeup audio does not match its capture timeline or configured format.");
        }

        bool restartForDiscontinuity = false;
        bool feedWakeDetector = false;
        WakeupAudioRoute routeBeforeFeed = WakeupAudioRoute.WakeDetectorOnly;
        WakeupAudioDecision? completedBeforeFeed = null;
        CaptureRange sourceRange = frame.SourceRange;
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_started)
                return CreateStoppedDecisionLocked();

            if (_timeline.OriginPending)
            {
                _timeline.SetOrigin(in sourceRange);
            }
            else if (frame.HasDiscontinuity ||
                frame.SourceRange.StreamEpoch != _timeline.WakeDetectorStreamEpoch)
            {
                _pendingWakeup = null;
                _authorization?.Reset();
                _timeline.ClearForDiscontinuity(in sourceRange);
                restartForDiscontinuity = true;
            }

            _timeline.AdvanceSamplesSeen(audio.Length);

            // A wake event raised by the previous KWS feed is evaluated before the
            // current block enters the bounded pre-roll. This preserves the entire
            // already-buffered wake word plus the current trailing block for replay.
            WakeupAudioDecision? beforeAppend = TryConsumeWakeupLocked(
                frame.SourceRange.EndSampleExclusive,
                frame.SourceRange.TimestampTicks,
                audio);
            if (beforeAppend.HasValue)
            {
                completedBeforeFeed = TouchAcceptedWakeupLocked(beforeAppend.Value);
            }
            else
            {
                if (_modeState.IsListeningForWakeWordOrInSession)
                    _timeline.Append(audio, frame.SourceRange.EndSampleExclusive);

                WakeupAudioDecision? pending = TryConsumeWakeupLocked(
                    frame.SourceRange.EndSampleExclusive,
                    frame.SourceRange.TimestampTicks);
                if (pending.HasValue)
                {
                    completedBeforeFeed = TouchAcceptedWakeupLocked(pending.Value);
                }
                else
                {
                    feedWakeDetector = _pendingWakeup is null &&
                        _modeState.IsListeningForWakeWordOrInSession;
                    routeBeforeFeed = _modeState.Mode == AssistantWakeMode.ContinuousConversation
                        ? WakeupAudioRoute.Conversation
                        : WakeupAudioRoute.WakeDetectorOnly;
                }
            }
        }

        if (completedBeforeFeed.HasValue)
        {
            _detector.StartIfActive();
            return completedBeforeFeed.Value;
        }
        if (restartForDiscontinuity)
            _detector.RestartIfActive();
        if (feedWakeDetector)
            _detector.FeedIfActive(audio);

        WakeupAudioDecision? completedAfterFeed;
        WakeupAudioDecision noWakeupDecision;
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_started)
                return CreateStoppedDecisionLocked();
            completedAfterFeed = TryConsumeWakeupLocked(
                frame.SourceRange.EndSampleExclusive,
                frame.SourceRange.TimestampTicks);
            if (completedAfterFeed.HasValue)
                completedAfterFeed = TouchAcceptedWakeupLocked(completedAfterFeed.Value);

            WakeupAudioRoute route = _modeState.Mode == AssistantWakeMode.ContinuousConversation
                ? WakeupAudioRoute.Conversation
                : routeBeforeFeed;
            noWakeupDecision = new WakeupAudioDecision(
                route,
                null,
                null,
                WakeupTriggerKind.None,
                _modeState.Mode,
                _modeState.Mode,
                false);
        }

        if (!completedAfterFeed.HasValue)
            return noWakeupDecision;

        _detector.StartIfActive();
        return completedAfterFeed.Value;
    }

    public WakeupTimeoutDecision CheckIdleTimeout()
    {
        if (_idleTimeout == TimeSpan.Zero)
            return default;

        WakeupTimeoutDecision decision;
        lock (_lock)
        {
            if (!_modeState.IsIdleTimeoutElapsed(_idleTimeout))
                return default;

            if (!_modeState.TryLeaveConversation(out AssistantWakeMode old))
                return default;
            ResetWakeDetectorStateLocked();
            decision = new WakeupTimeoutDecision(old, _modeState.Mode, true);
        }
        _detector.RestartIfActive();
        return decision;
    }

    public WakeupTimeoutDecision ReturnToWakeWord()
    {
        WakeupTimeoutDecision decision;
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_modeState.TryLeaveConversation(out AssistantWakeMode old))
                return default;

            ResetWakeDetectorStateLocked();
            decision = new WakeupTimeoutDecision(old, _modeState.Mode, true);
        }
        _detector.RestartIfActive();
        return decision;
    }

    public void MarkUserActivity()
    {
        lock (_lock)
            _modeState.TouchActivityIfInConversation();
    }

    private WakeupAudioDecision? TryConsumeWakeupLocked(
        long captureEndSample,
        long timestampTicks,
        ReadOnlySpan<float> trailingAudio = default)
    {
        if (_pendingWakeup is null)
            return null;
        if (_waitingEvaluatedCaptureEnd == captureEndSample)
            return null;

        WakeupEventArgs wakeup = _pendingWakeup;
        WakeupAuthorizationOutcome authorization = _authorization?.Authorize(
            wakeup,
            _timeline,
            captureEndSample,
            timestampTicks) ?? WakeupAuthorizationOutcome.Authorized;
        if (authorization == WakeupAuthorizationOutcome.Waiting)
        {
            _waitingEvaluatedCaptureEnd = captureEndSample;
            return null;
        }

        _waitingEvaluatedCaptureEnd = -1;
        if (authorization.IsDenied())
        {
            _lastDeniedOutcome = authorization;
            Interlocked.Increment(ref _deniedWakeups);
            SpeakerVerificationResult evidence = _authorization?.LastResult ?? default;
            _logger.LogDebug(
                "唤醒词命中但未通过声纹门控: keyword={Keyword}, score={Score}, threshold={Threshold}, authorization={Authorization}, speakerStatus={SpeakerStatus}, speakerReason={SpeakerReason}, observed={Observed}, matched={Matched}/{Required}, invalid={Invalid}, mode={Mode}.",
                wakeup.Keyword,
                wakeup.Score,
                wakeup.Threshold,
                authorization,
                evidence.Status,
                evidence.Reason,
                evidence.ObservedWindows,
                evidence.MatchedWindows,
                evidence.RequiredMatches,
                evidence.InvalidWindows,
                _modeState.Mode);
            return RejectPendingWakeupLocked(captureEndSample);
        }

        Interlocked.Increment(ref _authorizedWakeups);
        _logger.LogDebug(
            "唤醒词命中并通过唤醒门: keyword={Keyword}, score={Score}, threshold={Threshold}, speakerAuthorization={SpeakerAuthorization}, mode={Mode}.",
            wakeup.Keyword,
            wakeup.Score,
            wakeup.Threshold,
            _authorization is null ? "disabled" : "authorized",
            _modeState.Mode);
        _pendingWakeup = null;
        _authorization?.Reset();
        AssistantWakeMode oldMode = _modeState.EnterConversation();
        WakeupTriggerKind triggerKind = oldMode == AssistantWakeMode.WaitingForWakeWord
            ? WakeupTriggerKind.InitialWakeup
            : WakeupTriggerKind.InSessionWakeup;
        float[] replayAudio = _timeline.TakeReplayAudio(trailingAudio, wakeup, triggerKind);
        _timeline.AnchorAt(captureEndSample);

        return new WakeupAudioDecision(
            WakeupAudioRoute.Conversation,
            replayAudio,
            wakeup,
            triggerKind,
            oldMode,
            _modeState.Mode,
            oldMode != _modeState.Mode);
    }

    private WakeupAudioDecision RejectPendingWakeupLocked(long captureEndSample)
    {
        AssistantWakeMode mode = _modeState.Mode;
        _pendingWakeup = null;
        _authorization?.Reset();
        _timeline.Clear();
        _timeline.AnchorAt(captureEndSample);
        return new WakeupAudioDecision(
            mode == AssistantWakeMode.ContinuousConversation
                ? WakeupAudioRoute.Conversation
                : WakeupAudioRoute.WakeDetectorOnly,
            null,
            null,
            WakeupTriggerKind.None,
            mode,
            mode,
            false);
    }

    private void ResetWakeDetectorStateLocked()
    {
        _pendingWakeup = null;
        _authorization?.Reset();
        _timeline.ClearForRestart();
    }

    private WakeupAudioDecision TouchAcceptedWakeupLocked(WakeupAudioDecision decision)
    {
        if (decision.Wakeup != null)
            _modeState.TouchActivity();
        return decision;
    }

    private WakeupAudioDecision CreateStoppedDecisionLocked()
        => new(
            WakeupAudioRoute.WakeDetectorOnly,
            null,
            null,
            WakeupTriggerKind.None,
            _modeState.Mode,
            _modeState.Mode,
            false);

    private bool IsDetectorActive()
    {
        lock (_lock)
            return _started && !_disposed;
    }

    private void OnWakeup(object? sender, WakeupEventArgs e)
    {
        bool stopWakeup;
        AssistantWakeMode mode;
        lock (_lock)
        {
            if (_disposed ||
                !_started ||
                !_modeState.IsListeningForWakeWordOrInSession ||
                _pendingWakeup != null)
            {
                return;
            }
            _pendingWakeup = e;
            _waitingEvaluatedCaptureEnd = -1;
            _authorization?.Reset();
            stopWakeup = true;
            mode = _modeState.Mode;
        }
        _logger.LogDebug(
            "唤醒词命中: keyword={Keyword}, score={Score}, threshold={Threshold}, durationMs={DurationMs}, startTimeMs={StartTimeMs}, mode={Mode}, speakerVerification={SpeakerVerification}.",
            e.Keyword,
            e.Score,
            e.Threshold,
            e.DurationMs,
            e.StartTimeMs,
            mode,
            _authorization is null ? "disabled" : "pending");
        if (stopWakeup)
            _detector.StopIfActive();
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
                return;
            _disposed = true;
            _started = false;
            _pendingWakeup = null;
            _authorization?.Reset();
        }
        _detector.Unsubscribe(OnWakeup);
        _detector.Dispose();
    }

    bool IWakeupDetectorActivationView.IsActive => IsDetectorActive();
}
