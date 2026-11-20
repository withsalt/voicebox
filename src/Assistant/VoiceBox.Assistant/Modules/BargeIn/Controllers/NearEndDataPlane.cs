using Microsoft.Extensions.Logging;
using System.Runtime.ExceptionServices;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Aggregate root of the real-time near-end module. It owns the single data-plane lock, drives the
/// timeline, endpoint, candidate, capture and confirmation state owners, and hands every external
/// side effect to <see cref="NearEndEffectExecutor"/> after releasing the lock. It owns no speaker
/// algorithm, no turn-kernel state and no ASR session.
/// </summary>
internal sealed class NearEndDataPlane
{
    private readonly BargeInRuntimeOptions _options;
    private readonly IActiveNearEndConversationView _conversation;
    private readonly IActiveNearEndDecisionSource _decisions;
    private readonly AssistantAudioFormat _audioFormat;
    private readonly ILogger _logger;
    private readonly IRuntimeLivenessView _runtime;
    private readonly INearEndAuthorizationPolicy _authorization;
    private readonly NearEndEffectExecutor _effects;
    private readonly TimeProvider _timeProvider;
    private readonly NearEndTimelineBuffer _timeline;
    private readonly NearEndEndpointTracker _endpoint;
    private readonly NearEndCandidateStateMachine _candidate;
    private readonly NearEndCaptureStateMachine _capture;
    private readonly NearEndConfirmationJournal _journal;
    private readonly NearEndReplayBuffer _replay;
    private readonly PlaybackDuckingPolicy _ducking;
    private readonly object _speakerEvaluationGate = new();
    private readonly object _lock = new();
    private readonly object _feedGate = new();
    private long _cooldownUntilTicks;
    private string _lastCandidateOutcome = "none";
    private long _commandAuthorizedCount;
    private long _commandDeniedCount;
    private int _dataPlaneFaultReported;
    private PendingCapture? _pendingCapture;
    private ITimer? _pendingCaptureTimer;
    private long _captureSpeechEpoch;
    private long _earliestEndpointSample;
    private CancellationTokenSource? _speakerWaitCancellation;
    private long _speakerWaitGeneration;

    private sealed record PendingCapture(long TurnId, NearEndCapturePlan Plan, NearEndConfirmationLease Lease);

    private readonly record struct PendingCaptureActivation(
        long TurnId,
        long StreamEpoch,
        long StartSample,
        long EndSampleExclusive,
        bool Rebased);

    public NearEndDataPlane(
        BargeInRuntimeOptions options,
        IActiveNearEndConversationView conversation,
        IActiveNearEndDecisionSource decisions,
        AssistantAudioFormat audioFormat,
        ILogger logger,
        IRuntimeLivenessView runtime,
        INearEndAuthorizationPolicy authorization,
        NearEndEffectExecutor effects,
        TimeProvider timeProvider,
        NearEndTimelineBuffer timeline,
        NearEndEndpointTracker endpoint,
        NearEndCandidateStateMachine candidate,
        NearEndCaptureStateMachine capture,
        NearEndConfirmationJournal journal,
        NearEndReplayBuffer replay,
        PlaybackDuckingPolicy ducking)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _conversation = conversation ?? throw new ArgumentNullException(nameof(conversation));
        _decisions = decisions ?? throw new ArgumentNullException(nameof(decisions));
        _audioFormat = audioFormat ?? throw new ArgumentNullException(nameof(audioFormat));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _authorization = authorization ?? throw new ArgumentNullException(nameof(authorization));
        _effects = effects ?? throw new ArgumentNullException(nameof(effects));
        _timeProvider = timeProvider ?? TimeProvider.System;
        _timeline = timeline ?? throw new ArgumentNullException(nameof(timeline));
        _endpoint = endpoint ?? throw new ArgumentNullException(nameof(endpoint));
        _candidate = candidate ?? throw new ArgumentNullException(nameof(candidate));
        _capture = capture ?? throw new ArgumentNullException(nameof(capture));
        _journal = journal ?? throw new ArgumentNullException(nameof(journal));
        _replay = replay ?? throw new ArgumentNullException(nameof(replay));
        _ducking = ducking ?? throw new ArgumentNullException(nameof(ducking));
    }

    public bool IsIdentityRequired => _authorization.RequiresSpeaker;

    public long CommandAuthorizedCount => Volatile.Read(ref _commandAuthorizedCount);

    public long CommandDeniedCount => Volatile.Read(ref _commandDeniedCount);

    public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot()
    {
        lock (_lock)
        {
            return new ActiveNearEndDiagnosticSnapshot(
                _candidate.Active,
                _candidate.Authorized,
                _candidate.Decision.EndSampleExclusive,
                _timeline.EndSample,
                _capture.ActiveTurnId,
                _lastCandidateOutcome,
                _journal.HasQueued,
                _candidate.CandidateId.Value,
                _candidate.DtdInterruptObserved);
        }
    }

    public ActiveNearEndProcessingState GetProcessingState()
    {
        lock (_lock)
        {
            return new ActiveNearEndProcessingState(_capture.ActiveTurnId != 0, _candidate.Active || _endpoint.CandidateActive);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Endpoint VAD ingress
    // ---------------------------------------------------------------------------------------

    public void ObserveEndpointSpeechStarted()
    {
        lock (_lock)
        {
            // A fresh VAD onset definitively separates this utterance from any endpoint that was
            // retained while the prior DTD candidate was still forming.
            _endpoint.ClearUnowned();
            if (_capture.DirectFeed)
            {
                _endpoint.MarkCaptureSpeechActive();
                // Synchronous VAD facts can lag the DTD endpoint in sample coordinates. A
                // fresh onset must reopen capture even before recognition catches up to it.
                _capture.TryResumeFromDrain(_timeline.StreamEpoch);
                return;
            }

            _endpoint.ClearCandidate();
            _endpoint.BeginCandidate();
        }
    }

    public void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio)
    {
        if (audio.IsEmpty)
            return;

        lock (_lock)
        {
            if (_capture.DirectFeed)
            {
                _endpoint.MarkCaptureSpeechActive();
                _capture.TryResumeFromDrain(_timeline.StreamEpoch);
                return;
            }

            // IVoiceActivityDetectService promises that OnSpeaking contains audio from Predict.
            // Locate that exact chunk in the main-AEC timeline instead of estimating its start
            // from the later fused decision window.
            if (!_endpoint.CandidateActive)
                _endpoint.BeginCandidate();

            if (_endpoint.CandidateGeneration == _timeline.Generation &&
                _endpoint.TryLocate(audio, out long streamEpoch, out long startSample, out long endSample) &&
                (_endpoint.CandidateStreamEpoch == 0 || _endpoint.CandidateStreamEpoch == streamEpoch))
            {
                _endpoint.ExtendCandidate(streamEpoch, startSample, endSample);
                if (_journal.TryGetReplayRequest(out var lease) && lease.CapturePlan is { PreferEndpointVadStart: true } &&
                    TryGetEndpointVadCandidateStartLocked(lease.Decision, out _, out long onset, out _))
                {
                    _journal.RecordVadStart(_timeline.Generation, onset);
                }
            }
            if (_pendingCapture is null)
                return;
        }

        lock (_feedGate)
            TryActivatePendingCaptureOrdered();
    }

    public void ObserveUnownedEndpointSpeechEnded()
    {
        lock (_lock)
        {
            _endpoint.RecordUnowned(_timeline.Generation, _timeline.StreamEpoch, _timeline.EndSample);
            // SpeechEnded arrived while the speaker verdict is still pending: the candidate's
            // input is complete, and the bounded wait for the verdict continues on its own.
            if (_candidate.Active &&
                !_candidate.Authorized &&
                _candidate.StreamEpoch == _timeline.StreamEpoch &&
                _timeline.EndSample > _candidate.MainStartSample)
            {
                _candidate.MarkEndpointObserved(_timeline.EndSample);
            }
        }
    }

    public void ObserveEndpointSpeechEnded(bool captureCommittedAfterEndpoint = false)
    {
        long activeTurnId = 0;
        bool finalize = false;
        TimeSpan speechDuration = TimeSpan.Zero;
        ActiveEndDrainOutcome drainOutcome = default;
        lock (_lock)
        {
            if (_capture.DirectFeed && captureCommittedAfterEndpoint &&
                (!_endpoint.UnownedObserved || _endpoint.UnownedProcessed ||
                 _endpoint.UnownedGeneration != _timeline.Generation ||
                 _endpoint.UnownedStreamEpoch != _timeline.StreamEpoch))
            {
                // Commit already retired this queued endpoint, or a fresh onset superseded it.
                return;
            }

            long endpointSample = captureCommittedAfterEndpoint ? _endpoint.UnownedEndSample : _timeline.EndSample;
            if (_capture.ActiveTurnId != 0 && EndpointPredatesSpeech(_timeline.StreamEpoch, endpointSample))
            {
                _endpoint.ClearUnowned();
                _endpoint.ClearCandidate();
                _endpoint.ClearCaptureObservation();
                return;
            }

            _endpoint.ClearCandidate();
            if (_capture.DirectFeed)
            {
                _endpoint.MarkCaptureSpeechEnded();
                activeTurnId = _capture.ActiveTurnId;
                // VAD already waited for its own silence threshold. Drain the observed primary
                // audio now; in-flight endpoint observations and feeds still protect the tail.
                _capture.BeginDrain(_timeline.StreamEpoch, _timeline.EndSample, hasDiscontinuity: false, _options.DrainTimeoutSamples);
                finalize = _capture.TryClaimEndFinalization(out speechDuration, out drainOutcome);

                _endpoint.ClearUnowned();
            }
            else if (_capture.Pending)
            {
                // The committed capture is still waiting for its first main-timeline frame. Keep
                // the VAD endpoint as the utterance tail for activation instead of dropping it;
                // otherwise ObserveCaptureStarted(endpointVad: true) would keep the capture
                // speech-active forever and no drain could ever start.
                if (!_endpoint.UnownedObserved)
                    _endpoint.RecordUnowned(_timeline.Generation, _timeline.StreamEpoch, _timeline.EndSample);
                _endpoint.MarkUnownedProcessed();
            }
            else if (_endpoint.UnownedObserved)
            {
                // This callback is the control-plane acknowledgement of an endpoint recorded by
                // the capture callback. Only an acknowledgement that runs before capture commits
                // can prove that no later endpoint observation is still queued behind it.
                _endpoint.MarkUnownedProcessed();
            }
            // A completed endpoint can precede the confirmation envelope, or even the DTD
            // decision that produces it. Keep its timeline fact until either a fresh VAD onset,
            // a timeline reset, or an accepted capture consumes it.
        }

        if (finalize)
            EndCapture(activeTurnId, speechDuration, drainOutcome);
    }

    // ---------------------------------------------------------------------------------------
    // Duplex callback
    // ---------------------------------------------------------------------------------------

    public void ObserveDuplexAudio(in DuplexAudioFrame frame, bool playbackStartupGuardActive, bool latestNearEndSpeech = false)
    {
        if (playbackStartupGuardActive)
        {
            ExecutePlayback(RestoreDucking());
            return;
        }

        Process(in frame);
        CaptureConfirmationTail(in frame, latestNearEndSpeech);
    }

    public void BeginEndpointObservation()
    {
        lock (_lock)
            _capture.BeginEndpointObservation();
    }

    public void CompleteEndpointObservation()
    {
        NearEndCaptureStateMachine.Finalization finalization;
        lock (_lock)
            finalization = _capture.CompleteEndpointObservation();

        if (finalization.Claimed)
            EndCapture(finalization.TurnId, finalization.SpeechDuration, finalization.Outcome);
    }

    public MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame)
    {
        ReadOnlySpan<float> audio = frame.Audio.Span;
        if (audio.IsEmpty)
            return default;

        bool retryAuthorizedCandidate = false;
        bool tryActivatePendingCapture = false;
        bool discontinuity;
        MainAudioObservation observation;
        PlaybackDuckingPolicy.Command restore = PlaybackDuckingPolicy.Command.None;
        lock (_lock)
        {
            CaptureRange sourceRange = frame.SourceRange;
            discontinuity = _timeline.Add(audio, in sourceRange, frame.HasDiscontinuity);
            if (discontinuity)
            {
                restore = _ducking.Restore();
                _captureSpeechEpoch = 0;
                _candidate.Clear();
                CancelSpeakerWaitLocked();
                _endpoint.ClearCandidate();
                _capture.ForceEndDrain("capture-discontinuity");
                if (_capture.ActiveTurnId == 0)
                    _capture.SetFeedWatermark(frame.SourceRange.EndSampleExclusive);
                if (_pendingCapture is { } pendingCapture &&
                    (pendingCapture.Plan.StreamEpoch != _timeline.StreamEpoch || pendingCapture.Plan.StartSample > _timeline.EndSample))
                {
                    // Old epoch coordinates never splice into the new stream. Keep the capture
                    // ownership and restart it at the new timeline end when it activates.
                    _pendingCapture = pendingCapture with
                    {
                        Plan = pendingCapture.Plan with
                        {
                            StreamEpoch = _timeline.StreamEpoch,
                            StartSample = _timeline.EndSample,
                        },
                    };
                    if (_endpoint.UnownedObserved && _endpoint.UnownedStreamEpoch != _timeline.StreamEpoch)
                        _endpoint.ClearUnowned();
                }
            }

            observation = new MainAudioObservation(
                _timeline.Generation,
                frame.SourceRange.StreamEpoch,
                frame.SourceRange.StartSample,
                frame.SourceRange.EndSampleExclusive,
                discontinuity);
            if (!discontinuity && _capture.ActiveTurnId == 0 && _candidate.Active && _candidate.Authorized)
                retryAuthorizedCandidate = true;
            tryActivatePendingCapture = _pendingCapture is not null;
        }

        ExecutePlayback(in restore);
        if (tryActivatePendingCapture)
        {
            lock (_feedGate)
                TryActivatePendingCaptureOrdered();
        }
        if (retryAuthorizedCandidate)
            TryStartAuthorizedCandidate();

        return observation;
    }

    public void FeedFromRecognitionAudio(ReadOnlySpan<float> audio, MainAudioObservation observation)
    {
        // Every feed, whichever callback observes the frame first, drains the still-unfed main
        // timeline range first and holds the gate across the provider call. Two producers (the
        // recognition callback and the control-plane confirmation commit) can therefore never
        // enqueue an older range after a newer one, which would permanently skip the gap.
        lock (_feedGate)
        {
            if (audio.IsEmpty)
                return;

            // The common direct-feed case has watermark == observation start and timeline end ==
            // observation end: feed the caller's span without a second reservoir snapshot/copy.
            if (NeedsBacklogDrain(in observation))
                DrainUnfedMainAudioBeforeFeed();
            if (!_replay.DrainActive)
                DrainConfirmationTail();
            FeedFromRecognitionAudioCore(audio, observation);
        }
    }

    private void FeedFromRecognitionAudioCore(ReadOnlySpan<float> audio, MainAudioObservation observation)
    {
        if (audio.IsEmpty)
            return;

        NearEndCaptureStateMachine.FeedDecision feed;
        lock (_lock)
        {
            feed = _capture.ReserveFeed(in observation, _timeline.Generation, _timeline.StreamEpoch);
        }

        if (!feed.Handled)
            return;

        if (feed.Reservation.IsValid)
        {
            NearEndFeedReservation reservation = feed.Reservation;
            int offset = checked((int)(reservation.StartSample - observation.StartSample));
            int length = checked((int)(reservation.EndSampleExclusive - reservation.StartSample));
            Exception? feedError = null;
            bool accepted = false;
            bool attached = false;
            try
            {
                NearEndEffectResult result = _effects.FeedRecognition(reservation.TurnId, audio.Slice(offset, length));
                accepted = result.Succeeded;
                attached = result.HasAttachedSession;
            }
            catch (Exception ex)
            {
                feedError = ex;
            }

            bool finalize;
            TimeSpan speechDuration;
            ActiveEndDrainOutcome drainOutcome;
            lock (_lock)
            {
                if (accepted)
                    _capture.AddSamples(reservation.TurnId, length);
                _capture.CompleteFeed(reservation.TurnId, in reservation, failed: feedError is not null || !accepted);
                finalize = _capture.TryClaimEndFinalization(out speechDuration, out drainOutcome);
            }

            if (feedError is null && !accepted)
            {
                ReportDataPlaneFault(
                    attached
                        ? "asr-feed-queue-overflow"
                        : "asr-pending-queue-overflow");
            }

            if (finalize)
                EndCapture(reservation.TurnId, speechDuration, drainOutcome);

            if (feedError is not null)
                ExceptionDispatchInfo.Capture(feedError).Throw();

            return;
        }

        if (feed.Finalized)
            EndCapture(feed.TurnId, feed.SpeechDuration, feed.Outcome);
    }

    public bool TryLocate(ReadOnlySpan<float> audio, out long streamEpoch, out long startSample, out long endSample)
    {
        lock (_lock)
            return _timeline.TryLocate(audio, out streamEpoch, out startSample, out endSample);
    }

    // ---------------------------------------------------------------------------------------
    // Confirmation transaction
    // ---------------------------------------------------------------------------------------

    public bool TryAcquire(long operationId, out NearEndConfirmationLease lease)
    {
        lock (_lock)
            return _journal.TryAcquire(operationId, out lease);
    }

    public bool TryCommit(in NearEndConfirmationLease lease, in SpeakingDecision confirmed)
    {
        // The capture start and its catch-up drain are one ordered unit with every recognition
        // feed. Otherwise a feed that observed the frame while the capture was still idle could
        // reserve only its own frame after the start, permanently skipping the backlog before it.
        lock (_feedGate)
            return TryCommitFeedOrdered(in lease, in confirmed);
    }

    private bool TryCommitFeedOrdered(in NearEndConfirmationLease lease, in SpeakingDecision confirmed)
    {
        // The control consumer owns the conversation snapshot; read it outside the data-plane
        // lock and require the decision to interrupt the same reply that admitted the candidate.
        if (lease.OwnerTurnId <= 0 || lease.OwnerTurnId != _conversation.CurrentTurnId || lease.OwnerTurnId != confirmed.InterruptedTurnId)
        {
            return false;
        }

        NearEndCapturePlan? capturePlan = lease.CapturePlan;
        bool deferred = capturePlan is { Ready: false };
        bool rebasedAfterStreamEpochChange = false;
        long firstFeedSample = 0;
        long committedStreamEpoch = 0;
        lock (_lock)
        {
            if (_capture.ActiveTurnId != 0 || _pendingCapture is not null)
                return false;
            if (!_journal.TryCommit(in lease))
                return false;

            _capture.ClearEndDrain();
            Volatile.Write(ref _dataPlaneFaultReported, 0);
            double speechMs = capturePlan is not null
                ? Math.Max(0, lease.Decision.AcceptedDurationMs)
                : Math.Max(0, _audioFormat.SamplesToMilliseconds(confirmed.ConfirmedCandidateSamples));
            committedStreamEpoch = _timeline.StreamEpoch;
            _captureSpeechEpoch = lease.Decision.StreamEpoch;
            _earliestEndpointSample = lease.Decision.LatestFrameAccepted
                ? lease.Decision.EndSampleExclusive
                : lease.Decision.FirstAcceptedStartSample + 1;
            if (deferred)
            {
                // Register capture ownership now; the stop that follows this commit no longer
                // depends on the pre-roll snapshot or on the main recognition chain advancing.
                NearEndCapturePlan pendingPlan = capturePlan!.Value;
                if (pendingPlan.PreferEndpointVadStart &&
                    _journal.LocatedVadStart is { } located &&
                    located.Generation == _timeline.Generation)
                {
                    pendingPlan = pendingPlan with
                    {
                        StartSample = Math.Min(pendingPlan.StartSample, Math.Max(0, located.StartSample - _options.PreRollSamples)),
                        EndpointVadCandidate = !_endpoint.UnownedObserved,
                        PreferEndpointVadStart = false,
                    };
                }
                _capture.BeginPending(confirmed.TurnId, pendingPlan.StartSample, speechMs, lease.Selection.Source, lease.Selection.Samples);
                _pendingCapture = new PendingCapture(confirmed.TurnId, pendingPlan, lease);
                long operationId = lease.OperationId;
                TimeSpan budget = TimeSpan.FromMilliseconds(_audioFormat.SamplesToMilliseconds(_options.DrainTimeoutSamples));
                _pendingCaptureTimer = _timeProvider.CreateTimer(
                    _ => ExpirePendingCapture(operationId),
                    null,
                    budget,
                    Timeout.InfiniteTimeSpan);
                _replay.Clear();
            }
            else
            {
                // Start at the immutable confirmation boundary so callbacks that arrived while
                // the control-plane confirmation was queued can be replayed exactly once. If the
                // capture stream restarted while the confirmation was pending, the selection
                // coordinates belong to the retired epoch; rebase to the current timeline end
                // instead of pinning the feed watermark to pre-restart samples.
                rebasedAfterStreamEpochChange = lease.Selection.StreamEpoch != committedStreamEpoch;
                firstFeedSample = rebasedAfterStreamEpochChange ? _timeline.EndSample : lease.Selection.SelectedEndSample;
                _capture.Start(confirmed.TurnId, firstFeedSample, speechMs, lease.Selection.Source, lease.Selection.Samples);
                // A restarted stream may never publish the old VAD endpoint, and an already-ended
                // utterance cannot publish it again. Both cases must fall back to current DTD frames.
                bool hasCurrentEndpoint = _endpoint.UnownedObserved &&
                    _endpoint.UnownedGeneration == _timeline.Generation &&
                    _endpoint.UnownedStreamEpoch == committedStreamEpoch;
                _endpoint.ObserveCaptureStarted(lease.EndpointVadCandidate && !rebasedAfterStreamEpochChange && !hasCurrentEndpoint);
                if (hasCurrentEndpoint && EndpointPredatesSpeech(committedStreamEpoch, _endpoint.UnownedEndSample))
                {
                    _endpoint.ClearUnowned();
                    hasCurrentEndpoint = false;
                }
                bool endpointPredatesCaptureCommit =
                    _endpoint.UnownedObserved &&
                    _endpoint.UnownedProcessed &&
                    _endpoint.UnownedGeneration == _timeline.Generation &&
                    _endpoint.UnownedStreamEpoch == _timeline.StreamEpoch &&
                    _endpoint.UnownedEndSample >= firstFeedSample &&
                    _endpoint.UnownedEndSample <= Math.Max(_timeline.EndSample, firstFeedSample);
                if (endpointPredatesCaptureCommit)
                {
                    _endpoint.MarkCaptureSpeechEnded();
                    // The control plane consumed an endpoint at or after the immutable confirmation
                    // boundary, so both it and any already-observed DTD tail are fixed facts. An
                    // endpoint before that boundary is stale: later DTD evidence already extended
                    // the candidate past it and must remain eligible for ASR.
                    _capture.BeginDrain(_timeline.StreamEpoch, _timeline.EndSample, hasDiscontinuity: false, _options.DrainTimeoutSamples);
                }

                // Keep a pending acknowledgement only when its endpoint reaches this capture's
                // boundary. Older endpoints must not terminate a later capture when replayed.
                if (!hasCurrentEndpoint || _endpoint.UnownedProcessed || _endpoint.UnownedEndSample < firstFeedSample)
                    _endpoint.ClearUnowned();
                _endpoint.ClearCandidate();
            }
        }

        if (deferred)
        {
            ExecutePlayback(RestoreDucking());
            _logger.LogDebug(
                "主动近端 barge-in 登记待采集: turnId={TurnId}, planSource={PlanSource}, planStart={PlanStart}, " +
                "mainEnd={MainEnd}, epoch={StreamEpoch}, feedsInFlight={FeedsInFlight}.",
                confirmed.TurnId,
                capturePlan!.Value.Source,
                capturePlan.Value.StartSample,
                _timeline.EndSample,
                _timeline.StreamEpoch,
                _capture.DirectFeedsInFlight);
            TryActivatePendingCaptureOrdered();
            return true;
        }

        if (rebasedAfterStreamEpochChange)
        {
            _logger.LogDebug(
                "近端确认在 stream epoch 变化后重定位: turnId={TurnId}, selectedEnd={SelectedEnd}, " +
                "rebasedFeedStart={RebasedFeedStart}, committedEpoch={CommittedEpoch}.",
                confirmed.TurnId,
                lease.Selection.SelectedEndSample,
                firstFeedSample,
                committedStreamEpoch);
        }

        LogCaptureStarted(confirmed.TurnId, lease.Selection);
        lock (_feedGate)
        {
            DrainUnfedMainAudioBeforeFeed();
            DrainConfirmationTail();
        }
        bool finalize;
        TimeSpan speechDuration;
        ActiveEndDrainOutcome drainOutcome;
        lock (_lock)
        {
            // A capture callback can observe newer PCM after the catch-up snapshot. Only a
            // feed reservation may advance the watermark; observation alone does not own it.
            finalize = _capture.TryClaimEndFinalization(out speechDuration, out drainOutcome);
        }

        if (finalize)
            EndCapture(confirmed.TurnId, speechDuration, drainOutcome);

        return true;
    }

    /// <summary>
    /// Promotes the committed pending capture as soon as the main timeline has reached its start.
    /// The pre-roll snapshot and the audio buffered while the control plane stopped playback are
    /// fed as one half-open range; later recognition frames continue at its exclusive end.
    /// </summary>
    private bool TryActivatePendingCaptureOrdered(bool allowVadFallback = false)
    {
        PendingCaptureActivation activation;
        lock (_lock)
        {
            if (_pendingCapture is not { } pending)
                return false;
            if (_capture.State != NearEndCaptureStateMachine.Status.Pending || _capture.ActiveTurnId != pending.TurnId)
            {
                ClearPendingCaptureLocked();
                return false;
            }
            if (_capture.DirectFeedsInFlight != 0)
                return false;
            if (!_timeline.TryGetBounds(out long streamEpoch, out long retainedStartSample, out long endSample))
                return false;

            bool rebased = streamEpoch != pending.Plan.StreamEpoch;
            bool locatedVad = false;
            long startSample;
            if (rebased)
            {
                startSample = endSample;
            }
            else
            {
                long desiredStart = pending.Plan.StartSample;
                if (pending.Plan.PreferEndpointVadStart)
                {
                    locatedVad = TryGetEndpointVadCandidateStartLocked(pending.Lease.Decision, out _, out long vadStart, out bool waiting);
                    if (locatedVad)
                        desiredStart = Math.Min(desiredStart, Math.Max(0, vadStart - _options.PreRollSamples));
                    else if (waiting && !allowVadFallback)
                        return false;
                }

                startSample = Math.Max(desiredStart, retainedStartSample);
                if (startSample >= endSample)
                    return false;
            }

            // Use the frozen decision's accepted speech duration, not the pre-roll-to-activation
            // span; the latter depends on how late the scheduler activated the capture.
            double speechMs = rebased ? 0 : Math.Max(0, pending.Lease.Decision.AcceptedDurationMs);
            if (!_capture.ActivatePending(startSample, speechMs))
            {
                ClearPendingCaptureLocked();
                return false;
            }

            bool hasCurrentEndpoint = _endpoint.UnownedObserved &&
                _endpoint.UnownedGeneration == _timeline.Generation &&
                _endpoint.UnownedStreamEpoch == streamEpoch;
            _endpoint.ObserveCaptureStarted((locatedVad || pending.Plan.EndpointVadCandidate) && !rebased && !hasCurrentEndpoint);
            if (hasCurrentEndpoint && EndpointPredatesSpeech(streamEpoch, _endpoint.UnownedEndSample))
            {
                _endpoint.ClearUnowned();
                hasCurrentEndpoint = false;
            }
            bool endpointInsideCapture = hasCurrentEndpoint &&
                _endpoint.UnownedEndSample >= startSample &&
                _endpoint.UnownedEndSample <= endSample;
            if (endpointInsideCapture)
            {
                _endpoint.MarkCaptureSpeechEnded();
                _capture.BeginDrain(streamEpoch, _endpoint.UnownedEndSample, hasDiscontinuity: false, _options.DrainTimeoutSamples);
            }

            if (!hasCurrentEndpoint || _endpoint.UnownedProcessed || _endpoint.UnownedEndSample < startSample)
                _endpoint.ClearUnowned();
            _endpoint.ClearCandidate();
            ClearPendingCaptureLocked();
            activation = new PendingCaptureActivation(pending.TurnId, streamEpoch, startSample, endSample, rebased);
        }

        if (activation.Rebased)
        {
            _logger.LogDebug(
                "近端待采集在 stream epoch 变化后重定位: turnId={TurnId}, feedStart={FeedStart}, epoch={StreamEpoch}.",
                activation.TurnId,
                activation.StartSample,
                activation.StreamEpoch);
        }
        else
        {
            _logger.LogDebug(
                "主动近端 barge-in 开始捕获: turnId={TurnId}, feed=[{FeedStart},{FeedEnd}), epoch={StreamEpoch}, rebased={Rebased}.",
                activation.TurnId,
                activation.StartSample,
                activation.EndSampleExclusive,
                activation.StreamEpoch,
                activation.Rebased);
        }

        DrainUnfedMainAudioBeforeFeed();
        return true;
    }

    private bool EndpointPredatesSpeech(long streamEpoch, long endpointSample)
        => streamEpoch > 0 && streamEpoch == _captureSpeechEpoch && endpointSample < _earliestEndpointSample;

    /// <summary>
    /// A pending capture whose main timeline never reaches its start must still settle its
    /// validation turn. After the configured drain budget, start an empty capture and finish it so
    /// the turn can complete without audio instead of waiting forever for a missing callback.
    /// </summary>
    private void ExpirePendingCapture(long operationId)
    {
        lock (_feedGate)
            ExpirePendingCaptureOrdered(operationId);
    }

    private void ExpirePendingCaptureOrdered(long operationId)
    {
        PendingCapture expired;
        lock (_lock)
        {
            if (_pendingCapture is not { } pending || pending.Lease.OperationId != operationId)
                return;
            expired = pending;
        }

        // At the deadline a missing VAD location may fall back to the stored start. Deliver any
        // retained PCM before closing, including when the device never sends us another callback.
        if (TryActivatePendingCaptureOrdered())
            return;
        if (expired.Plan.PreferEndpointVadStart)
            TryActivatePendingCaptureOrdered(allowVadFallback: true);
        TimeSpan speechDuration;
        ActiveEndDrainOutcome outcome;
        bool finalized;
        long turnId = expired.TurnId;
        lock (_lock)
        {
            if (_capture.ActiveTurnId != turnId)
                return;

            long streamEpoch = _timeline.StreamEpoch;
            long endSample = _timeline.EndSample;
            if (_capture.Pending)
            {
                _capture.ActivatePending(endSample, speechDurationMs: 0);
                ClearPendingCaptureLocked();
            }
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
            _capture.BeginDrain(streamEpoch, endSample, hasDiscontinuity: false, _options.DrainTimeoutSamples);
            finalized = _capture.TryClaimEndFinalization(out speechDuration, out outcome);
        }

        _logger.LogDebug(
            "近端待采集超时收尾: turnId={TurnId}, planStart={PlanStart}, mainEnd={MainEnd}, finalized={Finalized}.",
            turnId,
            expired.Plan.StartSample,
            _timeline.EndSample,
            finalized);
        if (finalized)
            EndCapture(turnId, speechDuration, outcome);
    }

    public void Abort(in NearEndConfirmationLease lease)
    {
        PlaybackDuckingPolicy.Command restore;
        lock (_lock)
        {
            if (!_journal.TryAbort(in lease))
                return;

            _replay.Clear();
            restore = _ducking.Restore();
        }

        ExecutePlayback(in restore);
    }

    // ---------------------------------------------------------------------------------------
    // Capture control and reset
    // ---------------------------------------------------------------------------------------

    public bool StopCapture(long turnId, bool startCooldown)
    {
        SpeakerCandidateId cancelled;
        lock (_lock)
        {
            if (_capture.ActiveTurnId != turnId)
                return false;

            _capture.Stop();
            ClearPendingCaptureLocked();
            _endpoint.ClearCaptureObservation();
            _journal.Clear();
            _replay.Clear();
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
            if (startCooldown && _options.ActiveNearEndCooldownMs > 0)
            {
                Volatile.Write(ref _cooldownUntilTicks, Environment.TickCount64 + _options.ActiveNearEndCooldownMs);
            }
        }
        ExecuteCancelCandidate(cancelled);
        return true;
    }

    public void Reset()
    {
        long turnId;
        SpeakerCandidateId cancelled;
        PlaybackDuckingPolicy.Command restore;
        lock (_lock)
        {
            restore = _ducking.Restore();
            Volatile.Write(ref _dataPlaneFaultReported, 0);
            turnId = _capture.ActiveTurnId;
            _capture.Stop();
            ClearPendingCaptureLocked();
            _endpoint.ClearCaptureObservation();
            _journal.Clear();
            _replay.Clear();
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
            _cooldownUntilTicks = 0;
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
        }

        ExecuteCancelCandidate(cancelled);
        ExecutePlayback(in restore);
        _effects.Execute(new NearEndEffect.ResetRecognition(turnId));
        ClearAudioReservoirs();
    }

    public void ClearAudioReservoirs(long playbackTurnId = 0)
    {
        SpeakerCandidateId cancelled;
        PlaybackDuckingPolicy.Command restore;
        lock (_lock)
        {
            // A pending or active barge-in capture owns the current timeline and its pre-roll.
            // A later playback boundary must not erase the utterance it is still assembling.
            if (_pendingCapture is not null || _capture.ActiveTurnId != 0)
                return;

            if (playbackTurnId > 0)
            {
                bool sameOwner = (_candidate.Active && _candidate.OwnerTurnId == playbackTurnId) ||
                    (_journal.TryGetReplayRequest(out NearEndConfirmationLease queued) && queued.OwnerTurnId == playbackTurnId);
                if (sameOwner)
                    return;
            }

            cancelled = _candidate.CandidateId;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
            restore = _ducking.Restore();
            _timeline.Clear();
            _capture.ResetFeedWatermark();
            // Playback start retires the previous reply's pending and acquired confirmations
            // together with their audio. A producer may still enqueue its now-stale operation.
            _journal.Clear();
            _replay.Clear();
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
        }
        ExecuteCancelCandidate(cancelled);
        ExecutePlayback(in restore);
    }

    /// <summary>
    /// Aborts the capture registered for a Kernel-rejected authorization. The physical stop has not
    /// been issued on this path, so the interrupted reply continues and no captured utterance is
    /// allowed to outlive its turn.
    /// </summary>
    public void AbortCapture(long turnId)
    {
        if (!StopCapture(turnId, startCooldown: true))
            return;

        ExecutePlayback(RestoreDucking());
        _effects.Execute(new NearEndEffect.ResetRecognition(turnId));
    }

    // ---------------------------------------------------------------------------------------
    // Trigger path
    // ---------------------------------------------------------------------------------------

    private void Process(in DuplexAudioFrame frame)
    {
        if (!_options.Enable || _runtime.IsStopping)
        {
            ExecutePlayback(RestoreDucking());
            return;
        }

        long activeTurnId = _capture.ReadActiveTurnIdVolatile();
        if (activeTurnId != 0)
        {
            // VAD 已接管 (directFeed=false) 时窗口决策不会被使用, 不必逐帧计算;
            // ProcessActiveAudio 内部仍持锁复核 turn/directFeed, 此处只是快速跳过。
            if (!IsDirectFeedActive(activeTurnId))
            {
                ExecutePlayback(RestoreDucking());
                return;
            }

            ProcessActiveAudio(activeTurnId, in frame, ComputeFrameDurationMs(in frame), GetRecentDecision());
            return;
        }

        // 触发路径 (每帧执行): 先做无副作用的廉价短路, 再算窗口决策/预滚快照。
        if (Environment.TickCount64 < Volatile.Read(ref _cooldownUntilTicks))
            return;
        if (!_conversation.IsSpeakingWithoutPendingCandidate)
        {
            ExecutePlayback(RestoreDucking());
            SetCandidateClearOutcome("cleared:not-speaking");
            ClearSpeakerCandidate();
            return;
        }

        // Speaking 在 LLM 首个文本增量时就会建立，可能早于首个 TTS 音频入队。
        if (!_effects.IsPlaybackActive)
        {
            ExecutePlayback(RestoreDucking());
            SetCandidateClearOutcome("cleared:playback-inactive");
            ClearSpeakerCandidate();
            return;
        }

        bool confirmationQueued;
        lock (_lock)
            confirmationQueued = _journal.HasQueued;
        if (confirmationQueued)
            return;

        BargeInDecision decision = GetRecentDecision();
        if (!_authorization.RequiresSpeaker)
        {
            // Confirmation off: authorization is NotRequired. The acoustic Interrupt is the
            // whole decision and no speaker is ever queried.
            bool timelineCandidateActive;
            lock (_lock)
                timelineCandidateActive = _candidate.Active;
            UpdatePlaybackDucking(in frame, in decision);
            if (!timelineCandidateActive)
            {
                if (!_authorization.ShouldOpenCandidate(in decision))
                    return;
                BeginTimelineCandidate(in decision);
            }

            TryStartAuthorizedCandidate();
            return;
        }

        // Identity enabled: either DTD evidence or located endpoint-VAD speech can open the
        // candidate. DTD may miss double talk; the candidate's own speaker Match remains the
        // only authorization, including when every DTD frame is rejected.
        bool candidateActive;
        bool candidateAuthorized;
        long generation;
        lock (_lock)
        {
            candidateActive = _candidate.Active;
            candidateAuthorized = _candidate.Authorized;
            generation = _candidate.Generation;
            if (candidateActive)
                _candidate.ObserveDecision(in decision, _authorization.IsStrongNearEndDecision(in decision));
        }

        if (!candidateActive)
        {
            if (!BeginSpeakerCandidate(decision, in frame, out generation))
                return;
        }

        if (!candidateAuthorized)
        {
            bool invalidFrame;
            lock (_lock)
            {
                if (!_candidate.IsCurrent(generation))
                    return;
                invalidFrame = frame.HasDiscontinuity || _candidate.StreamEpoch != frame.StreamEpoch;
                if (!invalidFrame && !_candidate.EndpointObserved)
                    _candidate.ExtendEndSample(frame.EndSampleExclusive);
            }
            if (invalidFrame)
            {
                SpeakerEvidenceQuery empty = default;
                var invalid = SpeakerVerificationResult.Unavailable(0, in empty, SpeakerVerificationReason.TimelineGap);
                DenyCandidate(generation, in invalid);
                return;
            }

            Task? update = _effects.SpeakerEvidenceChanged;
            if (!TryEvaluateCurrentSpeaker(generation, frame.TimestampTicks, out var result, out TimeSpan remaining))
                return;
            if (result.IsFinal)
                ApplySpeakerVerdict(generation, in result);
            else if (update is not null)
                EnsureSpeakerWait(generation, update, remaining);
            return;
        }

        TryStartAuthorizedCandidate();
    }

    private void UpdatePlaybackDucking(in DuplexAudioFrame frame, in BargeInDecision decision)
    {
        PlaybackDuckingPolicy.CommandBatch commands;
        lock (_lock)
        {
            commands = _ducking.Update(
                in frame,
                in decision,
                _candidate.Active,
                _journal.HasPending,
                _authorization.AllowsPlaybackDucking,
                _conversation.CurrentTurnId,
                _options.PlaybackDuckGain,
                _options.PlaybackDuckReleaseMs);
        }

        PlaybackDuckingPolicy.Command first = commands.First;
        PlaybackDuckingPolicy.Command second = commands.Second;
        ExecutePlayback(in first);
        ExecutePlayback(in second);
    }

    private PlaybackDuckingPolicy.Command RestoreDucking()
    {
        lock (_lock)
            return _ducking.Restore();
    }

    private void ExecutePlayback(in PlaybackDuckingPolicy.Command command)
    {
        switch (command.Kind)
        {
            case PlaybackDuckingPolicy.Kind.None:
                return;
            case PlaybackDuckingPolicy.Kind.Duck:
                NearEndEffectResult result = _effects.Execute(
                    new NearEndEffect.DuckPlayback(
                        command.OwnerTurnId,
                        command.StreamEpoch,
                        command.Gain,
                        command.OriginalGain,
                        command.Sequence));
                if (!result.Succeeded)
                {
                    lock (_lock)
                        _ducking.RollbackDuck(command.Sequence);
                }
                return;
            case PlaybackDuckingPolicy.Kind.Restore:
                _effects.Execute(new NearEndEffect.RestorePlayback(command.OwnerTurnId, command.StreamEpoch, command.Sequence));
                return;
        }
    }

    private void DenyCandidate(long generation, in SpeakerVerificationResult verdict)
    {
        string outcome = verdict.Status switch
        {
            SpeakerVerificationStatus.Mismatch => "command-denied-mismatch",
            SpeakerVerificationStatus.InsufficientEvidence => "command-denied-insufficient-evidence",
            _ => $"command-denied-speaker-unavailable:{verdict.Reason}",
        };
        long ownerTurnId;
        long candidateId;
        BargeInDecision decision;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation) || _candidate.Authorized)
                return;

            _lastCandidateOutcome = outcome;
            ownerTurnId = _candidate.OwnerTurnId;
            candidateId = _candidate.CandidateId.Value;
            decision = _candidate.Decision;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
        }

        Interlocked.Increment(ref _commandDeniedCount);
        _logger.LogDebug(
            "{Outcome}: candidate={CandidateId}, ownerTurnId={OwnerTurnId}, status={Status}, reason={Reason}, " +
            "score={Score:F4}, threshold={Threshold:F4}, observed={Observed}, matched={Matched}/{Required}, " +
            "dtdInterrupt={DtdInterrupt}, decisionReason={DecisionReason}, range=[{StartSample},{EndSample}), epoch={StreamEpoch}. 原播放继续。",
            outcome,
            candidateId,
            ownerTurnId,
            verdict.Status,
            verdict.Reason,
            verdict.Score,
            verdict.Threshold,
            verdict.ObservedWindows,
            verdict.MatchedWindows,
            verdict.RequiredMatches,
            decision.Action == BargeInAction.Interrupt,
            decision.Reason,
            decision.StartSample,
            decision.EndSampleExclusive,
            decision.StreamEpoch);

        ExecuteCancelCandidate(new SpeakerCandidateId(candidateId));
        // A later window of the same utterance may still authorize. Its onset and primary-AEC
        // audio belong to the timeline, not to the rejected identity candidate.
        if (_options.ActiveNearEndCooldownMs > 0)
        {
            Volatile.Write(ref _cooldownUntilTicks, Environment.TickCount64 + _options.ActiveNearEndCooldownMs);
        }
    }

    private void SetCandidateClearOutcome(string outcome)
    {
        lock (_lock)
        {
            if (_candidate.Active)
                _lastCandidateOutcome = outcome;
        }
    }

    private void ClearSpeakerCandidate()
    {
        SpeakerCandidateId cancelled;
        lock (_lock)
        {
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
        }
        ExecuteCancelCandidate(cancelled);
    }

    private void ExecuteCancelCandidate(SpeakerCandidateId candidateId)
    {
        if (!candidateId.IsNone)
            _effects.Execute(new NearEndEffect.CancelSpeakerCandidate(candidateId));
    }

    /// <summary>
    /// Starts one non-blocking wait per candidate. It is the active wake-up path for a verdict
    /// that becomes final while no further duplex frame arrives; the per-frame poll below stays as
    /// the zero-latency cache-hit path.
    /// </summary>
    private void EnsureSpeakerWait(long generation, Task update, TimeSpan remaining)
    {
        if (!_authorization.RequiresSpeaker)
            return;

        CancellationTokenSource cancellation;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation) || _candidate.Authorized)
                return;
            if (_speakerWaitCancellation is not null && _speakerWaitGeneration == generation)
                return;

            CancelSpeakerWaitLocked();
            cancellation = new CancellationTokenSource();
            _speakerWaitCancellation = cancellation;
            _speakerWaitGeneration = generation;
        }

        _ = Task.Run(() => WaitForSpeakerDecisionAsync(generation, update, remaining, cancellation));
    }

    private void CancelSpeakerWaitLocked()
    {
        CancellationTokenSource? cancellation = _speakerWaitCancellation;
        _speakerWaitCancellation = null;
        _speakerWaitGeneration = 0;
        if (cancellation is null)
            return;

        try
        {
            cancellation.Cancel();
        }
        catch (ObjectDisposedException)
        {
        }
    }

    private async Task WaitForSpeakerDecisionAsync(long generation, Task update, TimeSpan remaining, CancellationTokenSource cancellation)
    {
        try
        {
            while (!cancellation.IsCancellationRequested)
            {
                try
                {
                    await update.WaitAsync(remaining, _timeProvider, cancellation.Token).ConfigureAwait(false);
                }
                catch (TimeoutException)
                {
                    // Recheck the actual job: completion or a new window may have raced the timer.
                }

                update = _effects.SpeakerEvidenceChanged!;
                if (!TryEvaluateCurrentSpeaker(generation, _timeProvider.GetTimestamp(), out var result, out remaining))
                    return;
                if (result.IsFinal)
                {
                    ApplySpeakerVerdict(generation, in result);
                    return;
                }
            }
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "主动近端声纹等待失败: generation={Generation}.", generation);
        }
        finally
        {
            lock (_lock)
            {
                if (ReferenceEquals(_speakerWaitCancellation, cancellation))
                {
                    _speakerWaitCancellation = null;
                    _speakerWaitGeneration = 0;
                }
            }
            cancellation.Dispose();
        }
    }

    private bool TryBuildSpeakerQueryLocked(long generation, long timestamp, out SpeakerEvidenceQuery query)
    {
        query = default;
        if (!_candidate.Active || _candidate.Authorized || !_candidate.IsCurrent(generation) || _candidate.MainStartSample < 0)
            return false;

        long candidateEnd = Math.Min(_candidate.EndSample, _candidate.DeadlineSample);
        if (candidateEnd <= _candidate.MainStartSample)
            return false;

        bool inputComplete = _candidate.EndSample >= _candidate.DeadlineSample || _candidate.EndpointObserved;
        query = new SpeakerEvidenceQuery(
            SpeakerVerificationPurpose.BargeInCommand,
            _candidate.StreamEpoch,
            _candidate.MainStartSample,
            candidateEnd,
            timestamp,
            InputComplete: inputComplete,
            StrongUpstreamEvidence: _candidate.StrongUpstreamEvidence,
            CandidateId: _candidate.CandidateId,
            OwnerTurnId: _candidate.OwnerTurnId);
        return true;
    }

    private bool TryEvaluateCurrentSpeaker(long generation, long timestamp, out SpeakerVerificationResult result, out TimeSpan remaining)
    {
        result = default;
        remaining = Timeout.InfiniteTimeSpan;
        // Serialize cache reads and deadline updates, so an older poll cannot reinstall a completed job.
        lock (_speakerEvaluationGate)
        {
            SpeakerEvidenceQuery query;
            lock (_lock)
            {
                if (!TryBuildSpeakerQueryLocked(generation, timestamp, out query))
                    return false;
                if (_candidate.SpeakerResult.IsFinal)
                {
                    result = _candidate.SpeakerResult;
                    return true;
                }
            }

            result = _effects.EvaluateSpeaker(in query);
            TimeSpan budget = _effects.GetInferenceTimeout(SpeakerVerificationPurpose.BargeInCommand);
            lock (_lock)
            {
                if (!_candidate.IsCurrent(generation) || _candidate.Authorized)
                    return false;
                long now = _timeProvider.GetTimestamp();
                _candidate.ObserveInferenceWait(in result, now);
                if (_candidate.InferenceReadyTimestamp >= 0)
                {
                    remaining = budget - _timeProvider.GetElapsedTime(_candidate.InferenceReadyTimestamp, now);
                    if (remaining <= TimeSpan.Zero)
                        result = result.WithInferenceTimeout();
                }
                _candidate.SetSpeakerResult(in result);
            }
        }
        return true;
    }

    private void ApplySpeakerVerdict(long generation, in SpeakerVerificationResult result)
    {
        if (result.Status != SpeakerVerificationStatus.Match)
        {
            DenyCandidate(generation, in result);
            return;
        }

        BargeInDecision decision;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation) || _candidate.Authorized)
                return;
            _candidate.Authorize();
            _lastCandidateOutcome = "command-authorized";
            decision = _candidate.AuthorizedDecision;
        }

        Interlocked.Increment(ref _commandAuthorizedCount);
        LogCommandAuthorized(generation, in result, in decision);
        TryStartAuthorizedCandidate();
    }

    private bool BeginSpeakerCandidate(in BargeInDecision decision, in DuplexAudioFrame frame, out long generation)
    {
        generation = 0;
        if (!_authorization.RequiresSpeaker)
            return false;
        if (decision.HasDiscontinuity || frame.HasDiscontinuity ||
            decision.StreamEpoch <= 0 ||
            decision.StreamEpoch != frame.StreamEpoch ||
            decision.StartSample < 0 ||
            decision.EndSampleExclusive <= decision.StartSample)
        {
            //_logger.LogDebug(
            //    "主动近端 barge-in 未创建说话人候选: reason=invalid-decision-timeline, " +
            //    "decision=[{DecisionStart},{DecisionEnd}), decisionEpoch={DecisionEpoch}, " +
            //    "frameEpoch={FrameEpoch}, discontinuity={Discontinuity}.",
            //    decision.StartSample,
            //    decision.EndSampleExclusive,
            //    decision.StreamEpoch,
            //    frame.StreamEpoch,
            //    decision.HasDiscontinuity || frame.HasDiscontinuity);
            return false;
        }

        // Check admission under the data-plane lock, but allocate the external candidate ID
        // outside it. Recheck below because the capture callback can retire VAD state meanwhile.
        lock (_lock)
        {
            if (!TryGetSpeakerCandidateAnchorLocked(in decision, out _, out _))
                return false;
        }

        SpeakerCandidateId candidateId = _effects.NewSpeakerCandidateId();
        long ownerTurnId = _conversation.CurrentTurnId;
        long mainStartSample;
        bool strongUpstreamEvidence;
        string trigger;
        lock (_lock)
        {
            if (_candidate.Active)
            {
                generation = _candidate.Generation;
                return true;
            }

            if (!TryGetSpeakerCandidateAnchorLocked(in decision, out long anchorSample, out trigger))
                return false;
            _candidate.BeginSpeaker(
                in decision,
                candidateId,
                ownerTurnId,
                anchorSample,
                frame.EndSampleExclusive,
                NearEndCaptureStateMachine.SaturatingAdd(anchorSample, _options.MaxCandidateSamples),
                _authorization.IsStrongNearEndDecision(in decision));
            _lastCandidateOutcome = "awaiting-speaker";
            generation = _candidate.Generation;
            mainStartSample = _candidate.MainStartSample;
            strongUpstreamEvidence = _candidate.StrongUpstreamEvidence;
        }

        _logger.LogDebug(
            "主动近端 barge-in 创建说话人候选(并行 DTD): candidate={CandidateId}, ownerTurnId={OwnerTurnId}, " +
            "decisionAction={Action}, decisionReason={DecisionReason}, scoreMs={ScoreMs:F0}/{RequiredMs:F0}, " +
            "range=[{StartSample},{EndSample}), speakerAnchor={SpeakerAnchor}, strongUpstream={StrongUpstream}, epoch={StreamEpoch}, trigger={Trigger}.",
            candidateId,
            ownerTurnId,
            decision.Action,
            decision.Reason,
            decision.ScoreDurationMs,
            decision.RequiredDurationMs,
            decision.StartSample,
            decision.EndSampleExclusive,
            mainStartSample,
            strongUpstreamEvidence,
            decision.StreamEpoch,
            trigger);
        return true;
    }

    private bool TryGetSpeakerCandidateAnchorLocked(in BargeInDecision decision, out long anchorSample, out string trigger)
    {
        bool dtdEvidence = _authorization.ShouldOpenCandidate(in decision);
        bool vadEvidence = !_endpoint.UnownedObserved &&
            TryGetEndpointVadCandidateStartLocked(in decision, out _, out _, out _) &&
            _endpoint.CandidateEndSample > decision.StartSample &&
            _endpoint.CandidateStartSample < decision.EndSampleExclusive;
        trigger = dtdEvidence ? "dtd" : "endpoint-vad";

        // The already-located utterance start can precede the DTD window. Keep it for speaker
        // windows too, so the 500ms evidence budget does not restart at a late DTD onset.
        anchorSample = vadEvidence
            ? _endpoint.CandidateStartSample
            : decision.FirstAcceptedStartSample >= decision.StartSample && decision.FirstAcceptedStartSample < decision.EndSampleExclusive
                ? decision.FirstAcceptedStartSample
                : decision.StartSample;
        return dtdEvidence || vadEvidence;
    }

    private void BeginTimelineCandidate(in BargeInDecision decision)
    {
        bool created = false;
        long ownerTurnId = _conversation.CurrentTurnId;
        lock (_lock)
        {
            if (_candidate.Active)
                return;

            // 主识别链可能落后于检测链。固定首次确认决定，后续重试不能改用一个
            // 持续向前滑动的新窗口，否则两条等速消费线程会永远追不上目标起点。
            _candidate.BeginAuthorizedTimeline(in decision, ownerTurnId);
            _lastCandidateOutcome = "created:timeline";
            created = true;
        }

        if (created)
        {
            _logger.LogDebug(
                "主动近端 barge-in 创建时间线候选: decisionReason={DecisionReason}, " +
                "scoreMs={ScoreMs:F0}/{RequiredMs:F0}, range=[{StartSample},{EndSample}), epoch={StreamEpoch}.",
                decision.Reason,
                decision.ScoreDurationMs,
                decision.RequiredDurationMs,
                decision.StartSample,
                decision.EndSampleExclusive,
                decision.StreamEpoch);
        }
    }

    private void TryStartAuthorizedCandidate()
    {
        BargeInDecision decision = default;
        NearEndCapturePlan plan = default;
        ActiveNearEndPreRollSelection selection = default;
        IReadOnlyList<float[]> initialAudio = [];
        long generation;
        bool ready;
        bool hasPlan;
        lock (_lock)
        {
            if (!_candidate.Active || !_candidate.Authorized || _capture.ActiveTurnId != 0)
            {
                _lastCandidateOutcome = "skipped:candidate-state-mismatch";
                return;
            }

            decision = _candidate.AuthorizedDecision;
            generation = _candidate.Generation;
            hasPlan = TryBuildCapturePlanLocked(in decision, out plan);
            if (hasPlan && !plan.PreferEndpointVadStart && _capture.DirectFeedsInFlight == 0 &&
                TrySnapshotCapturePlanLocked(in plan, in decision, out selection, out initialAudio))
            {
                ready = true;
            }
            else
            {
                ready = false;
                initialAudio = [];
                if (hasPlan)
                    selection = CreatePendingSelection(in plan, in decision);
            }
        }

        if (!hasPlan)
        {
            string outcome = "failed:missing-decision-timeline:terminal";
            bool outcomeChanged;
            lock (_lock)
            {
                outcomeChanged = !string.Equals(_lastCandidateOutcome, outcome, StringComparison.Ordinal);
                _lastCandidateOutcome = outcome;
            }
            if (outcomeChanged)
            {
                _logger.LogDebug(
                    "主动近端 barge-in 候选无法启动: outcome={Outcome}, decisionReason={DecisionReason}, " +
                    "scoreMs={ScoreMs:F0}/{RequiredMs:F0}, range=[{StartSample},{EndSample}), epoch={StreamEpoch}.",
                    outcome,
                    decision.Reason,
                    decision.ScoreDurationMs,
                    decision.RequiredDurationMs,
                    decision.StartSample,
                    decision.EndSampleExclusive,
                    decision.StreamEpoch);
            }

            ClearSpeakerCandidate();
            ExecutePlayback(RestoreDucking());
            return;
        }

        long operationId = _effects.NextOperationId();
        long queuedCandidateId = 0;
        bool queued = false;
        lock (_lock)
        {
            if (_journal.HasQueued)
                return;
            if (!_candidate.IsCurrent(generation))
                return;

            NearEndCapturePlan committedPlan = plan with
            {
                Ready = ready,
            };
            var lease = new NearEndConfirmationLease(
                operationId,
                _candidate.CandidateId,
                _candidate.OwnerTurnId,
                _candidate.StreamEpoch,
                selection,
                initialAudio,
                decision,
                committedPlan.EndpointVadCandidate,
                _candidate.IdentityRequired,
                _candidate.IdentityRequired && _candidate.Authorized,
                _candidate.SpeakerResult,
                committedPlan,
                _candidate.Generation);
            if (_journal.TryQueue(in lease))
            {
                _replay.ResetNextSample(ready ? selection.SelectedEndSample : plan.StartSample);
                _lastCandidateOutcome = ready ? "queued:confirmation" : "queued:confirmation-pending-capture";
                queuedCandidateId = _candidate.CandidateId.Value;
                queued = true;
            }
        }

        if (queued)
        {
            NearEndEffectResult enqueued = _effects.Execute(new NearEndEffect.QueueConfirmation(operationId));
            if (!enqueued.Succeeded)
            {
                PlaybackDuckingPolicy.Command restore;
                lock (_lock)
                {
                    restore = _ducking.Restore();
                    if (_journal.TryDiscardPending(operationId))
                    {
                        _lastCandidateOutcome = "failed:confirmation-overflow";
                    }
                }
                ExecutePlayback(in restore);
            }
        }

        if (queued)
        {
            _logger.LogDebug(
                "主动近端 barge-in 已提交确认: candidate={CandidateId}, ready={Ready}, planSource={PlanSource}, " +
                "planStart={PlanStart}, mainEnd={MainEnd}, epoch={StreamEpoch}.",
                queuedCandidateId,
                ready,
                plan.Source,
                plan.StartSample,
                _timeline.EndSample,
                plan.StreamEpoch);
        }

        // The candidate has been handed to the control plane (or failed terminally); its
        // module-private state is released either way. Cached embeddings stay reusable.
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation))
                return;
            _candidate.Clear();
            CancelSpeakerWaitLocked();
        }
    }

    /// <summary>
    /// Resolves the capture start from the frozen decision and the current endpoint/candidate
    /// evidence without requiring the main timeline to have reached any particular end. The plan
    /// remains valid while the recognition chain is paused; only the pre-roll snapshot waits.
    /// </summary>
    private bool TryBuildCapturePlanLocked(in BargeInDecision decision, out NearEndCapturePlan plan)
    {
        plan = default;
        int preRollSamples = _options.PreRollSamples;
        bool hasEndpointVadAudio = TryGetEndpointVadCandidateStartLocked(
            decision,
            out long endpointVadStreamEpoch,
            out long endpointVadStartSample,
            out bool waitForEndpointVadLocation);
        if (hasEndpointVadAudio)
        {
            plan = new NearEndCapturePlan(
                endpointVadStreamEpoch,
                Math.Max(0, endpointVadStartSample - preRollSamples),
                "main-vad-timeline",
                !_endpoint.UnownedObserved,
                PreferEndpointVadStart: false,
                Ready: false);
            return true;
        }

        bool preferEndpointVadStart = waitForEndpointVadLocation;
        if (_candidate.MainStartSample >= 0)
        {
            plan = new NearEndCapturePlan(
                _candidate.StreamEpoch,
                Math.Max(0, _candidate.MainStartSample - preRollSamples),
                "main-candidate",
                EndpointVadCandidate: false,
                preferEndpointVadStart,
                Ready: false);
            return true;
        }

        if (!NearEndDecisionRules.HasUsableTimeline(in decision))
            return false;

        plan = new NearEndCapturePlan(
            decision.StreamEpoch,
            ComputeDecisionHistoryStart(in decision, preRollSamples),
            "main-timeline",
            EndpointVadCandidate: false,
            PreferEndpointVadStart: waitForEndpointVadLocation,
            Ready: false);
        return true;
    }

    private bool TrySnapshotCapturePlanLocked(
        in NearEndCapturePlan plan,
        in BargeInDecision decision,
        out ActiveNearEndPreRollSelection selection,
        out IReadOnlyList<float[]> audio)
    {
        audio = [];
        selection = default;
        if (!_timeline.TryGetBounds(out long streamEpoch, out long retainedStart, out long endSample))
            return false;
        if (streamEpoch != plan.StreamEpoch)
            return false;
        if (endSample <= Math.Max(plan.StartSample, retainedStart))
            return false;
        if (!TrySnapshotTimelineAligned(
            plan.StreamEpoch,
            plan.StartSample,
            endSample,
            _options.TimelineAlignedAudioLimitSamples,
            out audio,
            out long selectedStartSample,
            out long selectedEndSample,
            out _))
        {
            audio = [];
            return false;
        }

        int samples = CountSamples(audio);
        selection = new ActiveNearEndPreRollSelection(
            plan.Source,
            samples,
            plan.StreamEpoch,
            decision.StartSample,
            decision.EndSampleExclusive,
            selectedStartSample,
            selectedEndSample);
        return samples > 0;
    }

    private static ActiveNearEndPreRollSelection CreatePendingSelection(in NearEndCapturePlan plan, in BargeInDecision decision)
        => new(
            plan.Source,
            Samples: 0,
            StreamEpoch: plan.StreamEpoch,
            DecisionStartSample: decision.StartSample,
            DecisionEndSample: decision.EndSampleExclusive,
            SelectedStartSample: plan.StartSample,
            SelectedEndSample: plan.StartSample);

    private bool TryGetEndpointVadCandidateStartLocked(
        in BargeInDecision decision,
        out long streamEpoch,
        out long startSample,
        out bool waitForLocation)
    {
        streamEpoch = 0;
        startSample = 0;
        waitForLocation = false;
        if (!_endpoint.CandidateActive ||
            _endpoint.CandidateGeneration != _timeline.Generation ||
            _endpoint.CandidateStreamEpoch == 0 ||
            _endpoint.CandidateStreamEpoch != _timeline.StreamEpoch ||
            !NearEndDecisionRules.HasUsableTimeline(in decision) ||
            _endpoint.CandidateStreamEpoch != decision.StreamEpoch)
        {
            return false;
        }

        if (!_endpoint.CandidateLocated)
        {
            // FireRed raises OnSpeechStarted immediately before its first OnSpeaking.
            // A detector-thread confirmation can race through that narrow interval;
            // retry on the next duplex frame so the exact VAD start is not lost.
            waitForLocation = !_endpoint.CandidateLocationAttempted;
            return false;
        }

        long ageSamples = decision.EndSampleExclusive - _endpoint.CandidateStartSample;
        long maxAgeSamples = AssistantAudioFormat.SamplesFromMilliseconds(
            _options.PlaybackMaxCandidateMs,
            _audioFormat.AudioSampleRate,
            _audioFormat.AudioChannels);
        if (ageSamples < 0 || ageSamples > maxAgeSamples || _endpoint.CandidateEndSample <= _endpoint.CandidateStartSample)
        {
            return false;
        }

        streamEpoch = _endpoint.CandidateStreamEpoch;
        startSample = _endpoint.CandidateStartSample;
        return true;
    }

    private long ComputeDecisionHistoryStart(in BargeInDecision decision, int preRollSamples)
    {
        int candidateMs = Math.Max(_options.PlaybackMaxCandidateMs, _options.ActiveNearEndWindowMs);
        long candidateSamples = AssistantAudioFormat.SamplesFromMilliseconds(
            candidateMs,
            _audioFormat.AudioSampleRate,
            _audioFormat.AudioChannels);
        long historySamples = NearEndCaptureStateMachine.SaturatingAdd(candidateSamples, preRollSamples);

        if (_options.PlaybackRecognitionAudioLimitMs > 0)
        {
            long recognitionLimitSamples = AssistantAudioFormat.SamplesFromMilliseconds(
                _options.PlaybackRecognitionAudioLimitMs,
                _audioFormat.AudioSampleRate,
                _audioFormat.AudioChannels);
            historySamples = Math.Min(historySamples, recognitionLimitSamples);
        }

        long historyStart = Math.Max(0, decision.EndSampleExclusive - historySamples);
        if (decision.FirstAcceptedStartSample >= 0 && decision.FirstAcceptedStartSample < decision.EndSampleExclusive)
        {
            // The first causal onset prevents a late retry from moving the candidate forward.
            long onsetStart = Math.Max(0, decision.FirstAcceptedStartSample - preRollSamples);
            historyStart = Math.Min(historyStart, onsetStart);
        }

        return historyStart;
    }

    private bool TrySnapshotTimelineAligned(
        long streamEpoch,
        long requestedStartSample,
        long requiredEndSample,
        int maxSamples,
        out IReadOnlyList<float[]> audio,
        out long selectedStartSample,
        out long selectedEndSample,
        out bool retryable)
    {
        audio = [];
        selectedStartSample = 0;
        selectedEndSample = 0;
        retryable = false;
        if (!_timeline.TryGetBounds(out long retainedEpoch, out long retainedStart, out long end))
        {
            retryable = true;
            return false;
        }
        if (retainedEpoch != streamEpoch)
        {
            retryable = retainedEpoch < streamEpoch;
            return false;
        }
        if (end < requiredEndSample)
        {
            retryable = true;
            return false;
        }
        if (requestedStartSample >= end)
        {
            retryable = true;
            return false;
        }
        // 刚清空过 reservoir 或处理线程长期领先时，目标起点可能已经不在缓存内。
        // 已丢失的历史无法恢复，但仍应从最早可用样本开始，不能让整次打断失效。
        selectedStartSample = Math.Max(requestedStartSample, retainedStart);
        if (!_timeline.TrySnapshotFrom(streamEpoch, selectedStartSample, maxSamples, out audio) || audio.Count == 0)
        {
            selectedStartSample = 0;
            retryable = true;
            return false;
        }

        selectedEndSample = selectedStartSample + CountSamples(audio);
        return true;
    }

    // ---------------------------------------------------------------------------------------
    // Confirmation tail replay
    // ---------------------------------------------------------------------------------------

    private void CaptureConfirmationTail(in DuplexAudioFrame frame, bool latestNearEndSpeech)
    {
        bool overflow = false;
        lock (_lock)
        {
            if (!_journal.TryGetReplayRequest(out NearEndConfirmationLease lease) || lease.Selection.StreamEpoch != frame.StreamEpoch)
            {
                return;
            }

            // A plan-based capture owns the processed main timeline end to end. Replaying the DTD
            // tail would let detection-chain coordinates overtake pre-roll frames that the main
            // chain has not delivered yet, permanently skipping those earlier samples. The
            // committed capture feeds the authoritative main timeline instead.
            if (lease.CapturePlan is not null)
                return;

            // The confirmation can be formed only after several DTD frames have already
            // advanced beyond its immutable selection boundary. The first replay must
            // therefore start at that boundary rather than at the current frame start.
            long tailStart = Math.Max(lease.Selection.SelectedEndSample, _replay.NextSample);
            if (frame.EndSampleExclusive <= tailStart)
                return;

            var tail = new NearEndConfirmationTailFrame(
                _timeline.Generation,
                frame.StreamEpoch,
                tailStart,
                frame.EndSampleExclusive,
                frame.HasDiscontinuity,
                IsLatestSpeechActiveLocked(latestNearEndSpeech),
                ComputeTailDurationMs(in frame, tailStart));
            // The write stays inside the state lock so a concurrent Reset/Abort/Clear cannot
            // interleave between the queue clear and this metadata write.
            if (!_replay.TryWrite(in tail))
            {
                overflow = true;
            }
            else if (_journal.IsQueuedOperation(lease.OperationId))
            {
                _replay.AdvanceNextSample(frame.EndSampleExclusive);
            }
        }

        if (overflow)
            ReportDataPlaneFault("confirmation-tail-overflow");
    }

    private void DrainConfirmationTail()
    {
        if (!_replay.TryEnterDrain())
            return;

        try
        {
            while (_replay.TryPeek(out NearEndConfirmationTailFrame tail))
            {
                long activeTurnId;
                bool discard;
                bool fault;
                float[] audio = [];
                lock (_lock)
                {
                    activeTurnId = _capture.ActiveTurnId;
                    if (activeTurnId == 0 || !_capture.DirectFeed)
                        return;
                    if (tail.Generation != _timeline.Generation || tail.StreamEpoch != _timeline.StreamEpoch)
                    {
                        discard = tail.Generation < _timeline.InvalidatedGeneration;
                        fault = !discard;
                    }
                    else
                    {
                        discard = false;
                        fault = false;
                        if (!_timeline.TrySnapshotRange(tail.StreamEpoch, tail.StartSample, tail.EndSampleExclusive, out audio))
                        {
                            return;
                        }
                    }

                    if (discard || !fault)
                        _ = _replay.TryRead(out _);

                    // Apply resumed speech before its PCM can reach the old drain target.
                    // Duration/silence accounting still runs once, after the feed below.
                    if (!discard && !fault && tail.LatestSpeechActive && !tail.HasDiscontinuity &&
                        tail.EndSampleExclusive > _capture.EndDrainTargetSample)
                    {
                        _capture.TryResumeFromDrain(tail.StreamEpoch);
                    }
                }

                if (fault)
                {
                    ReportDataPlaneFault("confirmation-tail-timeline-mismatch");
                    return;
                }

                // ObserveRecognitionAudio already retired this range from the timeline. Only
                // this drain owns queue reads, so discard the obsolete metadata here rather
                // than treating the declared capture reset as a data-plane fault.
                if (discard)
                    continue;

                FeedFromRecognitionAudioCore(
                    audio,
                    new MainAudioObservation(
                        tail.Generation,
                        tail.StreamEpoch,
                        tail.StartSample,
                        tail.EndSampleExclusive,
                        tail.HasDiscontinuity));
                ProcessActiveAudioFacts(
                    activeTurnId,
                    tail.StreamEpoch,
                    tail.EndSampleExclusive,
                    tail.HasDiscontinuity,
                    tail.DurationMs,
                    tail.LatestSpeechActive);
            }
        }
        finally
        {
            _replay.ExitDrain();
        }
    }

    private bool NeedsBacklogDrain(in MainAudioObservation observation)
    {
        lock (_lock)
        {
            if (_capture.ActiveTurnId == 0 || !_capture.DirectFeed)
                return false;
            return observation.StartSample != _capture.DirectFeedNextSample ||
                observation.EndSample != _timeline.EndSample;
        }
    }

    private void DrainUnfedMainAudioBeforeFeed()
    {
        long generation;
        long streamEpoch;
        long startSample;
        long endSample;
        float[] audio;
        lock (_lock)
        {
            if (_capture.ActiveTurnId == 0 || !_capture.DirectFeed)
                return;
            startSample = _capture.DirectFeedNextSample;
            endSample = _timeline.EndSample;
            if (endSample <= startSample)
                return;
            generation = _timeline.Generation;
            streamEpoch = _timeline.StreamEpoch;
            if (!_timeline.TrySnapshotRange(streamEpoch, startSample, endSample, out audio))
                return;
        }

        FeedFromRecognitionAudioCore(audio, new MainAudioObservation(generation, streamEpoch, startSample, endSample, HasDiscontinuity: false));
    }

    // ---------------------------------------------------------------------------------------
    // Direct-feed facts and finalization
    // ---------------------------------------------------------------------------------------

    private void ProcessActiveAudio(long activeTurnId, in DuplexAudioFrame frame, double frameMs, BargeInDecision decision)
    {
        ProcessActiveAudioFacts(
            activeTurnId,
            frame.StreamEpoch,
            frame.EndSampleExclusive,
            frame.HasDiscontinuity,
            frameMs,
            IsLatestSpeechActive(in decision));
    }

    private void ProcessActiveAudioFacts(
        long activeTurnId,
        long streamEpoch,
        long endSampleExclusive,
        bool hasDiscontinuity,
        double frameMs,
        bool latestSpeechActive)
    {
        NearEndCaptureStateMachine.Finalization finalization;
        lock (_lock)
        {
            finalization = _capture.ObserveAudioFrame(
                activeTurnId,
                streamEpoch,
                endSampleExclusive,
                hasDiscontinuity,
                frameMs,
                latestSpeechActive,
                _options.ActiveNearEndEndSilenceMs,
                _options.DrainTimeoutSamples);
        }

        if (finalization.Claimed)
            EndCapture(finalization.TurnId, finalization.SpeechDuration, finalization.Outcome);
    }

    private bool IsLatestSpeechActive(in BargeInDecision decision)
        => IsLatestSpeechActive(decision.LatestFrameAccepted);

    private bool IsLatestSpeechActive(bool latestNearEndSpeech)
    {
        lock (_lock)
            return IsLatestSpeechActiveLocked(latestNearEndSpeech);
    }

    private bool IsLatestSpeechActiveLocked(bool latestNearEndSpeech)
    {
        // DTD remains the acoustic authority. Once a candidate is authorized, the main
        // recognition VAD may extend its endpoint so low-energy syllables are not truncated.
        return _endpoint.CaptureObserved ? _endpoint.CaptureSpeechActive : latestNearEndSpeech;
    }

    private bool IsDirectFeedActive(long activeTurnId)
    {
        lock (_lock)
            return _capture.IsActive(activeTurnId) && _capture.DirectFeed;
    }

    private void ClearPendingCaptureLocked()
    {
        _pendingCaptureTimer?.Dispose();
        _pendingCaptureTimer = null;
        _pendingCapture = null;
    }

    private void EndCapture(long turnId, TimeSpan speechDuration, ActiveEndDrainOutcome drainOutcome)
    {
        ActiveCaptureAudioStats stats = SnapshotActiveCaptureStats();
        if (!StopCapture(turnId, startCooldown: true))
            return;

        LogCaptureEnded(turnId, speechDuration, stats.PreRollSource, stats.PreRollSamples, stats.DirectFeedSamples, drainOutcome);

        _effects.Execute(new NearEndEffect.PublishSpeechEnded(turnId, speechDuration));
    }

    private ActiveCaptureAudioStats SnapshotActiveCaptureStats()
    {
        lock (_lock)
        {
            return new ActiveCaptureAudioStats(_capture.PreRollSource, _capture.PreRollSamples, _capture.DirectFeedSamples);
        }
    }

    private readonly record struct ActiveCaptureAudioStats(string PreRollSource, int PreRollSamples, long DirectFeedSamples);

    private void ReportDataPlaneFault(string detail)
    {
        if (Interlocked.Exchange(ref _dataPlaneFaultReported, 1) != 0)
            return;

        _effects.Execute(
            new NearEndEffect.ReportDataPlaneFault(new AssistantFailure("VBX-CTRL-018", "capacity", "active-near-end-data-plane", detail)));
    }

    // ---------------------------------------------------------------------------------------
    // Diagnostics
    // ---------------------------------------------------------------------------------------

    private BargeInDecision GetRecentDecision()
        => _decisions.GetRecentDecision(TimeSpan.FromMilliseconds(_options.ActiveNearEndWindowMs), _options.MinDurationMs, _options.MinRms);

    private void LogCommandAuthorized(long generation, in SpeakerVerificationResult verdict, in BargeInDecision decision)
    {
        long ownerTurnId;
        long streamEpoch;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation))
                return;
            ownerTurnId = _candidate.OwnerTurnId;
            streamEpoch = _candidate.StreamEpoch;
        }

        _logger.LogDebug(
            "command-authorized: candidate={CandidateId}, ownerTurnId={OwnerTurnId}, reason={Reason}, " +
            "score={Score:F4}, threshold={Threshold:F4}, matched={Matched}/{Required}, dtdInterrupt={DtdInterrupt}, " +
            "range=[{StartSample},{EndSample}), epoch={StreamEpoch}.",
            verdict.CandidateId,
            ownerTurnId,
            verdict.Reason,
            verdict.Score,
            verdict.Threshold,
            verdict.MatchedWindows,
            verdict.RequiredMatches,
            decision.Action == BargeInAction.Interrupt,
            verdict.EvidenceStartSample,
            verdict.EvidenceEndSampleExclusive,
            streamEpoch);
    }

    private void LogCaptureStarted(long turnId, in ActiveNearEndPreRollSelection preRoll)
    {
        if (!_logger.IsEnabled(LogLevel.Debug))
            return;

        long mainLeadSamples = preRoll.SelectedEndSample - preRoll.DecisionEndSample;
        _logger.LogDebug(
            "主动近端 barge-in 开始捕获: turnId={TurnId}, preRollSource={PreRollSource}, " +
            "preRollSamples={PreRollSamples}, preRollMs={PreRollMs:F0}, epoch={StreamEpoch}, " +
            "decision=[{DecisionStart},{DecisionEnd}), selected=[{SelectedStart},{SelectedEnd}), " +
            "mainLead={MainLeadSamples}/{MainLeadMs:F0}ms.",
            turnId,
            preRoll.Source,
            preRoll.Samples,
            _audioFormat.SamplesToMilliseconds(preRoll.Samples),
            preRoll.StreamEpoch,
            preRoll.DecisionStartSample,
            preRoll.DecisionEndSample,
            preRoll.SelectedStartSample,
            preRoll.SelectedEndSample,
            mainLeadSamples,
            _audioFormat.SamplesToMilliseconds(mainLeadSamples));
    }

    private void LogCaptureEnded(
        long turnId,
        TimeSpan speechDuration,
        string preRollSource,
        int preRollSamples,
        long directFeedSamples,
        ActiveEndDrainOutcome drainOutcome)
    {
        if (!_logger.IsEnabled(LogLevel.Debug))
            return;

        long totalSamples = preRollSamples + directFeedSamples;
        _logger.LogDebug(
            "主动近端 barge-in ASR 音频: turnId={TurnId}, preRollSource={PreRollSource}, preRoll={PreRollSamples}/{PreRollMs:F0}ms, direct={DirectSamples}/{DirectMs:F0}ms, total={TotalSamples}/{TotalMs:F0}ms, speechDuration={SpeechDurationMs:F0}ms, drain={DrainReason}, epoch={DrainStreamEpoch}, target={DrainTargetSample}, fedThrough={DrainFedThroughSample}, missing={DrainMissingSamples}/{DrainMissingMs:F0}ms.",
            turnId,
            preRollSource,
            preRollSamples,
            _audioFormat.SamplesToMilliseconds(preRollSamples),
            directFeedSamples,
            _audioFormat.SamplesToMilliseconds(directFeedSamples),
            totalSamples,
            _audioFormat.SamplesToMilliseconds(totalSamples),
            speechDuration.TotalMilliseconds,
            drainOutcome.Reason,
            drainOutcome.StreamEpoch,
            drainOutcome.TargetSample,
            drainOutcome.FedThroughSample,
            drainOutcome.MissingSamples,
            _audioFormat.SamplesToMilliseconds(drainOutcome.MissingSamples));
    }

    private static int CountSamples(IReadOnlyList<float[]> chunks)
    {
        int total = 0;
        for (int i = 0; i < chunks.Count; i++)
            total += chunks[i].Length;
        return total;
    }

    private static double ComputeFrameDurationMs(in DuplexAudioFrame frame)
    {
        if (frame.SampleRate <= 0 || frame.FrameCount <= 0)
            return 20;

        return Math.Clamp(frame.FrameCount * 1000.0 / frame.SampleRate, 1, 100);
    }

    /// <summary>
    /// Duration of the sub-frame tail [tailStart, EndSampleExclusive). Capture coordinates count
    /// interleaved samples, so the frame fraction divides by frame samples per channel, not by the
    /// raw interleaved sample count.
    /// </summary>
    internal static double ComputeTailDurationMs(in DuplexAudioFrame frame, long tailStartSample)
    {
        if (frame.SampleRate <= 0 || frame.FrameCount <= 0 || frame.Channels == 0)
            return 0;

        long interleavedSamples = frame.EndSampleExclusive - tailStartSample;
        if (interleavedSamples <= 0)
            return 0;

        double samplesPerChannel = frame.FrameCount * (double)frame.Channels;
        return ComputeFrameDurationMs(in frame) * (interleavedSamples / samplesPerChannel);
    }
}
