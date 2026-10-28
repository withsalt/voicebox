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
    private readonly object _lock = new();
    private long _cooldownUntilTicks;
    private string _lastCandidateOutcome = "none";
    private long _commandAuthorizedCount;
    private long _commandDeniedCount;
    private int _dataPlaneFaultReported;

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
            return new ActiveNearEndProcessingState(
                _capture.ActiveTurnId != 0,
                _candidate.Active || _endpoint.CandidateActive);
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
                if (_endpoint.CaptureObserved)
                    return;
                _endpoint.MarkCaptureSpeechActive();
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
                return;
            }

            // IVoiceActivityDetectService promises that OnSpeaking contains audio from Predict.
            // Locate that exact chunk in the main-AEC timeline instead of estimating its start
            // from the later fused decision window.
            if (!_endpoint.CandidateActive)
                _endpoint.BeginCandidate();

            if (_endpoint.CandidateGeneration != _timeline.Generation ||
                !_endpoint.TryLocate(audio, out long streamEpoch, out long startSample, out long endSample) ||
                (_endpoint.CandidateStreamEpoch != 0 &&
                 _endpoint.CandidateStreamEpoch != streamEpoch))
            {
                return;
            }

            _endpoint.ExtendCandidate(streamEpoch, startSample, endSample);
        }
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
            _endpoint.ClearCandidate();
            if (_capture.DirectFeed)
            {
                _endpoint.MarkCaptureSpeechEnded();
                bool dtdTailPredatesCaptureCommit = captureCommittedAfterEndpoint &&
                    _endpoint.UnownedObserved &&
                    _endpoint.UnownedGeneration == _timeline.Generation &&
                    _endpoint.UnownedStreamEpoch == _timeline.StreamEpoch &&
                    _timeline.EndSample > _endpoint.UnownedEndSample;
                if (dtdTailPredatesCaptureCommit)
                {
                    activeTurnId = _capture.ActiveTurnId;
                    // The control plane committed the capture after both the VAD endpoint and a
                    // later DTD frame. Those tail frames cannot be replayed as future callbacks,
                    // so drain to the known main-AEC boundary now.
                    _capture.BeginDrain(
                        _timeline.StreamEpoch,
                        _timeline.EndSample,
                        hasDiscontinuity: false,
                        _options.DrainTimeoutSamples);
                    finalize = _capture.TryClaimEndFinalization(out speechDuration, out drainOutcome);
                }

                _endpoint.ClearUnowned();
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

    public void ObserveDuplexAudio(
        in DuplexAudioFrame frame,
        bool playbackStartupGuardActive,
        bool latestNearEndSpeech = false)
    {
        if (playbackStartupGuardActive)
        {
            ExecutePlayback(RestoreDucking());
            return;
        }

        Process(in frame);
        CaptureConfirmationTail(in frame, latestNearEndSpeech);
    }

    public MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame)
    {
        ReadOnlySpan<float> audio = frame.Audio.Span;
        if (audio.IsEmpty)
            return default;

        bool retryAuthorizedCandidate = false;
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
                _candidate.Clear();
                _endpoint.ClearCandidate();
                if (_capture.ActiveTurnId == 0)
                    _capture.SetFeedWatermark(frame.SourceRange.EndSampleExclusive);
            }

            observation = new MainAudioObservation(
                _timeline.Generation,
                frame.SourceRange.StreamEpoch,
                frame.SourceRange.StartSample,
                frame.SourceRange.EndSampleExclusive,
                discontinuity);
            if (!discontinuity &&
                _capture.ActiveTurnId == 0 &&
                _candidate.Active &&
                _candidate.Authorized)
            {
                retryAuthorizedCandidate = true;
            }
        }

        ExecutePlayback(in restore);
        if (retryAuthorizedCandidate)
            TryStartAuthorizedCandidate();

        return observation;
    }

    public void FeedFromRecognitionAudio(
        ReadOnlySpan<float> audio,
        MainAudioObservation observation)
    {
        if (audio.IsEmpty)
            return;

        if (!_replay.DrainActive)
            DrainConfirmationTail();

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
                NearEndEffectResult result = _effects.FeedRecognition(
                    reservation.TurnId,
                    audio.Slice(offset, length));
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
                _capture.CompleteFeed(
                    reservation.TurnId,
                    in reservation,
                    failed: feedError is not null || !accepted);
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

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
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
        // The control consumer owns the conversation snapshot; read it outside the data-plane
        // lock and require the decision to interrupt the same reply that admitted the candidate.
        if (lease.OwnerTurnId <= 0 ||
            lease.OwnerTurnId != _conversation.CurrentTurnId ||
            lease.OwnerTurnId != confirmed.InterruptedTurnId)
        {
            return false;
        }

        bool rebasedAfterStreamEpochChange;
        long firstFeedSample;
        long committedStreamEpoch;
        lock (_lock)
        {
            if (_capture.ActiveTurnId != 0)
                return false;
            if (!_journal.TryCommit(in lease))
                return false;

            _capture.ClearEndDrain();
            Volatile.Write(ref _dataPlaneFaultReported, 0);
            double speechMs = Math.Max(
                0,
                _audioFormat.SamplesToMilliseconds(confirmed.ConfirmedCandidateSamples));
            // Start at the immutable confirmation boundary so callbacks that arrived while the
            // control-plane confirmation was queued can be replayed exactly once. If the capture
            // stream restarted while the confirmation was pending, the selection coordinates
            // belong to the retired epoch; rebase to the current timeline end instead of pinning
            // the feed watermark to pre-restart samples.
            committedStreamEpoch = _timeline.StreamEpoch;
            rebasedAfterStreamEpochChange = lease.Selection.StreamEpoch != committedStreamEpoch;
            firstFeedSample = rebasedAfterStreamEpochChange
                ? _timeline.EndSample
                : lease.Selection.SelectedEndSample;
            _capture.Start(
                confirmed.TurnId,
                firstFeedSample,
                speechMs,
                lease.Selection.Source,
                lease.Selection.Samples);
            // A restarted stream may never publish the old VAD utterance's endpoint. Until VAD
            // observes speech on the new stream, use its current DTD frames for end detection.
            _endpoint.ObserveCaptureStarted(lease.EndpointVadCandidate && !rebasedAfterStreamEpochChange);
            bool endpointPredatesCaptureCommit =
                _endpoint.UnownedObserved &&
                _endpoint.UnownedProcessed &&
                _endpoint.UnownedGeneration == _timeline.Generation &&
                _endpoint.UnownedStreamEpoch == _timeline.StreamEpoch &&
                _endpoint.UnownedEndSample >= firstFeedSample &&
                _endpoint.UnownedEndSample <= Math.Max(
                    _timeline.EndSample,
                    firstFeedSample);
            if (endpointPredatesCaptureCommit)
            {
                _endpoint.MarkCaptureSpeechEnded();
                // The control plane consumed an endpoint at or after the immutable confirmation
                // boundary, so both it and any already-observed DTD tail are fixed facts. An
                // endpoint before that boundary is stale: later DTD evidence already extended
                // the candidate past it and must remain eligible for ASR.
                _capture.BeginDrain(
                    _timeline.StreamEpoch,
                    _timeline.EndSample,
                    hasDiscontinuity: false,
                    _options.DrainTimeoutSamples);
            }

            _endpoint.ClearUnowned();
            _endpoint.ClearCandidate();
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
        DrainConfirmationTail();
        DrainUnfedMainAudioAfterConfirmation();
        bool finalize;
        TimeSpan speechDuration;
        ActiveEndDrainOutcome drainOutcome;
        lock (_lock)
        {
            if (_capture.ActiveTurnId != 0 && _capture.DirectFeed && !_capture.EndDrainPending)
            {
                // Confirmation is requested on the duplex callback, before the same callback's
                // processed-capture frame is observed. Any remaining gap is already-owned main
                // audio and must be fed from the timeline; only a range that can no longer be
                // snapshotted is skipped.
                _capture.SetFeedWatermarkAtLeast(_timeline.EndSample);
            }

            finalize = _capture.TryClaimEndFinalization(out speechDuration, out drainOutcome);
        }

        if (finalize)
            EndCapture(confirmed.TurnId, speechDuration, drainOutcome);

        return true;
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
            _endpoint.ClearCaptureObservation();
            _journal.Clear();
            _replay.Clear();
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
            if (startCooldown && _options.ActiveNearEndCooldownMs > 0)
            {
                Volatile.Write(
                    ref _cooldownUntilTicks,
                    Environment.TickCount64 + _options.ActiveNearEndCooldownMs);
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
            _endpoint.ClearCaptureObservation();
            _journal.Clear();
            _replay.Clear();
            _endpoint.ClearCandidate();
            _endpoint.ClearUnowned();
            _cooldownUntilTicks = 0;
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
        }

        ExecuteCancelCandidate(cancelled);
        ExecutePlayback(in restore);
        _effects.Execute(new NearEndEffect.ResetRecognition(turnId));
        ClearAudioReservoirs();
    }

    public void ClearAudioReservoirs()
    {
        SpeakerCandidateId cancelled;
        PlaybackDuckingPolicy.Command restore;
        lock (_lock)
        {
            cancelled = _candidate.CandidateId;
            _candidate.Clear();
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
                return;

            ProcessActiveAudio(
                activeTurnId,
                in frame,
                ComputeFrameDurationMs(in frame),
                GetRecentDecision());
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

        // KWS enabled: Speaker and DTD run in parallel on the same candidate window. The
        // candidate opens on valid near-end speech evidence, not on Interrupt.
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
            if (!_authorization.ShouldOpenCandidate(in decision))
                return;
            if (!BeginSpeakerCandidate(decision, in frame, out generation))
                return;
        }

        if (!candidateAuthorized)
        {
            BargeInSpeakerGateStatus speakerStatus = GetSpeakerGateStatus(
                in frame,
                generation,
                out SpeakerVerificationResult verdict);
            if (speakerStatus == BargeInSpeakerGateStatus.Pending)
            {
                // Pending: the interrupted reply keeps playing at its normal gain. A DTD
                // Interrupt that already arrived is only recorded on the candidate.
                return;
            }
            if (speakerStatus == BargeInSpeakerGateStatus.Rejected)
            {
                DenyCandidate(generation, in verdict, in decision);
                return;
            }

            if (!TryAuthorizeSpeakerCandidate(generation))
                return;
            Interlocked.Increment(ref _commandAuthorizedCount);
            LogCommandAuthorized(generation, in verdict, in decision);
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
                _effects.Execute(
                    new NearEndEffect.RestorePlayback(
                        command.OwnerTurnId,
                        command.StreamEpoch,
                        command.Sequence));
                return;
        }
    }

    private void DenyCandidate(
        long generation,
        in SpeakerVerificationResult verdict,
        in BargeInDecision decision)
    {
        string outcome = verdict.Status switch
        {
            SpeakerVerificationStatus.Mismatch => "command-denied-mismatch",
            SpeakerVerificationStatus.InsufficientEvidence => "command-denied-insufficient-evidence",
            _ => $"command-denied-speaker-unavailable:{verdict.Reason}",
        };
        long ownerTurnId;
        long candidateId;
        lock (_lock)
        {
            if (_candidate.Active)
                _lastCandidateOutcome = outcome;
            ownerTurnId = _candidate.OwnerTurnId;
            candidateId = _candidate.CandidateId.Value;
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

        if (!TryClearSpeakerCandidate(generation))
            return;
        ClearAudioReservoirs();
        if (_options.ActiveNearEndCooldownMs > 0)
        {
            Volatile.Write(
                ref _cooldownUntilTicks,
                Environment.TickCount64 + _options.ActiveNearEndCooldownMs);
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
        }
        ExecuteCancelCandidate(cancelled);
    }

    private bool TryClearSpeakerCandidate(long generation)
    {
        SpeakerCandidateId cancelled;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation))
                return false;

            cancelled = _candidate.CandidateId;
            _candidate.Clear();
        }

        ExecuteCancelCandidate(cancelled);
        return true;
    }

    private void ExecuteCancelCandidate(SpeakerCandidateId candidateId)
    {
        if (!candidateId.IsNone)
            _effects.Execute(new NearEndEffect.CancelSpeakerCandidate(candidateId));
    }

    private bool TryAuthorizeSpeakerCandidate(long generation)
    {
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation))
                return false;

            _candidate.Authorize();
            _lastCandidateOutcome = "command-authorized";
            return true;
        }
    }

    private bool BeginSpeakerCandidate(
        in BargeInDecision decision,
        in DuplexAudioFrame frame,
        out long generation)
    {
        generation = 0;
        if (!_authorization.RequiresSpeaker)
            return false;
        if (decision.HasDiscontinuity ||
            decision.StreamEpoch <= 0 ||
            decision.StreamEpoch != frame.StreamEpoch ||
            decision.StartSample < 0 ||
            decision.EndSampleExclusive <= decision.StartSample)
        {
            _logger.LogDebug(
                "主动近端 barge-in 未创建说话人候选: reason=invalid-decision-timeline, " +
                "decision=[{DecisionStart},{DecisionEnd}), decisionEpoch={DecisionEpoch}, " +
                "frameEpoch={FrameEpoch}, discontinuity={Discontinuity}.",
                decision.StartSample,
                decision.EndSampleExclusive,
                decision.StreamEpoch,
                frame.StreamEpoch,
                decision.HasDiscontinuity || frame.HasDiscontinuity);
            return false;
        }

        SpeakerCandidateId candidateId = _effects.NewSpeakerCandidateId();
        long ownerTurnId = _conversation.CurrentTurnId;
        long mainStartSample;
        bool strongUpstreamEvidence;
        lock (_lock)
        {
            if (_candidate.Active)
            {
                generation = _candidate.Generation;
                return true;
            }

            long anchorSample = decision.FirstAcceptedStartSample >= decision.StartSample &&
                decision.FirstAcceptedStartSample < decision.EndSampleExclusive
                    ? decision.FirstAcceptedStartSample
                    : decision.StartSample;
            _candidate.BeginSpeaker(
                in decision,
                candidateId,
                ownerTurnId,
                anchorSample,
                frame.EndSampleExclusive,
                NearEndCaptureStateMachine.SaturatingAdd(
                    anchorSample,
                    _options.MaxCandidateSamples),
                _authorization.IsStrongNearEndDecision(in decision));
            _lastCandidateOutcome = "awaiting-speaker";
            generation = _candidate.Generation;
            mainStartSample = _candidate.MainStartSample;
            strongUpstreamEvidence = _candidate.StrongUpstreamEvidence;
        }

        _logger.LogDebug(
            "主动近端 barge-in 创建说话人候选(并行 DTD): candidate={CandidateId}, ownerTurnId={OwnerTurnId}, " +
            "decisionAction={Action}, decisionReason={DecisionReason}, scoreMs={ScoreMs:F0}/{RequiredMs:F0}, " +
            "range=[{StartSample},{EndSample}), speakerAnchor={SpeakerAnchor}, strongUpstream={StrongUpstream}, epoch={StreamEpoch}.",
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
            decision.StreamEpoch);
        return true;
    }

    private BargeInSpeakerGateStatus GetSpeakerGateStatus(
        in DuplexAudioFrame frame,
        long generation,
        out SpeakerVerificationResult result)
    {
        result = default;
        if (!_authorization.RequiresSpeaker)
            return BargeInSpeakerGateStatus.Match;

        SpeakerEvidenceQuery query = default;
        lock (_lock)
        {
            if (!_candidate.IsCurrent(generation) ||
                frame.StreamEpoch != _candidate.StreamEpoch ||
                frame.HasDiscontinuity)
            {
                result = SpeakerVerificationResult.Unavailable(
                    0,
                    in query,
                    frame.HasDiscontinuity
                        ? SpeakerVerificationReason.TimelineGap
                        : SpeakerVerificationReason.StreamEpochMismatch);
                return BargeInSpeakerGateStatus.Rejected;
            }
            if (!_candidate.EndpointObserved)
                _candidate.ExtendEndSample(frame.EndSampleExclusive);
            if (_candidate.MainStartSample < 0)
                return BargeInSpeakerGateStatus.Pending;

            long candidateEnd = Math.Min(_candidate.EndSample, _candidate.DeadlineSample);
            bool inputComplete = _candidate.EndSample >= _candidate.DeadlineSample ||
                _candidate.EndpointObserved;
            query = new SpeakerEvidenceQuery(
                SpeakerVerificationPurpose.BargeInCommand,
                _candidate.StreamEpoch,
                _candidate.MainStartSample,
                candidateEnd,
                frame.TimestampTicks,
                InputComplete: inputComplete,
                StrongUpstreamEvidence: _candidate.StrongUpstreamEvidence,
                CandidateId: _candidate.CandidateId,
                OwnerTurnId: _candidate.OwnerTurnId);
        }

        result = _effects.EvaluateSpeaker(in query);

        if (result.Status == SpeakerVerificationStatus.Pending &&
            result.Reason == SpeakerVerificationReason.WaitingForInference)
        {
            // The first complete window exists; the worker gets the purpose's bounded timeout.
            long now = _timeProvider.GetTimestamp();
            // Provider policy is queried outside the data-plane lock: the executor is the only
            // boundary to speaker-owned code, and no foreign owner may be entered under _lock.
            TimeSpan inferenceTimeout = _effects.GetInferenceTimeout(
                SpeakerVerificationPurpose.BargeInCommand);
            lock (_lock)
            {
                if (!_candidate.IsCurrent(generation))
                {
                    result = SpeakerVerificationResult.Cancelled(
                        result.Threshold,
                        in query,
                        SpeakerVerificationReason.CancelledByNewTurn);
                    return BargeInSpeakerGateStatus.Rejected;
                }

                _candidate.MarkInferenceReady(now);
                if (_timeProvider.GetElapsedTime(_candidate.InferenceReadyTimestamp, now) >=
                    inferenceTimeout)
                {
                    result = SpeakerVerificationResult.Unavailable(
                        result.Threshold,
                        in query,
                        SpeakerVerificationReason.InferenceTimeout,
                        result.RequiredMatches);
                }
            }
        }

        lock (_lock)
        {
            if (_candidate.IsCurrent(generation))
                _candidate.SetSpeakerResult(in result);
        }

        return result.Status switch
        {
            SpeakerVerificationStatus.Match => BargeInSpeakerGateStatus.Match,
            SpeakerVerificationStatus.Pending => BargeInSpeakerGateStatus.Pending,
            _ => BargeInSpeakerGateStatus.Rejected,
        };
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
        BargeInDecision decision;
        long generation;
        lock (_lock)
        {
            if (!_candidate.Active ||
                !_candidate.Authorized ||
                _capture.ActiveTurnId != 0)
            {
                _lastCandidateOutcome = "skipped:candidate-state-mismatch";
                return;
            }
            decision = _candidate.Decision;
            generation = _candidate.Generation;
        }

        if (!TryStartCapture(
            decision,
            out IReadOnlyList<float[]> initialAudio,
            out ActiveNearEndPreRollSelection preRoll,
            out bool endpointVadCandidate,
            out bool retryable,
            out bool waitingForEndpointVadLocation))
        {
            string outcome = $"failed:{preRoll.Source}:{(retryable ? "retryable" : "terminal")}";
            bool outcomeChanged;
            lock (_lock)
            {
                outcomeChanged = !string.Equals(_lastCandidateOutcome, outcome, StringComparison.Ordinal);
                _lastCandidateOutcome = outcome;
            }
            if (waitingForEndpointVadLocation)
            {
                _logger.LogDebug(
                    "近端窗口已确认，等待主 VAD 首段音频定位: decision=[{DecisionStart},{DecisionEnd}), " +
                    "mainEnd={MainEnd}, epoch={StreamEpoch}.",
                    decision.StartSample,
                    decision.EndSampleExclusive,
                    _timeline.EndSample,
                    decision.StreamEpoch);
            }
            else if (outcomeChanged)
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

            if (!retryable)
            {
                ClearSpeakerCandidate();
                ExecutePlayback(RestoreDucking());
            }
            return;
        }

        long operationId = _effects.NextOperationId();
        bool queued = false;
        lock (_lock)
        {
            if (_journal.HasQueued)
                return;
            if (!_candidate.IsCurrent(generation))
                return;

            var lease = new NearEndConfirmationLease(
                operationId,
                _candidate.CandidateId,
                _candidate.OwnerTurnId,
                _candidate.StreamEpoch,
                preRoll,
                initialAudio,
                decision,
                endpointVadCandidate,
                _candidate.IdentityRequired,
                _candidate.IdentityRequired && _candidate.Authorized,
                _candidate.SpeakerResult);
            if (_journal.TryQueue(in lease))
            {
                _replay.ResetNextSample(preRoll.SelectedEndSample);
                _lastCandidateOutcome = "queued:confirmation";
                queued = true;
            }
        }

        if (queued)
        {
            NearEndEffectResult enqueued = _effects.Execute(
                new NearEndEffect.QueueConfirmation(operationId));
            if (!enqueued.Succeeded)
            {
                PlaybackDuckingPolicy.Command restore;
                lock (_lock)
                {
                    restore = _ducking.Restore();
                    if (_journal.TryDiscardPending(operationId))
                        _lastCandidateOutcome = "failed:confirmation-overflow";
                }
                ExecutePlayback(in restore);
            }
        }

        // The candidate has been handed to the control plane (or failed terminally); its
        // module-private state is released either way. Cached embeddings stay reusable.
        lock (_lock)
            _candidate.Clear();
    }

    private bool TryStartCapture(
        in BargeInDecision decision,
        out IReadOnlyList<float[]> initialAudio,
        out ActiveNearEndPreRollSelection selection,
        out bool endpointVadCandidate,
        out bool retryable,
        out bool waitingForEndpointVadLocation)
    {
        retryable = false;
        endpointVadCandidate = false;
        waitingForEndpointVadLocation = false;
        int preRollSamples = _options.PreRollSamples;
        int candidateAudioLimitSamples = _options.TimelineAlignedAudioLimitSamples;
        lock (_lock)
        {
            // 上一回合若被外部路径停止，最后一次同步 FeedAudio 可能仍在锁外执行。
            // 等它退出后再创建新回合，避免旧回合尾帧被状态机路由到新 ASR 会话。
            if (_capture.DirectFeedsInFlight != 0)
            {
                initialAudio = [];
                selection = ActiveNearEndPreRollSelection.Empty("in-flight-feed");
                retryable = true;
                return false;
            }

            IReadOnlyList<float[]> preRoll;
            string source;
            long selectedStreamEpoch;
            long selectedStartSample;
            long selectedEndSample;
            endpointVadCandidate = TryGetEndpointVadCandidateStartLocked(
                decision,
                out long endpointVadStreamEpoch,
                out long endpointVadStartSample,
                out bool waitForEndpointVadLocation);
            if (waitForEndpointVadLocation)
            {
                waitingForEndpointVadLocation = true;
                initialAudio = [];
                selection = ActiveNearEndPreRollSelection.Empty("main-vad-pending");
                retryable = true;
                return false;
            }

            if (endpointVadCandidate)
            {
                if (!TrySnapshotTimelineAligned(
                    endpointVadStreamEpoch,
                    endpointVadStartSample,
                    Math.Max(decision.EndSampleExclusive, _timeline.EndSample),
                    candidateAudioLimitSamples,
                    out preRoll,
                    out selectedStartSample,
                    out selectedEndSample,
                    out retryable))
                {
                    initialAudio = [];
                    selection = ActiveNearEndPreRollSelection.Empty("main-vad-timeline");
                    return false;
                }
                source = "main-vad-timeline";
                selectedStreamEpoch = endpointVadStreamEpoch;
            }
            else if (_candidate.MainStartSample >= 0)
            {
                long speakerPreRollStart = Math.Max(
                    0,
                    _candidate.MainStartSample - preRollSamples);
                if (!TrySnapshotTimelineAligned(
                    _candidate.StreamEpoch,
                    speakerPreRollStart,
                    Math.Max(decision.EndSampleExclusive, _timeline.EndSample),
                    candidateAudioLimitSamples,
                    out preRoll,
                    out selectedStartSample,
                    out selectedEndSample,
                    out retryable))
                {
                    initialAudio = [];
                    selection = ActiveNearEndPreRollSelection.Empty("main-candidate");
                    return false;
                }
                source = "main-candidate";
                selectedStreamEpoch = _candidate.StreamEpoch;
            }
            else if (NearEndDecisionRules.HasUsableTimeline(in decision))
            {
                // VAD 没有提供可靠句首时，使用裁决窗口覆盖的完整候选区间及其额外
                // pre-roll。检测与最终识别音频来自两条独立消费线程，必须以共享
                // capture timeline 为锚；main 领先时不能把 tail 右移，main 落后时
                // 则固定首次决定等待追上。
                long decisionPreRollStart = ComputeDecisionHistoryStart(decision, preRollSamples);
                if (!TrySnapshotTimelineAligned(
                    decision.StreamEpoch,
                    decisionPreRollStart,
                    decision.EndSampleExclusive,
                    candidateAudioLimitSamples,
                    out preRoll,
                    out selectedStartSample,
                    out selectedEndSample,
                    out retryable))
                {
                    initialAudio = [];
                    selection = ActiveNearEndPreRollSelection.Empty("main-timeline");
                    return false;
                }

                source = "main-timeline";
                selectedStreamEpoch = decision.StreamEpoch;
            }
            else
            {
                // 原子裁决要求概率、声学特征和识别音频共享 capture timeline。
                // 时间轴缺失时不猜测句首，也不启动打断。
                initialAudio = [];
                selection = ActiveNearEndPreRollSelection.Empty("missing-decision-timeline");
                retryable = false;
                return false;
            }

            int samples = CountSamples(preRoll);
            initialAudio = preRoll;
            selection = new ActiveNearEndPreRollSelection(
                source,
                samples,
                selectedStreamEpoch,
                decision.StartSample,
                decision.EndSampleExclusive,
                selectedStartSample,
                selectedEndSample);
            return true;
        }
    }

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
        if (ageSamples < 0 ||
            ageSamples > maxAgeSamples ||
            _endpoint.CandidateEndSample <= _endpoint.CandidateStartSample)
        {
            return false;
        }

        streamEpoch = _endpoint.CandidateStreamEpoch;
        startSample = _endpoint.CandidateStartSample;
        return true;
    }

    private long ComputeDecisionHistoryStart(
        in BargeInDecision decision,
        int preRollSamples)
    {
        int candidateMs = Math.Max(
            _options.PlaybackMaxCandidateMs,
            _options.ActiveNearEndWindowMs);
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
        if (decision.FirstAcceptedStartSample >= 0 &&
            decision.FirstAcceptedStartSample < decision.EndSampleExclusive)
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
        if (!_timeline.TryGetBounds(
            out long retainedEpoch,
            out long retainedStart,
            out long end))
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
        if (!_timeline.TrySnapshotFrom(
            streamEpoch,
            selectedStartSample,
            maxSamples,
            out audio) ||
            audio.Count == 0)
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

    private void CaptureConfirmationTail(
        in DuplexAudioFrame frame,
        bool latestNearEndSpeech)
    {
        bool overflow = false;
        lock (_lock)
        {
            if (!_journal.TryGetReplayRequest(out NearEndConfirmationLease lease) ||
                lease.Selection.StreamEpoch != frame.StreamEpoch)
            {
                return;
            }

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
                    if (tail.Generation != _timeline.Generation ||
                        tail.StreamEpoch != _timeline.StreamEpoch)
                    {
                        discard = tail.Generation < _timeline.InvalidatedGeneration;
                        fault = !discard;
                    }
                    else
                    {
                        discard = false;
                        fault = false;
                        if (!_timeline.TrySnapshotRange(
                            tail.StreamEpoch,
                            tail.StartSample,
                            tail.EndSampleExclusive,
                            out audio))
                        {
                            return;
                        }
                    }

                    if (discard || !fault)
                        _ = _replay.TryRead(out _);
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

                FeedFromRecognitionAudio(
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

    private void DrainUnfedMainAudioAfterConfirmation()
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

        FeedFromRecognitionAudio(
            audio,
            new MainAudioObservation(
                generation,
                streamEpoch,
                startSample,
                endSample,
                HasDiscontinuity: false));
    }

    // ---------------------------------------------------------------------------------------
    // Direct-feed facts and finalization
    // ---------------------------------------------------------------------------------------

    private void ProcessActiveAudio(
        long activeTurnId,
        in DuplexAudioFrame frame,
        double frameMs,
        BargeInDecision decision)
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
        return _endpoint.CaptureObserved
            ? _endpoint.CaptureSpeechActive
            : latestNearEndSpeech;
    }

    private bool IsDirectFeedActive(long activeTurnId)
    {
        lock (_lock)
            return _capture.IsActive(activeTurnId) && _capture.DirectFeed;
    }

    private void EndCapture(
        long turnId,
        TimeSpan speechDuration,
        ActiveEndDrainOutcome drainOutcome)
    {
        ActiveCaptureAudioStats stats = SnapshotActiveCaptureStats();
        if (!StopCapture(turnId, startCooldown: true))
            return;

        LogCaptureEnded(
            turnId,
            speechDuration,
            stats.PreRollSource,
            stats.PreRollSamples,
            stats.DirectFeedSamples,
            drainOutcome);

        _effects.Execute(new NearEndEffect.PublishSpeechEnded(turnId, speechDuration));
    }

    private ActiveCaptureAudioStats SnapshotActiveCaptureStats()
    {
        lock (_lock)
        {
            return new ActiveCaptureAudioStats(
                _capture.PreRollSource,
                _capture.PreRollSamples,
                _capture.DirectFeedSamples);
        }
    }

    private readonly record struct ActiveCaptureAudioStats(
        string PreRollSource,
        int PreRollSamples,
        long DirectFeedSamples);

    private void ReportDataPlaneFault(string detail)
    {
        if (Interlocked.Exchange(ref _dataPlaneFaultReported, 1) != 0)
            return;

        _effects.Execute(
            new NearEndEffect.ReportDataPlaneFault(
                new AssistantFailure(
                    "VBX-CTRL-018",
                    "capacity",
                    "active-near-end-data-plane",
                    detail)));
    }

    // ---------------------------------------------------------------------------------------
    // Diagnostics
    // ---------------------------------------------------------------------------------------

    private BargeInDecision GetRecentDecision()
        => _decisions.GetRecentDecision(
            TimeSpan.FromMilliseconds(_options.ActiveNearEndWindowMs),
            _options.MinDurationMs,
            _options.MinRms);

    private void LogCommandAuthorized(
        long generation,
        in SpeakerVerificationResult verdict,
        in BargeInDecision decision)
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
