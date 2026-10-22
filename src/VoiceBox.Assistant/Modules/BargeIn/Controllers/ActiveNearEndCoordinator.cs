using Microsoft.Extensions.Logging;
using System.Runtime.ExceptionServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers
{
    /// <summary>
    /// Collects near-end audio and confirms interruptions during playback. With identity confirmation enabled, the
    /// current candidate's speaker Match can confirm the interruption; a DTD Interrupt alone cannot.
    /// Otherwise early near-end evidence lowers playback gain until DTD confirms or the evidence expires.
    /// </summary>
    internal sealed class ActiveNearEndCoordinator :
        INearEndCapturePort,
        INearEndRenderPort,
        INearEndEndpointPort,
        INearEndConfirmationStore,
        ISpeechTimelineLocator,
        INearEndCaptureControlPort,
        INearEndResetPort
    {
        private readonly VoiceAssistantBargeInOptions _options;
        private readonly IRecognitionAudioPort _recognitionAudio;
        private readonly IActiveNearEndConversationView _conversation;
        private readonly IActiveNearEndDecisionSource _decisions;
        private readonly AssistantAudioFormat _audioFormat;
        private readonly ILogger _logger;
        private readonly IRuntimeLivenessView _runtime;
        private readonly IActiveNearEndPlaybackPort _playback;
        private readonly IActiveNearEndSignalSink _control;
        private readonly ISpeakerEvidenceProvider? _speakerEvidence;
        private readonly TimeProvider _timeProvider;
        private readonly NearEndAudioReservoir _mainAudioReservoir;
        // Endpoint policy state is owned by its own type but stays under this single lock; the
        // data plane must not gain a second lock.
        private readonly NearEndEndpointPolicy _endpoint;
        private readonly object _lock = new();
        // The recognition module owns provider admission, pending PCM, and its feed worker. The
        // near-end owner keeps only capture timeline state and uses the typed audio port.
        // Capture can advance before the control loop commits confirmation. Keep PCM in the
        // reservoir until it can be replayed; this queue stores only timeline metadata.
        private readonly BoundedMpscQueue<ConfirmationTailFrame> _confirmationTailFrames = new(256);
        private int _confirmationTailDrainActive;
        private int _dataPlaneFaultReported;
        // Capture-session ownership (confirmed turn, direct ASR feed watermark, bounded end
        // drain) lives in its own type but stays under this single data-plane lock.
        private readonly NearEndCaptureSession _capture = new();
        private double _speechDurationMs;
        private double _silenceMs;
        private string _activePreRollSource = "none";
        private int _activePreRollSamples;
        private long _cooldownUntilTicks;
        private long _mainAudioGeneration;
        private long _confirmationTailInvalidationGeneration;
        private long _mainAudioStreamEpoch;
        private long _mainAudioEndSample;
        private readonly NearEndSpeakerCandidateState _speakerCandidate = new();
        private string _lastCandidateOutcome = "none";
        private long _commandAuthorizedCount;
        private long _commandDeniedCount;
        private long _duckedOwnerTurnId;
        private long _duckedStreamEpoch;
        private long _lastDuckingSpeechEndSample;
        private ActiveNearEndConfirmationRequest? _pendingConfirmation;
        private ActiveNearEndConfirmationRequest? _confirmationTailCapture;
        private long _confirmationTailNextSample;
        private readonly record struct ConfirmationTailFrame(
            long Generation,
            long StreamEpoch,
            long StartSample,
            long EndSampleExclusive,
            bool HasDiscontinuity,
            bool LatestSpeechActive,
            double DurationMs);

        public ActiveNearEndCoordinator(
            VoiceAssistantBargeInOptions options,
            IRecognitionAudioPort recognitionAudio,
            IActiveNearEndConversationView conversation,
            IActiveNearEndDecisionSource decisions,
            AssistantAudioFormat audioFormat,
            ILogger logger,
            IRuntimeLivenessView runtime,
            IActiveNearEndPlaybackPort playback,
            IActiveNearEndSignalSink control,
            ISpeakerEvidenceProvider? speakerEvidence = null,
            TimeProvider? timeProvider = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _recognitionAudio = recognitionAudio
                ?? throw new ArgumentNullException(nameof(recognitionAudio));
            _conversation = conversation ?? throw new ArgumentNullException(nameof(conversation));
            _decisions = decisions ?? throw new ArgumentNullException(nameof(decisions));
            _audioFormat = audioFormat ?? throw new ArgumentNullException(nameof(audioFormat));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
            _playback = playback ?? throw new ArgumentNullException(nameof(playback));
            _control = control ?? throw new ArgumentNullException(nameof(control));
            // Non-null exactly when online identity confirmation is enabled.
            _speakerEvidence = speakerEvidence;
            _timeProvider = timeProvider ?? TimeProvider.System;

            int reservoirSamples = ComputeReservoirSamples();
            _mainAudioReservoir = new NearEndAudioReservoir(reservoirSamples);
            _endpoint = new NearEndEndpointPolicy(_mainAudioReservoir);
        }

        /// <summary>KWS enabled: playback interruptions require the current candidate's speaker Match.</summary>
        public bool IsIdentityRequired => _speakerEvidence is not null;

        public long CommandAuthorizedCount => Volatile.Read(ref _commandAuthorizedCount);

        public long CommandDeniedCount => Volatile.Read(ref _commandDeniedCount);

        public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot()
        {
            lock (_lock)
            {
                return new ActiveNearEndDiagnosticSnapshot(
                    _speakerCandidate.Active,
                    _speakerCandidate.Authorized,
                    _speakerCandidate.Decision.EndSampleExclusive,
                    _mainAudioEndSample,
                    _capture.ActiveTurnId,
                    _lastCandidateOutcome,
                    _pendingConfirmation is not null || _confirmationTailCapture is not null,
                    _speakerCandidate.CandidateId.Value,
                    _speakerCandidate.DtdInterruptObserved);
            }
        }

        public ActiveNearEndProcessingState GetProcessingState()
        {
            lock (_lock)
            {
                return new ActiveNearEndProcessingState(
                    _capture.ActiveTurnId != 0,
                    _speakerCandidate.Active || _endpoint.CandidateActive);
            }
        }

        public void ObserveEndpointSpeechStarted()
        {
            lock (_lock)
            {
                // A fresh VAD onset definitively separates this utterance from any endpoint
                // that was retained while the prior DTD candidate was still forming.
                ClearUnownedEndpointVadObservationLocked();
                if (_capture.ActiveTurnId != 0 && _capture.DirectFeed)
                {
                    if (_endpoint.Observed)
                        return;
                    _endpoint.Observed = true;
                    _endpoint.SpeechActive = true;
                    return;
                }

                ClearEndpointVadCandidateLocked();
                _endpoint.BeginCandidate(_mainAudioGeneration, _mainAudioStreamEpoch);
            }
        }

        public void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty)
                return;

            lock (_lock)
            {
                if (_capture.ActiveTurnId != 0 && _capture.DirectFeed)
                {
                    _endpoint.Observed = true;
                    _endpoint.SpeechActive = true;
                    return;
                }

                // IVoiceActivityDetectService promises that OnSpeaking contains audio from
                // Predict. Locate that exact chunk in the main-AEC reservoir instead of
                // estimating its start from the later fused decision window.
                if (!_endpoint.CandidateActive)
                    _endpoint.BeginCandidate(_mainAudioGeneration, _mainAudioStreamEpoch);

                if (_endpoint.CandidateGeneration != _mainAudioGeneration ||
                    !TryLocateEndpointVadAudioLocked(
                        audio,
                        out long streamEpoch,
                        out long startSample,
                        out long endSample) ||
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
                _endpoint.RecordUnowned(
                    _mainAudioGeneration,
                    _mainAudioStreamEpoch,
                    _mainAudioEndSample);
                // SpeechEnded arrived while the speaker verdict is still pending: the candidate's
                // input is complete, and the bounded wait for the verdict continues on its own. The
                // candidate interval is frozen at the endpoint so trailing silence never displaces
                // the utterance's own windows from the evidence.
                if (_speakerCandidate.Active && !_speakerCandidate.Authorized &&
                    _speakerCandidate.StreamEpoch == _mainAudioStreamEpoch &&
                    _mainAudioEndSample > _speakerCandidate.MainStartSample)
                {
                    _speakerCandidate.EndpointObserved = true;
                    _speakerCandidate.EndSample = Math.Min(
                        Math.Max(_speakerCandidate.EndSample, _mainAudioEndSample),
                        _speakerCandidate.DeadlineSample);
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
                ClearEndpointVadCandidateLocked();
                if (_capture.ActiveTurnId != 0 && _capture.DirectFeed)
                {
                    _endpoint.Observed = true;
                    _endpoint.SpeechActive = false;
                    bool dtdTailPredatesCaptureCommit = captureCommittedAfterEndpoint &&
                        _endpoint.UnownedObserved &&
                        _endpoint.UnownedGeneration == _mainAudioGeneration &&
                        _endpoint.UnownedStreamEpoch == _mainAudioStreamEpoch &&
                        _mainAudioEndSample > _endpoint.UnownedEndSample;
                    if (dtdTailPredatesCaptureCommit)
                    {
                        activeTurnId = _capture.ActiveTurnId;
                        // The control plane committed the capture after both the VAD endpoint
                        // and a later DTD frame. Those tail frames cannot be replayed as future
                        // callbacks, so drain to the known main-AEC boundary now.
                        BeginEndDrainLocked(
                            activeTurnId,
                            _mainAudioStreamEpoch,
                            _mainAudioEndSample,
                            hasDiscontinuity: false);
                        finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                    }

                    ClearUnownedEndpointVadObservationLocked();
                }
                else if (_endpoint.UnownedObserved)
                {
                    // This callback is the control-plane acknowledgement of an endpoint
                    // recorded by the capture callback. Only an acknowledgement that runs
                    // before capture commits can prove that no later endpoint observation is
                    // still queued behind the confirmation.
                    _endpoint.UnownedProcessed = true;
                }
                // A completed endpoint can precede the confirmation envelope, or even the
                // DTD decision that produces it. Keep its timeline fact until either a fresh
                // VAD onset, a timeline reset, or an accepted capture consumes it.
            }

            if (finalize)
                EndCapture(activeTurnId, speechDuration, drainOutcome);
        }

        private void ClearEndpointVadCandidateLocked() => _endpoint.ClearCandidate();

        private void ClearUnownedEndpointVadObservationLocked() => _endpoint.ClearUnowned();

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
                _endpoint.CandidateGeneration != _mainAudioGeneration ||
                _endpoint.CandidateStreamEpoch == 0 ||
                _endpoint.CandidateStreamEpoch != _mainAudioStreamEpoch ||
                !HasUsableTimeline(decision) ||
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
            if (ageSamples < 0 || ageSamples > maxAgeSamples ||
                _endpoint.CandidateEndSample <= _endpoint.CandidateStartSample)
            {
                return false;
            }

            streamEpoch = _endpoint.CandidateStreamEpoch;
            startSample = _endpoint.CandidateStartSample;
            return true;
        }

        private bool TryLocateEndpointVadAudioLocked(
            ReadOnlySpan<float> audio,
            out long streamEpoch,
            out long startSample,
            out long endSample)
            => _endpoint.TryLocate(audio, out streamEpoch, out startSample, out endSample);

        public void ObserveDuplexAudio(
            in DuplexAudioFrame frame,
            bool playbackStartupGuardActive,
            bool latestNearEndSpeech = false)
        {
            if (playbackStartupGuardActive)
            {
                RestorePlaybackDucking();
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
            MainAudioObservation observation;
            lock (_lock)
            {
                CaptureRange sourceRange = frame.SourceRange;
                bool resetExisting = _mainAudioReservoir.Add(
                    audio,
                    in sourceRange,
                    frame.HasDiscontinuity);
                bool discontinuity = frame.HasDiscontinuity || resetExisting;
                if (discontinuity)
                {
                    RestorePlaybackDuckingLocked();
                    _mainAudioGeneration = checked(_mainAudioGeneration + 1);
                    _confirmationTailInvalidationGeneration = _mainAudioGeneration;
                    ClearSpeakerCandidateLocked();
                    ClearEndpointVadCandidateLocked();
                    if (_capture.ActiveTurnId == 0)
                        _capture.DirectFeedNextSample = frame.SourceRange.EndSampleExclusive;
                }
                _mainAudioStreamEpoch = frame.SourceRange.StreamEpoch;
                _mainAudioEndSample = frame.SourceRange.EndSampleExclusive;
                observation = new MainAudioObservation(
                    _mainAudioGeneration,
                    frame.SourceRange.StreamEpoch,
                    frame.SourceRange.StartSample,
                    frame.SourceRange.EndSampleExclusive,
                    discontinuity);
                if (!discontinuity &&
                    _capture.ActiveTurnId == 0 &&
                    _speakerCandidate.Active &&
                    _speakerCandidate.Authorized)
                {
                    retryAuthorizedCandidate = true;
                }
            }

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

            if (Volatile.Read(ref _confirmationTailDrainActive) == 0)
                DrainConfirmationTail();

            int offset = 0;
            int length = 0;
            long reservedFeedStart = 0;
            long reservedFeedEnd = 0;
            long turnId = 0;
            bool feedReserved = false;
            bool finalize = false;
            TimeSpan speechDuration = TimeSpan.Zero;
            ActiveEndDrainOutcome drainOutcome = default;
            lock (_lock)
            {
                if (_capture.ActiveTurnId == 0 ||
                    !_capture.DirectFeed ||
                    observation.Generation != _mainAudioGeneration ||
                    observation.StreamEpoch != _mainAudioStreamEpoch)
                {
                    return;
                }

                turnId = _capture.ActiveTurnId;
                if (observation.HasDiscontinuity)
                {
                    // 处理链报告时间轴断裂时，目标点之前的缺口已无法恢复。当前帧位于缺口
                    // 之后，不能假装它与已喂音频连续；等待中的结束排空必须立即以 partial
                    // 收尾，非结束态则从下一帧重新建立直送起点。
                    if (_capture.EndDrainPending)
                    {
                        ForceEndDrainLocked("capture-discontinuity");
                    }
                    else
                    {
                        _capture.DirectFeedNextSample = Math.Max(
                            _capture.DirectFeedNextSample,
                            observation.EndSample);
                    }
                    finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                }
                else if (observation.EndSample > _capture.DirectFeedNextSample)
                {
                    long feedStart = Math.Max(observation.StartSample, _capture.DirectFeedNextSample);
                    long feedEnd = observation.EndSample;
                    if (_capture.EndDrainPending)
                    {
                        if (observation.StreamEpoch != _capture.EndDrainStreamEpoch)
                        {
                            ForceEndDrainLocked("stream-epoch-changed");
                            feedEnd = feedStart;
                        }
                        else
                        {
                            feedEnd = Math.Min(feedEnd, _capture.EndDrainTargetSample);
                        }
                    }

                    if (feedEnd > feedStart)
                    {
                        offset = checked((int)(feedStart - observation.StartSample));
                        length = checked((int)(feedEnd - feedStart));
                        reservedFeedStart = feedStart;
                        reservedFeedEnd = feedEnd;
                        _capture.DirectFeedNextSample = feedEnd;
                        _capture.DirectFeedsInFlight++;
                        feedReserved = true;
                    }

                    if (_capture.EndDrainPending &&
                        _capture.DirectFeedNextSample >= _capture.EndDrainTargetSample &&
                        !_capture.EndDrainForced)
                    {
                        _capture.MarkEndDrainAligned();
                    }

                    if (!feedReserved)
                        finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                }
                else
                {
                    finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                }
            }

            if (feedReserved)
            {
                Exception? feedError = null;
                try
                {
                    FeedAudio(turnId, audio.Slice(offset, length));
                }
                catch (Exception ex)
                {
                    feedError = ex;
                }

                lock (_lock)
                {
                    _capture.DirectFeedsInFlight--;
                    if (feedError is not null &&
                        _capture.ActiveTurnId == turnId &&
                        _capture.DirectFeedNextSample == reservedFeedEnd)
                    {
                        _capture.DirectFeedNextSample = reservedFeedStart;
                        if (_capture.EndDrainPending)
                            ForceEndDrainLocked("asr-feed-failed");
                    }
                    finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                }

                if (finalize)
                    EndCapture(turnId, speechDuration, drainOutcome);

                if (feedError is not null)
                    ExceptionDispatchInfo.Capture(feedError).Throw();

                return;
            }

            if (finalize)
                EndCapture(turnId, speechDuration, drainOutcome);
        }

        public bool TryLocate(
            ReadOnlySpan<float> audio,
            out long streamEpoch,
            out long startSample,
            out long endSample)
        {
            lock (_lock)
                return _mainAudioReservoir.TryLocate(audio, out streamEpoch, out startSample, out endSample);
        }

        public bool StopCapture(long turnId, bool startCooldown)
        {
            lock (_lock)
            {
                if (_capture.ActiveTurnId != turnId)
                    return false;

                _capture.ActiveTurnId = 0;
                _capture.DirectFeed = false;
                _speechDurationMs = 0;
                _silenceMs = 0;
                _activePreRollSource = "none";
                _activePreRollSamples = 0;
                _capture.DirectFeedSamples = 0;
                _capture.DirectFeedNextSample = 0;
                _endpoint.Observed = false;
                _endpoint.SpeechActive = false;
                _pendingConfirmation = null;
                _confirmationTailCapture = null;
                ClearConfirmationTailFramesLocked();
                ClearEndpointVadCandidateLocked();
                ClearUnownedEndpointVadObservationLocked();
                ClearEndDrainLocked();
                ClearSpeakerCandidateLocked();
                if (startCooldown && _options.ActiveNearEndCooldownMs > 0)
                {
                    Volatile.Write(
                        ref _cooldownUntilTicks,
                        Environment.TickCount64 + _options.ActiveNearEndCooldownMs);
                }
            }
            return true;
        }

        public void Reset()
        {
            lock (_lock)
            {
                RestorePlaybackDuckingLocked();
                Volatile.Write(ref _dataPlaneFaultReported, 0);
                _recognitionAudio.ResetAudio(_capture.ActiveTurnId);
                _capture.ActiveTurnId = 0;
                _capture.DirectFeed = false;
                _speechDurationMs = 0;
                _silenceMs = 0;
                _activePreRollSource = "none";
                _activePreRollSamples = 0;
                _capture.DirectFeedSamples = 0;
                _capture.DirectFeedNextSample = 0;
                _endpoint.Observed = false;
                _endpoint.SpeechActive = false;
                _pendingConfirmation = null;
                _confirmationTailCapture = null;
                ClearConfirmationTailFramesLocked();
                ClearEndpointVadCandidateLocked();
                ClearUnownedEndpointVadObservationLocked();
                ClearEndDrainLocked();
                _cooldownUntilTicks = 0;
                ClearSpeakerCandidateLocked();
            }
            ClearAudioReservoirs();
        }

        public void ClearAudioReservoirs()
        {
            ClearSpeakerCandidate();
            lock (_lock)
            {
                RestorePlaybackDuckingLocked();
                _mainAudioReservoir.Clear();
                _mainAudioEndSample = 0;
                _mainAudioStreamEpoch = 0;
                _mainAudioGeneration = checked(_mainAudioGeneration + 1);
                _capture.DirectFeedNextSample = 0;
                ClearConfirmationTailFramesLocked();
                ClearEndpointVadCandidateLocked();
                ClearUnownedEndpointVadObservationLocked();
            }
        }

        private void Process(in DuplexAudioFrame frame)
        {
            if (!_options.Enable || _runtime.IsStopping)
            {
                RestorePlaybackDucking();
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
            // 各条件与本 coordinator 后续窗口复核一致, 提前判定只省去无谓计算, 不改变结果;
            // 尤其 state != Speaking 时旧实现会在 Idle 说话期间逐帧做整段预滚快照再被状态机拒绝, 属纯浪费。
            if (Environment.TickCount64 < Volatile.Read(ref _cooldownUntilTicks))
                return;
            if (!_conversation.IsSpeakingWithoutPendingCandidate)
            {
                RestorePlaybackDucking();
                SetCandidateClearOutcome("cleared:not-speaking");
                ClearSpeakerCandidate();
                return;
            }

            // Speaking 在 LLM 首个文本增量时就会建立，可能早于首个 TTS 音频入队。
            // 用播放生命周期阻止那段空窗触发；不能要求当前声学帧 RenderActive，
            // 否则用户在 TTS 句间停顿或低能量段说话时会丢失已满足的近端判决。
            if (!_playback.IsPlaybackActive)
            {
                RestorePlaybackDucking();
                SetCandidateClearOutcome("cleared:playback-inactive");
                ClearSpeakerCandidate();
                return;
            }

            BargeInDecision decision = GetRecentDecision();
            if (_speakerEvidence is null)
            {
                // Confirmation off: authorization is NotRequired. The acoustic Interrupt is the
                // whole decision and no speaker is ever queried.
                bool timelineCandidateActive;
                lock (_lock)
                    timelineCandidateActive = _speakerCandidate.Active;
                UpdatePlaybackDucking(in frame, in decision, timelineCandidateActive);
                if (!timelineCandidateActive)
                {
                    if (decision.Action != BargeInAction.Interrupt)
                        return;
                    BeginTimelineCandidate(decision);
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
                candidateActive = _speakerCandidate.Active;
                candidateAuthorized = _speakerCandidate.Authorized;
                generation = _speakerCandidate.Generation;
                if (candidateActive)
                    _speakerCandidate.ObserveDecision(in decision, IsStrongNearEndDecision(in decision));
            }

            if (!candidateActive)
            {
                if (!HasNearEndSpeechEvidence(in decision))
                    return;
                if (!BeginSpeakerCandidate(decision, in frame, out generation))
                    return;
            }

            if (!candidateAuthorized)
            {
                BargeInSpeakerGateStatus speakerStatus = GetSpeakerGateStatus(in frame, generation, out SpeakerVerificationResult verdict);
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
                _logger.LogDebug(
                    "command-authorized: candidate={CandidateId}, ownerTurnId={OwnerTurnId}, reason={Reason}, " +
                    "score={Score:F4}, threshold={Threshold:F4}, matched={Matched}/{Required}, dtdInterrupt={DtdInterrupt}, " +
                    "range=[{StartSample},{EndSample}), epoch={StreamEpoch}.",
                    verdict.CandidateId,
                    _speakerCandidate.OwnerTurnId,
                    verdict.Reason,
                    verdict.Score,
                    verdict.Threshold,
                    verdict.MatchedWindows,
                    verdict.RequiredMatches,
                    decision.Action == BargeInAction.Interrupt,
                    verdict.EvidenceStartSample,
                    verdict.EvidenceEndSampleExclusive,
                    verdict.StreamEpoch);
            }

            TryStartAuthorizedCandidate();
        }

        private void UpdatePlaybackDucking(
            in DuplexAudioFrame frame,
            in BargeInDecision decision,
            bool confirmationStarted)
        {
            lock (_lock)
            {
                long ownerTurnId = _conversation.CurrentTurnId;
                if (_duckedOwnerTurnId != 0 &&
                    (_duckedOwnerTurnId != ownerTurnId || _duckedStreamEpoch != frame.StreamEpoch))
                    RestorePlaybackDuckingLocked();

                if (frame.HasDiscontinuity || !HasUsableTimeline(in decision) ||
                    decision.StreamEpoch != frame.StreamEpoch || _options.PlaybackDuckGain >= 1)
                {
                    RestorePlaybackDuckingLocked();
                    return;
                }

                // Once DTD confirms, keep the reduced gain until the owner stops and clears its
                // audio. Do not briefly restore the old reply while the control loop catches up.
                if (confirmationStarted || _pendingConfirmation is not null || decision.Action == BargeInAction.Interrupt)
                    return;

                long releaseSamples = AssistantAudioFormat.SamplesFromMilliseconds(
                    _options.PlaybackDuckReleaseMs,
                    checked((int)frame.SampleRate),
                    checked((int)frame.Channels));
                bool recentSpeech = HasNearEndSpeechEvidence(in decision) &&
                    frame.EndSampleExclusive - decision.EndSampleExclusive <= releaseSamples;
                if (recentSpeech)
                {
                    if (_duckedOwnerTurnId == 0 &&
                        _playback.TryDuckPlayback(ownerTurnId, _options.PlaybackDuckGain, frame.PlaybackGain))
                    {
                        _duckedOwnerTurnId = ownerTurnId;
                        _duckedStreamEpoch = frame.StreamEpoch;
                    }
                    if (_duckedOwnerTurnId != 0)
                        _lastDuckingSpeechEndSample = Math.Max(_lastDuckingSpeechEndSample, decision.EndSampleExclusive);
                }
                else if (_duckedOwnerTurnId != 0 &&
                    frame.EndSampleExclusive - _lastDuckingSpeechEndSample >= releaseSamples)
                {
                    RestorePlaybackDuckingLocked();
                }
            }
        }

        private void RestorePlaybackDucking()
        {
            lock (_lock)
                RestorePlaybackDuckingLocked();
        }

        private void RestorePlaybackDuckingLocked()
        {
            if (_duckedOwnerTurnId == 0)
                return;
            _playback.TryRestorePlayback(_duckedOwnerTurnId);
            _duckedOwnerTurnId = 0;
            _duckedStreamEpoch = 0;
            _lastDuckingSpeechEndSample = 0;
        }

        /// <summary>
        /// Valid near-end speech on the shared timeline: the DTD accepted the latest frame and has
        /// causal near-end evidence. This is the candidate trigger, not an authorization.
        /// </summary>
        private static bool HasNearEndSpeechEvidence(in BargeInDecision decision)
            => HasUsableTimeline(in decision) &&
               decision.LatestFrameAccepted &&
               decision.AcceptedFrameCount > 0 &&
               decision.AcceptedDurationMs > 0;

        private void DenyCandidate(long generation, in SpeakerVerificationResult verdict, in BargeInDecision decision)
        {
            string outcome = verdict.Status switch
            {
                SpeakerVerificationStatus.Mismatch => "command-denied-mismatch",
                SpeakerVerificationStatus.InsufficientEvidence => "command-denied-insufficient-evidence",
                _ => $"command-denied-speaker-unavailable:{verdict.Reason}",
            };
            SetCandidateClearOutcome(outcome);
            Interlocked.Increment(ref _commandDeniedCount);
            _logger.LogDebug(
                "{Outcome}: candidate={CandidateId}, ownerTurnId={OwnerTurnId}, status={Status}, reason={Reason}, " +
                "score={Score:F4}, threshold={Threshold:F4}, observed={Observed}, matched={Matched}/{Required}, " +
                "dtdInterrupt={DtdInterrupt}, decisionReason={DecisionReason}, range=[{StartSample},{EndSample}), epoch={StreamEpoch}. 原播放继续。",
                outcome,
                verdict.CandidateId,
                _speakerCandidate.OwnerTurnId,
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

        private void CaptureConfirmationTail(
            in DuplexAudioFrame frame,
            bool latestNearEndSpeech)
        {
            ActiveNearEndConfirmationRequest? confirmation;
            ActiveNearEndConfirmationRequest request;
            long generation;
            long tailStart;
            lock (_lock)
            {
                confirmation = _pendingConfirmation ?? _confirmationTailCapture;
                generation = _mainAudioGeneration;
                if (confirmation is not { } pending ||
                    pending.Selection.StreamEpoch != frame.StreamEpoch)
                {
                    return;
                }
                request = pending;

                // The confirmation can be formed only after several DTD frames have already
                // advanced beyond its immutable selection boundary. The first replay must
                // therefore start at that boundary rather than at the current frame start.
                tailStart = Math.Max(
                    request.Selection.SelectedEndSample,
                    _confirmationTailNextSample);
            }

            if (frame.EndSampleExclusive <= tailStart)
                return;

            var tail = new ConfirmationTailFrame(
                generation,
                frame.StreamEpoch,
                tailStart,
                frame.EndSampleExclusive,
                frame.HasDiscontinuity,
                IsLatestSpeechActive(latestNearEndSpeech),
                ComputeFrameDurationMs(in frame) *
                    (frame.EndSampleExclusive - tailStart) / Math.Max(1, frame.FrameCount));
            if (!_confirmationTailFrames.TryWrite(tail))
            {
                ReportDataPlaneFault("confirmation-tail-overflow");
                return;
            }

            lock (_lock)
            {
                if ((_pendingConfirmation ?? _confirmationTailCapture)?.OperationId == request.OperationId)
                {
                    _confirmationTailNextSample = Math.Max(
                        _confirmationTailNextSample,
                        frame.EndSampleExclusive);
                }
            }
        }

        private void DrainConfirmationTail()
        {
            if (Interlocked.CompareExchange(ref _confirmationTailDrainActive, 1, 0) != 0)
                return;

            try
            {
                while (_confirmationTailFrames.TryPeek(out ConfirmationTailFrame tail))
                {
                    long activeTurnId;
                    bool invalidatedByCaptureDiscontinuity;
                    lock (_lock)
                    {
                        activeTurnId = _capture.ActiveTurnId;
                        if (activeTurnId == 0 || !_capture.DirectFeed)
                            return;
                        if (tail.Generation != _mainAudioGeneration ||
                            tail.StreamEpoch != _mainAudioStreamEpoch)
                        {
                            invalidatedByCaptureDiscontinuity =
                                tail.Generation < _confirmationTailInvalidationGeneration;
                            if (!invalidatedByCaptureDiscontinuity)
                            {
                                ReportDataPlaneFault("confirmation-tail-timeline-mismatch");
                                return;
                            }
                        }
                        else
                        {
                            invalidatedByCaptureDiscontinuity = false;
                        }
                    }

                    if (invalidatedByCaptureDiscontinuity)
                    {
                        // ObserveRecognitionAudio already retired this range from the reservoir.
                        // Only this drain owns queue reads, so discard the obsolete metadata here
                        // rather than treating the declared capture reset as a data-plane fault.
                        _ = _confirmationTailFrames.TryRead(out _);
                        continue;
                    }

                    if (!_mainAudioReservoir.TrySnapshotRange(
                            tail.StreamEpoch,
                            tail.StartSample,
                            tail.EndSampleExclusive,
                            out float[] audio))
                    {
                        return;
                    }

                    // DrainConfirmationTail owns the only consumer lease, so the item
                    // observed by TryPeek is necessarily the item removed here.
                    _ = _confirmationTailFrames.TryRead(out _);

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
                Volatile.Write(ref _confirmationTailDrainActive, 0);
            }
        }

        private void DrainUnfedMainAudioAfterConfirmation()
        {
            long generation;
            long streamEpoch;
            long startSample;
            long endSample;
            lock (_lock)
            {
                if (_capture.ActiveTurnId == 0 || !_capture.DirectFeed)
                    return;
                startSample = _capture.DirectFeedNextSample;
                endSample = _mainAudioEndSample;
                if (endSample <= startSample)
                    return;
                generation = _mainAudioGeneration;
                streamEpoch = _mainAudioStreamEpoch;
            }

            if (!_mainAudioReservoir.TrySnapshotRange(
                    streamEpoch,
                    startSample,
                    endSample,
                    out float[] audio))
            {
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
            bool finalize = false;
            TimeSpan speechDuration = TimeSpan.Zero;
            ActiveEndDrainOutcome drainOutcome = default;
            lock (_lock)
            {
                if (_capture.ActiveTurnId != activeTurnId || !_capture.DirectFeed)
                    return;

                if (_capture.EndDrainPending)
                {
                    if (hasDiscontinuity || streamEpoch != _capture.EndDrainStreamEpoch)
                        ForceEndDrainLocked("capture-discontinuity");
                    else if (endSampleExclusive >= _capture.EndDrainDeadlineSample)
                        ForceEndDrainLocked("drain-timeout");

                    finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                }
                else
                {
                    _speechDurationMs += frameMs;
                    // 结束检测必须看最新帧，而不是整个滑窗里是否还残留任意已接受帧。
                    // 否则用户停说后，窗口内一个旧语音帧或偶发噪声尖峰就会持续把
                    // 静音计时归零；在真实设备上可把约 1s 的指令拖成数十秒 ASR 音频。
                    if (latestSpeechActive)
                        _silenceMs = 0;
                    else
                        _silenceMs += frameMs;

                    if (_silenceMs >= _options.ActiveNearEndEndSilenceMs)
                    {
                        BeginEndDrainLocked(
                            activeTurnId,
                            streamEpoch,
                            endSampleExclusive,
                            hasDiscontinuity);
                        finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
                    }
                }
            }

            if (finalize)
                EndCapture(activeTurnId, speechDuration, drainOutcome);
        }

        private bool IsLatestSpeechActive(in BargeInDecision decision)
            => IsLatestSpeechActive(decision.LatestFrameAccepted);

        private bool IsLatestSpeechActive(bool latestNearEndSpeech)
        {
            // DTD remains the acoustic authority. Once a candidate is authorized, the main
            // recognition VAD may extend its endpoint so low-energy syllables are not truncated.
            lock (_lock)
            {
                if (_endpoint.Observed)
                    return _endpoint.SpeechActive;
            }

            return latestNearEndSpeech;
        }

        private bool IsDirectFeedActive(long activeTurnId)
        {
            lock (_lock)
                return _capture.ActiveTurnId == activeTurnId && _capture.DirectFeed;
        }

        private void BeginEndDrainLocked(
            long activeTurnId,
            long streamEpoch,
            long endSampleExclusive,
            bool hasDiscontinuity)
        {
            if (_capture.ActiveTurnId != activeTurnId || !_capture.DirectFeed || _capture.EndDrainPending)
                return;

            long timeoutSamples = _audioFormat.SamplesFromMillisecondsToInt32(
                _options.ActiveNearEndAudioDrainTimeoutMs,
                _audioFormat.AudioSampleRate,
                _audioFormat.AudioChannels);
            _capture.BeginEndDrain(
                streamEpoch,
                endSampleExclusive,
                TimeSpan.FromMilliseconds(_speechDurationMs),
                timeoutSamples);

            if (hasDiscontinuity || streamEpoch <= 0 || endSampleExclusive < 0)
                ForceEndDrainLocked("invalid-capture-timeline");
            else if (_capture.DirectFeedNextSample >= _capture.EndDrainTargetSample)
                _capture.MarkEndDrainAligned();
        }

        private void ForceEndDrainLocked(string reason) => _capture.ForceEndDrain(reason);

        private bool TryClaimEndFinalizationLocked(
            out TimeSpan speechDuration,
            out ActiveEndDrainOutcome outcome)
            => _capture.TryClaimEndFinalization(out speechDuration, out outcome);

        private void ClearEndDrainLocked() => _capture.ClearEndDrain();

        private static long SaturatingAdd(long left, long right)
            => NearEndCaptureSession.SaturatingAdd(left, right);

        private BargeInDecision GetRecentDecision()
            => _decisions.GetRecentDecision(
                TimeSpan.FromMilliseconds(_options.ActiveNearEndWindowMs),
                _options.MinDurationMs,
                _options.MinRms);

        /// <summary>
        /// Opens the KWS-enabled command candidate on the shared capture timeline. The anchor is the
        /// first causally accepted near-end sample so speaker windows cover the actual onset; the
        /// speaker query starts on the next frame without waiting for a DTD Interrupt.
        /// </summary>
        private bool BeginSpeakerCandidate(
            BargeInDecision decision,
            in DuplexAudioFrame frame,
            out long generation)
        {
            generation = 0;
            if (_speakerEvidence is null)
                return false;
            if (decision.HasDiscontinuity || decision.StreamEpoch <= 0 ||
                decision.StreamEpoch != frame.StreamEpoch ||
                decision.StartSample < 0 || decision.EndSampleExclusive <= decision.StartSample)
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

            SpeakerCandidateId candidateId = _speakerEvidence.NewCandidateId();
            long ownerTurnId = _conversation.CurrentTurnId;
            lock (_lock)
            {
                if (_speakerCandidate.Active)
                {
                    generation = _speakerCandidate.Generation;
                    return true;
                }

                long anchorSample = decision.FirstAcceptedStartSample >= decision.StartSample &&
                    decision.FirstAcceptedStartSample < decision.EndSampleExclusive
                        ? decision.FirstAcceptedStartSample
                        : decision.StartSample;
                long maxCandidateSamples = AssistantAudioFormat.SamplesFromMilliseconds(
                    _options.PlaybackMaxCandidateMs,
                    _audioFormat.AudioSampleRate,
                    _audioFormat.AudioChannels);
                _speakerCandidate.Begin(
                    in decision,
                    candidateId,
                    ownerTurnId,
                    anchorSample,
                    frame.EndSampleExclusive,
                    SaturatingAdd(anchorSample, maxCandidateSamples),
                    IsStrongNearEndDecision(in decision));
                _lastCandidateOutcome = "awaiting-speaker";
                generation = _speakerCandidate.Generation;
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
                _speakerCandidate.MainStartSample,
                _speakerCandidate.StrongUpstreamEvidence,
                decision.StreamEpoch);
            return true;
        }

        private BargeInSpeakerGateStatus GetSpeakerGateStatus(
            in DuplexAudioFrame frame,
            long generation,
            out SpeakerVerificationResult result)
        {
            result = default;
            if (_speakerEvidence is null)
                return BargeInSpeakerGateStatus.Match;

            SpeakerEvidenceQuery query = default;
            lock (_lock)
            {
                if (!_speakerCandidate.IsCurrent(generation) ||
                    frame.StreamEpoch != _speakerCandidate.StreamEpoch ||
                    frame.HasDiscontinuity)
                {
                    result = SpeakerVerificationResult.Unavailable(
                        0,
                        in query,
                        frame.HasDiscontinuity ? SpeakerVerificationReason.TimelineGap : SpeakerVerificationReason.StreamEpochMismatch);
                    return BargeInSpeakerGateStatus.Rejected;
                }
                if (!_speakerCandidate.EndpointObserved)
                {
                    _speakerCandidate.EndSample = Math.Max(
                        _speakerCandidate.EndSample,
                        frame.EndSampleExclusive);
                }
                if (_speakerCandidate.MainStartSample < 0)
                    return BargeInSpeakerGateStatus.Pending;
                long candidateEnd = Math.Min(
                    _speakerCandidate.EndSample,
                    _speakerCandidate.DeadlineSample);
                bool inputComplete = _speakerCandidate.EndSample >= _speakerCandidate.DeadlineSample ||
                    _speakerCandidate.EndpointObserved;
                query = new SpeakerEvidenceQuery(
                    SpeakerVerificationPurpose.BargeInCommand,
                    _speakerCandidate.StreamEpoch,
                    _speakerCandidate.MainStartSample,
                    candidateEnd,
                    frame.TimestampTicks,
                    InputComplete: inputComplete,
                    StrongUpstreamEvidence: _speakerCandidate.StrongUpstreamEvidence,
                    CandidateId: _speakerCandidate.CandidateId,
                    OwnerTurnId: _speakerCandidate.OwnerTurnId);
            }

            result = _speakerEvidence.Evaluate(in query);

            if (result.Status == SpeakerVerificationStatus.Pending &&
                result.Reason == SpeakerVerificationReason.WaitingForInference)
            {
                // The first complete window exists; the worker gets the purpose's bounded timeout.
                long now = _timeProvider.GetTimestamp();
                lock (_lock)
                {
                    if (!_speakerCandidate.IsCurrent(generation))
                    {
                        result = SpeakerVerificationResult.Cancelled(result.Threshold, in query, SpeakerVerificationReason.CancelledByNewTurn);
                        return BargeInSpeakerGateStatus.Rejected;
                    }
                    if (_speakerCandidate.InferenceReadyTimestamp < 0)
                        _speakerCandidate.InferenceReadyTimestamp = now;
                    if (_timeProvider.GetElapsedTime(_speakerCandidate.InferenceReadyTimestamp, now) >=
                        _speakerEvidence.GetInferenceTimeout(SpeakerVerificationPurpose.BargeInCommand))
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
                if (_speakerCandidate.IsCurrent(generation))
                    _speakerCandidate.SpeakerResult = result;
            }

            return result.Status switch
            {
                SpeakerVerificationStatus.Match => BargeInSpeakerGateStatus.Match,
                SpeakerVerificationStatus.Pending => BargeInSpeakerGateStatus.Pending,
                _ => BargeInSpeakerGateStatus.Rejected,
            };
        }

        private void ClearSpeakerCandidate()
        {
            SpeakerCandidateId cancelled;
            lock (_lock)
            {
                cancelled = _speakerCandidate.CandidateId;
                ClearSpeakerCandidateLocked();
            }
            if (!cancelled.IsNone)
                _speakerEvidence?.Cancel(cancelled);
        }

        private void ClearSpeakerCandidateLocked() => _speakerCandidate.Clear();

        private bool TryAuthorizeSpeakerCandidate(long generation)
        {
            lock (_lock)
            {
                if (!_speakerCandidate.IsCurrent(generation))
                    return false;

                _speakerCandidate.Authorize();
                _lastCandidateOutcome = "command-authorized";
                return true;
            }
        }

        private bool TryClearSpeakerCandidate(long generation)
        {
            SpeakerCandidateId cancelled;
            lock (_lock)
            {
                if (!_speakerCandidate.IsCurrent(generation))
                    return false;

                cancelled = _speakerCandidate.CandidateId;
                ClearSpeakerCandidateLocked();
            }
            if (!cancelled.IsNone)
                _speakerEvidence?.Cancel(cancelled);
            return true;
        }

        private static bool IsStrongNearEndDecision(in BargeInDecision decision)
            => decision.Action == BargeInAction.Interrupt &&
                !decision.HasDiscontinuity &&
                decision.LatestFrameAccepted &&
                decision.EchoLikeFrameCount == 0 &&
                decision.AcceptedDurationMs >= decision.RequiredDurationMs;

        private void TryStartAuthorizedCandidate()
        {
            BargeInDecision decision;
            long generation;
            lock (_lock)
            {
                if (!_speakerCandidate.Active ||
                    !_speakerCandidate.Authorized ||
                    _capture.ActiveTurnId != 0)
                {
                    _lastCandidateOutcome = "skipped:candidate-state-mismatch";
                    return;
                }
                decision = _speakerCandidate.Decision;
                generation = _speakerCandidate.Generation;
            }

            if (!TryStartCapture(
                decision,
                out IReadOnlyList<float[]> initialAudio,
                out ActiveNearEndPreRollSelection preRoll,
                out bool endpointVadCandidate,
                out bool retryable))
            {
                string outcome = $"failed:{preRoll.Source}:{(retryable ? "retryable" : "terminal")}";
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
                if (!retryable)
                {
                    ClearSpeakerCandidate();
                    RestorePlaybackDucking();
                }
                return;
            }

            long operationId;
            lock (_lock)
            {
                if (_pendingConfirmation is not null)
                    return;
                if (!_speakerCandidate.IsCurrent(generation))
                    return;

                operationId = _control.NextOperationId();
                _pendingConfirmation = new ActiveNearEndConfirmationRequest(
                    operationId,
                    initialAudio,
                    decision,
                    preRoll,
                    endpointVadCandidate,
                    _speakerCandidate.IdentityRequired,
                    _speakerCandidate.IdentityRequired && _speakerCandidate.Authorized,
                    _speakerCandidate.CandidateId,
                    _speakerCandidate.OwnerTurnId,
                    _speakerCandidate.SpeakerResult);
                _confirmationTailNextSample = preRoll.SelectedEndSample;
                _lastCandidateOutcome = "queued:confirmation";
            }

            if (!_control.TryEnqueueConfirmation(operationId))
            {
                lock (_lock)
                {
                    RestorePlaybackDuckingLocked();
                    if (_pendingConfirmation?.OperationId == operationId)
                        _pendingConfirmation = null;
                    _lastCandidateOutcome = "failed:confirmation-overflow";
                }
            }

            // The candidate has been handed to the control plane (or failed terminally); its
            // module-private state is released either way. Cached embeddings stay reusable.
            lock (_lock)
                ClearSpeakerCandidateLocked();
        }

        private bool TryStartCapture(
            BargeInDecision decision,
            out IReadOnlyList<float[]> initialAudio,
            out ActiveNearEndPreRollSelection selection,
            out bool endpointVadCandidate,
            out bool retryable)
        {
            retryable = false;
            endpointVadCandidate = false;
            int preRollSamples = ComputePreRollSamples();
            int candidateAudioLimitSamples = ComputeTimelineAlignedAudioLimitSamples();
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
                long selectedStreamEpoch = 0;
                long selectedStartSample = 0;
                long selectedEndSample = 0;
                endpointVadCandidate = TryGetEndpointVadCandidateStartLocked(
                    decision,
                    out long endpointVadStreamEpoch,
                    out long endpointVadStartSample,
                    out bool waitForEndpointVadLocation);
                if (waitForEndpointVadLocation)
                {
                    _logger.LogDebug(
                        "近端窗口已确认，等待主 VAD 首段音频定位: decision=[{DecisionStart},{DecisionEnd}), " +
                        "mainEnd={MainEnd}, epoch={StreamEpoch}.",
                        decision.StartSample,
                        decision.EndSampleExclusive,
                        _mainAudioEndSample,
                        decision.StreamEpoch);
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
                        Math.Max(decision.EndSampleExclusive, _mainAudioEndSample),
                        candidateAudioLimitSamples,
                        out preRoll,
                        out selectedStartSample,
                        out selectedEndSample,
                        out retryable))
                    {
                        _logger.LogDebug(
                            "主 VAD 起点快照尚不可用: requestedStart={RequestedStart}, mainEnd={MainEnd}, " +
                            "maxSamples={MaxSamples}, retryable={Retryable}.",
                            endpointVadStartSample,
                            _mainAudioEndSample,
                            candidateAudioLimitSamples,
                            retryable);
                        initialAudio = [];
                        selection = ActiveNearEndPreRollSelection.Empty("main-vad-timeline");
                        return false;
                    }
                    source = "main-vad-timeline";
                    selectedStreamEpoch = endpointVadStreamEpoch;
                }
                else if (_speakerCandidate.MainStartSample >= 0)
                {
                    long speakerPreRollStart = Math.Max(
                        0,
                        _speakerCandidate.MainStartSample - preRollSamples);
                    if (!TrySnapshotTimelineAligned(
                        _speakerCandidate.StreamEpoch,
                        speakerPreRollStart,
                        Math.Max(decision.EndSampleExclusive, _mainAudioEndSample),
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
                    selectedStreamEpoch = _speakerCandidate.StreamEpoch;
                }
                else if (HasUsableTimeline(decision))
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

        public bool TryTakeConfirmation(
            long operationId,
            out ActiveNearEndConfirmationRequest request)
        {
            lock (_lock)
            {
                if (_pendingConfirmation is { } pending && pending.OperationId == operationId)
                {
                    _pendingConfirmation = null;
                    _confirmationTailCapture = pending;
                    request = pending;
                    return true;
                }
            }

            request = default;
            return false;
        }

        public void DiscardTakenConfirmation(long operationId)
        {
            lock (_lock)
            {
                if (_confirmationTailCapture?.OperationId != operationId)
                    return;

                _confirmationTailCapture = null;
                ClearConfirmationTailFramesLocked();
                RestorePlaybackDuckingLocked();
            }
        }

        public void ApplyConfirmedCapture(
            in ActiveNearEndConfirmationRequest request,
            in SpeakingDecision confirmed)
        {
            bool finalize = false;
            TimeSpan speechDuration = TimeSpan.Zero;
            ActiveEndDrainOutcome drainOutcome = default;
            lock (_lock)
            {
                if (_capture.ActiveTurnId != 0)
                    return;

                ClearEndDrainLocked();
                Volatile.Write(ref _dataPlaneFaultReported, 0);
                _capture.ActiveTurnId = confirmed.TurnId;
                _capture.DirectFeed = true;
                _speechDurationMs = Math.Max(
                    0,
                    _audioFormat.SamplesToMilliseconds(confirmed.ConfirmedCandidateSamples));
                _silenceMs = 0;
                _activePreRollSource = request.Selection.Source;
                _activePreRollSamples = request.Selection.Samples;
                _capture.DirectFeedSamples = 0;
                // Start at the immutable confirmation boundary so callbacks that arrived
                // while the control-plane confirmation was queued can be replayed exactly once.
                _capture.DirectFeedNextSample = request.Selection.SelectedEndSample;
                _endpoint.Observed = request.EndpointVadCandidate;
                _endpoint.SpeechActive = _endpoint.Observed;
                bool endpointPredatesCaptureCommit =
                    _endpoint.UnownedObserved &&
                    _endpoint.UnownedProcessed &&
                    _endpoint.UnownedGeneration == _mainAudioGeneration &&
                    _endpoint.UnownedStreamEpoch == _mainAudioStreamEpoch &&
                    _endpoint.UnownedEndSample >= request.Selection.SelectedEndSample &&
                    _endpoint.UnownedEndSample <= Math.Max(
                        _mainAudioEndSample,
                        request.Selection.SelectedEndSample);
                if (endpointPredatesCaptureCommit)
                {
                    _endpoint.Observed = true;
                    _endpoint.SpeechActive = false;
                    // The control plane consumed an endpoint at or after the immutable
                    // confirmation boundary, so both it and any already-observed DTD tail are
                    // fixed facts. An endpoint before that boundary is stale: later DTD evidence
                    // already extended the candidate past it and must remain eligible for ASR.
                    BeginEndDrainLocked(
                        confirmed.TurnId,
                        _mainAudioStreamEpoch,
                        _mainAudioEndSample,
                        hasDiscontinuity: false);
                }
                ClearUnownedEndpointVadObservationLocked();
                ClearEndpointVadCandidateLocked();
                _confirmationTailCapture = null;
            }

            LogCaptureStarted(confirmed.TurnId, request.Selection);
            DrainConfirmationTail();
            DrainUnfedMainAudioAfterConfirmation();
            lock (_lock)
            {
                if (_capture.ActiveTurnId != 0 && _capture.DirectFeed && !_capture.EndDrainPending)
                {
                    // Confirmation is requested on the duplex callback, before the same
                    // callback's processed-capture frame is observed. Any remaining gap is
                    // already-owned main audio and must be fed from the reservoir; only a
                    // range that can no longer be snapshotted is skipped.
                    _capture.DirectFeedNextSample = Math.Max(
                        _capture.DirectFeedNextSample,
                        _mainAudioEndSample);
                }
                finalize = TryClaimEndFinalizationLocked(out speechDuration, out drainOutcome);
            }

            if (finalize)
                EndCapture(confirmed.TurnId, speechDuration, drainOutcome);
        }

        private void BeginTimelineCandidate(in BargeInDecision decision)
        {
            bool created = false;
            long ownerTurnId = _conversation.CurrentTurnId;
            lock (_lock)
            {
                if (_speakerCandidate.Active)
                    return;

                // 主识别链可能落后于检测链。固定首次确认决定，后续重试不能改用一个
                // 持续向前滑动的新窗口，否则两条等速消费线程会永远追不上目标起点。
                _speakerCandidate.BeginPreAuthorizedTimeline(in decision, ownerTurnId);
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

        private void SetCandidateClearOutcome(string outcome)
        {
            lock (_lock)
            {
                if (_speakerCandidate.Active)
                    _lastCandidateOutcome = outcome;
            }
        }

        private static bool HasUsableTimeline(in BargeInDecision decision)
            => !decision.HasDiscontinuity &&
               decision.StreamEpoch > 0 &&
               decision.StartSample >= 0 &&
               decision.EndSampleExclusive > decision.StartSample;

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
            long historySamples = SaturatingAdd(candidateSamples, preRollSamples);

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
                // The first causal onset prevents a late retry from moving the
                // candidate forward. Keep the existing candidate budget as a lower
                // bound so configured ASR pre-roll and the captured weak prefix remain
                // intact when they are still retained in the shared reservoir.
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
            if (!_mainAudioReservoir.TryGetBounds(
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
            if (!_mainAudioReservoir.TrySnapshotFrom(
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

        private static int CountSamples(IReadOnlyList<float[]> chunks)
        {
            int total = 0;
            for (int i = 0; i < chunks.Count; i++)
                total += chunks[i].Length;
            return total;
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

        /// <summary>
        /// Credits fed samples to the turn that reserved them, so a capture that stopped during the
        /// lock-free provider call cannot attribute them to a later turn's diagnostics.
        /// </summary>
        private void AddDirectFeedSamples(long turnId, int samples)
        {
            if (samples <= 0)
                return;

            lock (_lock)
            {
                if (_capture.ActiveTurnId == turnId && _capture.DirectFeed)
                    _capture.DirectFeedSamples += samples;
            }
        }

        private ActiveCaptureAudioStats SnapshotActiveCaptureStats()
        {
            lock (_lock)
            {
                return new ActiveCaptureAudioStats(
                    _activePreRollSource,
                    _activePreRollSamples,
                    _capture.DirectFeedSamples);
            }
        }

        private readonly record struct ActiveCaptureAudioStats(
            string PreRollSource,
            int PreRollSamples,
            long DirectFeedSamples);

        /// <summary>
        /// Feeds one reserved range to ASR outside the data-plane lock. The turn identity is the one
        /// the caller reserved under that lock: re-reading the live field here would let a
        /// concurrent StopCapture route the reserved samples to another turn (or to turn 0, whose
        /// missing session then misreports the failure as a pending-queue overflow).
        /// </summary>
        private void FeedAudio(long turnId, ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty)
                return;

            if (!_recognitionAudio.TryFeedAudio(turnId, audio))
            {
                ReportDataPlaneFault(
                    _recognitionAudio.HasAttachedSession(turnId)
                        ? "asr-feed-queue-overflow"
                        : "asr-pending-queue-overflow");
                return;
            }

            AddDirectFeedSamples(turnId, audio.Length);
        }

        private void ClearConfirmationTailFramesLocked()
        {
            while (_confirmationTailFrames.TryRead(out _))
            {
            }
            _confirmationTailNextSample = 0;
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

            _control.EnqueueSpeechEnded(turnId, speechDuration);
        }

        private void ReportDataPlaneFault(string detail)
        {
            if (Interlocked.Exchange(ref _dataPlaneFaultReported, 1) != 0)
                return;

            _control.ReportDataPlaneFault(new AssistantFailure(
                "VBX-CTRL-018",
                "capacity",
                "active-near-end-data-plane",
                detail));
        }

        private int ComputeReservoirSamples()
        {
            int reservoirMs = Math.Max(
                ComputeTimelineAlignedAudioLimitMs(),
                _options.ActiveNearEndWindowMs);
            return _audioFormat.SamplesFromMillisecondsToInt32(
                reservoirMs,
                _audioFormat.AudioSampleRate,
                _audioFormat.AudioChannels);
        }

        private int ComputePreRollSamples()
            => _audioFormat.SamplesFromMillisecondsToInt32(
                _options.ActiveNearEndPreRollMs,
                _audioFormat.AudioSampleRate,
                _audioFormat.AudioChannels);

        private int ComputeTimelineAlignedAudioLimitSamples()
            => _audioFormat.SamplesFromMillisecondsToInt32(
                ComputeTimelineAlignedAudioLimitMs(),
                _audioFormat.AudioSampleRate,
                _audioFormat.AudioChannels);

        private int ComputeTimelineAlignedAudioLimitMs()
        {
            // 两条异步处理链共享同一 capture timeline。复用已有的跨链 drain 预算作为
            // 对齐保留上限，不引入任何由当前设备实验得到的常数。
            int candidateMs = Math.Max(
                ComputeSpeakerCandidateAudioLimitMs(),
                Math.Max(
                    checked(_options.PlaybackMaxCandidateMs + _options.ActiveNearEndPreRollMs),
                    checked(_options.ActiveNearEndWindowMs + _options.ActiveNearEndPreRollMs)));
            return checked(candidateMs + _options.ActiveNearEndAudioDrainTimeoutMs);
        }

        private int ComputeSpeakerCandidateAudioLimitMs()
            => _speakerEvidence is null
                ? _options.ActiveNearEndPreRollMs
                : checked(_options.ActiveNearEndPreRollMs + _options.PlaybackMaxCandidateMs);

        private static double ComputeFrameDurationMs(in DuplexAudioFrame frame)
        {
            if (frame.SampleRate <= 0 || frame.FrameCount <= 0)
                return 20;

            return Math.Clamp(frame.FrameCount * 1000.0 / frame.SampleRate, 1, 100);
        }
    }
}
