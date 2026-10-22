using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Gates;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Controllers;

/// <summary>
/// The control-loop side of the turn boundary. It turns admitted speech facts into exactly one of
/// three outcomes — a candidate observation, a Kernel speech transition, or a recognition
/// hand-off — and owns none of the state those outcomes act on.
/// </summary>
internal sealed class SpeechBoundaryCoordinator : ITurnBoundaryEndpointSink
{
    private readonly ITurnStateView _turnState;
    private readonly TurnStateProjection _projection;
    private readonly ISpeechCandidatePort _candidates;
    private readonly IRecognitionBoundaryPort _recognitionTurn;
    private readonly IBargeInValidationOwnershipPort _validation;
    private readonly IRecognitionAudioPort _recognitionAudio;
    private readonly ISpeakerGateBoundaryPort _speakerGate;
    private readonly IBargeInCandidateDiagnosticsPort _diagnostics;
    private readonly IPlaybackStartupGuardView _startupGuard;
    private readonly IWakeupModule? _wakeup;
    private readonly InputGate _gate;
    private readonly AssistantAudioFormat _audioFormat;
    private readonly VoiceAssistantListeningOptions _listeningOptions;
    private readonly IAssistantControlIngressTerminator _terminalFaults;
    private readonly bool _bargeInInstalled;
    private readonly bool _speakerInstalled;
    private readonly ILogger _logger;
    private readonly INearEndEndpointPort _nearEndEndpoint;
    private readonly INearEndCaptureControlPort _nearEndCapture;
    private bool _skipSingleTalkForReplayCandidate;

    public SpeechBoundaryCoordinator(
        ITurnStateView turnState,
        TurnStateProjection projection,
        ISpeechCandidatePort candidates,
        IRecognitionBoundaryPort recognitionTurn,
        IBargeInValidationOwnershipPort validation,
        IRecognitionAudioPort recognitionAudio,
        ISpeakerGateBoundaryPort speakerGate,
        IBargeInCandidateDiagnosticsPort diagnostics,
        IPlaybackStartupGuardView startupGuard,
        IWakeupModule? wakeup,
        InputGate gate,
        AssistantAudioFormat audioFormat,
        VoiceAssistantListeningOptions listeningOptions,
        IAssistantControlIngressTerminator terminalFaults,
        INearEndEndpointPort nearEndEndpoint,
        INearEndCaptureControlPort nearEndCapture,
        bool bargeInInstalled,
        bool speakerInstalled,
        ILogger logger)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _projection = projection ?? throw new ArgumentNullException(nameof(projection));
        _candidates = candidates ?? throw new ArgumentNullException(nameof(candidates));
        _recognitionTurn = recognitionTurn ?? throw new ArgumentNullException(nameof(recognitionTurn));
        _validation = validation ?? throw new ArgumentNullException(nameof(validation));
        _recognitionAudio = recognitionAudio ?? throw new ArgumentNullException(nameof(recognitionAudio));
        _speakerGate = speakerGate ?? throw new ArgumentNullException(nameof(speakerGate));
        _diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        _startupGuard = startupGuard ?? throw new ArgumentNullException(nameof(startupGuard));
        _wakeup = wakeup;
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _audioFormat = audioFormat ?? throw new ArgumentNullException(nameof(audioFormat));
        _listeningOptions = listeningOptions ?? throw new ArgumentNullException(nameof(listeningOptions));
        _terminalFaults = terminalFaults ?? throw new ArgumentNullException(nameof(terminalFaults));
        _nearEndEndpoint = nearEndEndpoint ?? throw new ArgumentNullException(nameof(nearEndEndpoint));
        _nearEndCapture = nearEndCapture ?? throw new ArgumentNullException(nameof(nearEndCapture));
        _bargeInInstalled = bargeInInstalled;
        _speakerInstalled = speakerInstalled;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    private ActiveNearEndProcessingState NearEndState => _nearEndEndpoint.GetProcessingState();

    public SpeechBoundaryObservation ProcessSpeechStarted(
        bool endpointObservationApplied,
        long operationId,
        bool wakeupReplayAuthorized = false)
    {
        if (endpointObservationApplied || NearEndState.HasActiveCapture)
            return default;
        if (IgnoreDuringPlaybackStartupGuard("OnSpeechStarted"))
            return default;

        if (_projection.Current == AssistantState.Idle &&
            _listeningOptions.MinDurationMs == 0 &&
            _listeningOptions.MinRms == 0)
        {
            long turnId = _turnState.NextTurnOrdinal;
            TurnAuthorizationStatus authorization = EnsureSingleTalkCandidate();
            _logger.LogDebug(
                "VAD 监听到用户说话: turnId={TurnId}, 直接进入 Listening 并启动 ASR (identity={IdentityMode})。",
                turnId,
                DescribeSingleTalkMode(authorization));
            return SpeechBoundaryObservation.StartListening(
                TurnSignal.SpeechStarted(turnId, operationId, authorization: authorization),
                turnId);
        }

        // 注意: 不在此处刷新连续对话活跃时间 —— 候选可能是回声残差/噪声误报,
        // 只有确认为真实语音 (StartListening / Confirmed*) 时才 MarkUserActivity,
        // 否则误报会无限续期连续对话窗口。
        SpeechStartedDecision decision = ObserveSpeechStarted();
        switch (decision.Kind)
        {
            case SpeechStartedDecisionKind.ListeningCandidate:
                // The single-talk identity query starts at the utterance onset so the located range
                // covers the candidate audio as well; the Kernel turn opens only on confirmation.
                if (wakeupReplayAuthorized)
                {
                    _skipSingleTalkForReplayCandidate = true;
                    _logger.LogDebug(
                        "初次唤醒回放已通过 Wakeup 声纹认证，本次回放候选跳过 SingleTalk 身份复核: turnId={TurnId}。",
                        decision.TurnId);
                }
                else
                {
                    EnsureSingleTalkCandidate();
                }
                _logger.LogDebug(
                    "VAD 监听到候选语音: turnId={TurnId}, state={State}, 等待音频门槛确认 (minDurationMs={MinDurationMs}, minRms={MinRms:F4}).",
                    decision.TurnId,
                    decision.CurrentState,
                    _listeningOptions.MinDurationMs,
                    _listeningOptions.MinRms);
                return default;

            case SpeechStartedDecisionKind.BargeInCandidate:
                // KWS enabled: the candidate's own speaker Match is required before it may
                // interrupt; KWS disabled: the gate is a pass-through (NotRequired).
                _speakerGate.BeginCandidate(SpeakerVerificationPurpose.BargeInCommand);
                _logger.LogDebug(
                    "收到 barge-in 候选语音, 等待音频门槛确认 (identity={IdentityMode}).",
                    _speakerGate.IsIdentityRequired ? "pending-speaker" : "not-required:kws-disabled");
                return default;

            case SpeechStartedDecisionKind.Ignored:
                _logger.LogDebug("忽略 OnSpeechStarted: 当前状态 {State} 不接受新语音。", decision.CurrentState);
                return default;
        }

        TurnAuthorizationStatus startAuthorization = decision.OldState == AssistantState.Idle
            ? EnsureSingleTalkCandidate(wakeupReplayAuthorized)
            : TurnAuthorizationStatus.NotRequired;
        _logger.LogDebug(
            "VAD 监听到用户说话: turnId={TurnId}, state={OldState}->{NewState}, 直接进入 Listening 并启动 ASR (identity={IdentityMode})。",
            decision.TurnId,
            decision.OldState,
            AssistantState.Listening,
            DescribeSingleTalkMode(startAuthorization));
        return SpeechBoundaryObservation.StartListening(
            TurnSignal.SpeechStarted(decision.TurnId, operationId, authorization: startAuthorization),
            decision.TurnId);
    }

    /// <summary>
    /// With the effective SingleTalk purpose every Idle utterance carries its own owner-speaker
    /// candidate; the Listening turn then opens as <c>PendingSpeaker</c>. Confirmation.Enable
    /// off (no speaker module) keeps the plain <c>NotRequired</c> path.
    /// </summary>
    private TurnAuthorizationStatus EnsureSingleTalkCandidate(bool wakeupReplayAuthorized = false)
    {
        if (!_speakerGate.IsSingleTalkEffective)
            return TurnAuthorizationStatus.NotRequired;
        if (wakeupReplayAuthorized || _skipSingleTalkForReplayCandidate)
        {
            _skipSingleTalkForReplayCandidate = false;
            return TurnAuthorizationStatus.NotRequired;
        }
        if (_speakerGate.ActivePurpose != SpeakerVerificationPurpose.SingleTalk)
            _speakerGate.BeginCandidate(SpeakerVerificationPurpose.SingleTalk);
        return TurnAuthorizationStatus.PendingSpeaker;
    }

    private static string DescribeSingleTalkMode(TurnAuthorizationStatus authorization)
        => authorization == TurnAuthorizationStatus.PendingSpeaker
            ? "pending-speaker:single-talk"
            : "not-required";

    /// <summary>
    /// Turns a final gate status of the active single-talk candidate into the Kernel verdict of the
    /// current pending turn. Before the Listening turn exists (unconfirmed candidate) nothing is
    /// resolved: the verdict stays in the evidence cache and is re-read on the next chunk or at
    /// the endpoint.
    /// </summary>
    private SpeechBoundaryObservation TryResolveSingleTalkVerdict(
        BargeInSpeakerGateStatus status,
        long operationId,
        bool resetGate)
    {
        if (status == BargeInSpeakerGateStatus.Pending)
            return default;

        TurnSnapshot snapshot = _turnState.Snapshot;
        if (snapshot.ValidationTurnId != 0 ||
            snapshot.ConversationState is not (AssistantState.Listening or AssistantState.Recognizing) ||
            snapshot.Authorization != TurnAuthorizationStatus.PendingSpeaker)
        {
            return default;
        }

        TurnAuthorizationStatus resolved;
        string reason;
        if (status == BargeInSpeakerGateStatus.Match)
        {
            resolved = TurnAuthorizationStatus.Authorized;
            reason = "speaker-match";
            _logger.LogDebug("单讲身份复核通过: turnId={TurnId}。", snapshot.TurnId);
        }
        else
        {
            SpeakerVerificationStatus denied = _speakerGate.LastDeniedStatus;
            resolved = denied switch
            {
                SpeakerVerificationStatus.Mismatch => TurnAuthorizationStatus.DeniedMismatch,
                SpeakerVerificationStatus.InsufficientEvidence => TurnAuthorizationStatus.DeniedInsufficientEvidence,
                _ => TurnAuthorizationStatus.DeniedUnavailable,
            };
            reason = $"command-denied-{denied}";
            _logger.LogInformation(
                "单讲身份复核未通过 ({Status}): turnId={TurnId}, 本句识别结果不会进入语义回合。",
                denied,
                snapshot.TurnId);
        }

        if (resetGate)
            _speakerGate.Reset();
        return SpeechBoundaryObservation.SingleTalkAuthorization(
            TurnSignal.AuthorizationResolved(snapshot.TurnId, operationId, resolved, candidateId: 0, reason),
            snapshot.TurnId);
    }

    private bool IsSingleTalkTurnPending()
    {
        TurnSnapshot snapshot = _turnState.Snapshot;
        return snapshot.ValidationTurnId == 0 &&
            snapshot.ConversationState is (AssistantState.Listening or AssistantState.Recognizing) &&
            snapshot.Authorization == TurnAuthorizationStatus.PendingSpeaker;
    }

    public SpeechBoundaryObservation ProcessSpeaking(float[] audio, long operationId)
    {
        if (audio.Length == 0)
            return default;
        if (NearEndState.HasActiveCapture)
        {
            // Endpoint VAD facts are captured synchronously in the ingress. This queued
            // control-plane replay can lag behind OnSpeechEnded; reapplying it here would
            // incorrectly reopen an endpoint that has already completed.
            return default;
        }

        if (IgnoreDuringPlaybackStartupGuard("OnSpeaking"))
            return default;

        BargeInSpeakerGateStatus speakerGateStatus = _speakerGate.Evaluate(audio);
        SpeechBoundaryObservation singleTalkVerdict = default;
        if (_speakerGate.ActivePurpose == SpeakerVerificationPurpose.SingleTalk)
        {
            // Single talk verifies in parallel with the ASR feed: the verdict is recorded on the
            // pending turn, and the audio still reaches the recognition session unconditionally.
            singleTalkVerdict = TryResolveSingleTalkVerdict(speakerGateStatus, operationId, resetGate: true);
            speakerGateStatus = BargeInSpeakerGateStatus.Match;
        }

        SpeakingDecision decision = ObserveAudio(audio, speakerGateStatus);
        switch (decision.Kind)
        {
            case SpeakingDecisionKind.RejectedBySpeaker:
                _logger.LogDebug(
                    "command-denied-{Status}: VAD barge-in 候选未通过当前窗口说话人验证，保持当前回复: state={State}, samples={Samples}.",
                    _speakerGate.LastDeniedStatus,
                    decision.OldState,
                    decision.ConfirmedCandidateSamples);
                _speakerGate.Reset();
                return default;

            case SpeakingDecisionKind.ConfirmedListening:
                return SpeechBoundaryObservation.ConfirmedListening(
                    TurnSignal.SpeechStarted(
                        decision.TurnId,
                        operationId,
                        authorization: EnsureSingleTalkCandidate()),
                    decision);

            case SpeakingDecisionKind.ConfirmedBargeIn:
                {
                    BargeInIdentityContext identity = CreateIdentityContext();
                    _speakerGate.Reset();
                    return SpeechBoundaryObservation.ConfirmedBargeIn(
                        decision,
                        BargeInTriggerSource.Vad,
                        initialAudio: null,
                        pendingSpeechEnd: null,
                        identity);
                }

            case SpeakingDecisionKind.FeedAsr when decision.AsrSessionId is not null:
                if (!_recognitionAudio.TryFeedAudio(decision.TurnId, audio))
                    ReportRecognitionQueueOverflow(decision.TurnId);
                return singleTalkVerdict;

            case SpeakingDecisionKind.RecognitionQueueOverflow:
                ReportRecognitionQueueOverflow(decision.TurnId);
                return singleTalkVerdict;
        }

        return singleTalkVerdict;
    }

    private void ReportRecognitionQueueOverflow(long turnId)
    {
        _terminalFaults.TryTerminate(
            new AssistantFailure(
                "VBX-CTRL-020",
                "capacity",
                "recognition-audio-data-plane",
                _recognitionAudio.HasAttachedSession(turnId)
                    ? "asr-feed-queue-overflow"
                    : "asr-pending-queue-overflow"),
            Stopwatch.GetTimestamp(),
            out _);
    }

    public SpeechBoundaryObservation ProcessVadSpeakerGateObservation(
        SpeakerVerificationPurpose purpose,
        BargeInSpeakerGateStatus status,
        TimeSpan? pendingSpeechEnd,
        long operationId)
    {
        if (purpose == SpeakerVerificationPurpose.SingleTalk)
        {
            // The endpoint verdict of the single-talk candidate: the gate already retired the
            // candidate when it submitted the verdict, so nothing is reset here.
            return TryResolveSingleTalkVerdict(status, operationId, resetGate: false);
        }

        SpeakingDecision decision = _candidates.ResolveSpeakerGate(status, _turnState.NextTurnOrdinal);
        if (decision.Kind == SpeakingDecisionKind.None)
            return default;
        if (decision.Kind == SpeakingDecisionKind.RejectedBySpeaker)
        {
            _logger.LogDebug(
                "barge-in 候选未通过本人说话人验证，保持当前回复: state={State}, samples={Samples}.",
                decision.OldState,
                decision.ConfirmedCandidateSamples);
            return default;
        }

        if (decision.Kind != SpeakingDecisionKind.ConfirmedBargeIn)
            return default;

        return SpeechBoundaryObservation.ConfirmedBargeIn(
            decision,
            BargeInTriggerSource.Vad,
            initialAudio: null,
            pendingSpeechEnd,
            CreateIdentityContext());
    }

    /// <summary>
    /// A VAD barge-in reaches this point only once its gate answered Match (KWS enabled) or the
    /// gate is a pass-through (KWS disabled). The Kernel candidate id is not needed for this path.
    /// </summary>
    private BargeInIdentityContext CreateIdentityContext()
        => _speakerGate.IsIdentityRequired
            ? BargeInIdentityContext.AuthorizedSpeaker(candidateId: 0)
            : BargeInIdentityContext.NotRequired;

    public SpeechBoundaryObservation ProcessSpeechEnded(TimeSpan duration, bool endpointObservationApplied)
    {
        _skipSingleTalkForReplayCandidate = false;
        if (endpointObservationApplied)
        {
            _speakerGate.Reset();
            return default;
        }

        if (NearEndState.HasActiveCapture)
        {
            _nearEndEndpoint.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint: true);
            _speakerGate.Reset();
            return default;
        }

        _nearEndEndpoint.ObserveEndpointSpeechEnded();
        if (IgnoreDuringPlaybackStartupGuard("OnSpeechEnded"))
        {
            _speakerGate.Reset();
            return default;
        }

        if (_candidates.IsAwaitingSpeaker && _speakerGate.IsIdentityRequired)
        {
            // SpeechEnded arrived first: keep the endpoint and wait the bounded speaker deadline.
            // ASR text never substitutes for the verdict.
            return _speakerGate.CompleteInput(duration);
        }

        if (_speakerGate.ActivePurpose == SpeakerVerificationPurpose.SingleTalk &&
            IsSingleTalkTurnPending() &&
            !_candidates.IsCandidateActive)
        {
            // The utterance ended while its owner verdict is still pending: the whole-utterance ASR
            // finish and the bounded speaker wait now run in parallel; the recognition result is
            // parked until the verdict arrives through the gate observation.
            _ = _speakerGate.CompleteInput(duration);
            return HandleSpeechEnded(duration);
        }

        _speakerGate.Reset();
        return HandleSpeechEnded(duration);
    }

    public SpeechBoundaryObservation HandleSpeechEnded(TimeSpan duration)
    {
        SpeechEndedDecision decision = ObserveSpeechEnded(duration);
        if (decision.TurnId != 0)
            _nearEndCapture.StopCapture(decision.TurnId, startCooldown: true);

        switch (decision.Kind)
        {
            case SpeechEndedDecisionKind.DiscardedListeningCandidate:
                // 未达监听门槛: 视为回声残差/环境噪声, 静默丢弃, 不刷新连续对话活跃时间。
                _diagnostics.LogDiscardedListeningCandidate(decision);
                return default;

            case SpeechEndedDecisionKind.DiscardedBargeInCandidate:
                _diagnostics.RecordDiscardedPlaybackCandidate(decision, _turnState.ConversationState);
                return default;

            // None (迟到/孤立的结束事件) 不算用户活动。
            case SpeechEndedDecisionKind.None:
                return default;
        }

        if (decision.Kind == SpeechEndedDecisionKind.WaitingForAsrStart)
        {
            _gate.Close();
            _logger.LogDebug(
                "用户语音结束, 等待 ASR 会话启动: turnId={TurnId}, durationMs={DurationMs:F0}.",
                decision.TurnId,
                decision.SpeechDuration.TotalMilliseconds);
            return default;
        }

        if (decision.Kind != SpeechEndedDecisionKind.ReadyForRecognition || decision.AsrSessionId is null)
            return default;

        _logger.LogDebug(
            "用户语音结束, 准备开始识别: turnId={TurnId}, sessionId={SessionId}, durationMs={DurationMs:F0}.",
            decision.TurnId,
            decision.AsrSessionId,
            decision.SpeechDuration.TotalMilliseconds);
        return SpeechBoundaryObservation.BeginRecognition(
            decision.TurnId,
            decision.AsrSessionId,
            decision.SpeechDuration);
    }

    /// <summary>Clears the per-turn observation context the Kernel's new Listening turn replaces.</summary>
    public void PrepareListeningTurn(long turnId)
    {
        if (!_turnState.IsCurrentTurn(turnId) ||
            _turnState.ConversationState != AssistantState.Listening)
        {
            throw new InvalidOperationException("Kernel listening turn projection is not current.");
        }

        _recognitionTurn.PrepareTurn();
        _validation.ClearOwnership();
        _candidates.Clear();
        _skipSingleTalkForReplayCandidate = false;
    }

    private SpeechStartedDecision ObserveSpeechStarted()
    {
        if (_validation.HasValidation)
        {
            AssistantState state = _turnState.ConversationState;
            return new SpeechStartedDecision(
                SpeechStartedDecisionKind.Ignored,
                state,
                state,
                _validation.ValidationTurnId);
        }

        return _candidates.ObserveSpeechStarted(_bargeInInstalled, _turnState.NextTurnOrdinal);
    }

    private SpeakingDecision ObserveAudio(float[] audio, BargeInSpeakerGateStatus speakerGateStatus)
        => _candidates.TryObserveCandidateAudio(
            audio,
            speakerGateStatus,
            _turnState.NextTurnOrdinal,
            out SpeakingDecision candidateDecision)
            ? candidateDecision
            : _recognitionTurn.ObserveAudio(audio);

    private SpeechEndedDecision ObserveSpeechEnded(TimeSpan duration)
        => _candidates.TryDiscardOnSpeechEnded(duration, out SpeechEndedDecision discarded)
            ? discarded
            : _recognitionTurn.ObserveSpeechEnded(duration);

    public void CommitListeningObservation(
        SpeechBoundaryObservation observation,
        KernelCommitResult kernel)
    {
        if (!kernel.Accepted)
            return;

        SpeakingDecision decision = observation.SpeakingDecision;
        // Idle 态监听候选通过门槛: 此刻才算真实用户语音, 进入 Listening 并启动 ASR。
        // 候选期间累积的音频已由候选所有者保留, ASR session 就绪后补喂。
        if (decision.TurnId <= 0 ||
            !_candidates.CanMarkConfirmed(SpeechCandidateKind.Listening))
        {
            throw new InvalidOperationException(
                "Confirmed listening observation could not be committed after Kernel acceptance.");
        }

        _recognitionTurn.PrepareTurn();
        _validation.ClearOwnership();
        _candidates.TryMarkConfirmed(SpeechCandidateKind.Listening);

        _logger.LogDebug(
            "确认监听到用户说话: turnId={TurnId}, total={Samples} samples ({DurationMs:F0}ms), rms={Rms:F4}; confirm={ConfirmSamples} samples ({ConfirmDurationMs:F0}ms), rms={ConfirmRms:F4}; threshold=minDurationMs={MinDurationMs}, minRms={MinRms:F4}.",
            decision.TurnId,
            decision.ConfirmedCandidateSamples,
            _audioFormat.SamplesToMilliseconds(decision.ConfirmedCandidateSamples),
            decision.ConfirmedCandidateRms,
            decision.ConfirmedConfirmationSamples,
            _audioFormat.SamplesToMilliseconds(decision.ConfirmedConfirmationSamples),
            decision.ConfirmedConfirmationRms,
            decision.ConfirmationMinDurationMs,
            decision.ConfirmationMinRms);
    }

    public void ApplyStartListening(SpeechBoundaryObservation observation, KernelCommitResult kernel)
    {
        if (!kernel.Accepted)
            return;

        _wakeup?.MarkUserActivity();
        _gate.Open();
        PrepareListeningTurn(observation.TurnId);
    }

    private bool IgnoreDuringPlaybackStartupGuard(string eventName)
    {
        int remainingMs = _startupGuard.RemainingMilliseconds;
        if (remainingMs <= 0)
            return false;

        if (_logger.IsEnabled(LogLevel.Trace))
        {
            _logger.LogTrace(
                "播放启动保护期忽略 {EventName}: remainingMs={RemainingMs}.",
                eventName,
                remainingMs);
        }

        return true;
    }
}
