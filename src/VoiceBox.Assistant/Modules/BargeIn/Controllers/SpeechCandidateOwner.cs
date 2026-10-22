using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Owns speech-candidate accumulation for the Idle listening gate and for playback barge-in:
/// the candidate buffer, its confirmation thresholds and the candidate kind. All of it is module
/// private. Global Turn state is read from the committed Kernel snapshot and never written here.
/// </summary>
internal sealed class SpeechCandidateOwner : ICandidatePreRollSource, ISpeechCandidatePort
{
    private readonly object _lock = new();
    private readonly BargeInDetector _detector;
    private readonly BargeInConfirmationPolicy _policy;
    private readonly BargeInConfirmationSettings? _listeningGate;
    private readonly ITurnStateView _turnState;
    private readonly ITurnResourcePort _resources;
    private SpeechCandidateKind _kind;

    public SpeechCandidateOwner(
        BargeInDetector detector,
        BargeInConfirmationPolicy policy,
        BargeInConfirmationSettings? listeningGate,
        ITurnStateView turnState,
        ITurnResourcePort resources)
    {
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _resources = resources ?? throw new ArgumentNullException(nameof(resources));
        _listeningGate = listeningGate;
    }

    public bool IsCandidateActive
    {
        get { lock (_lock) return _detector.IsCandidateActive; }
    }

    /// <summary>A confirmed barge-in candidate still waiting for the speaker gate verdict.</summary>
    public bool IsAwaitingSpeaker
    {
        get
        {
            lock (_lock)
            {
                return _detector.IsCandidateActive &&
                    _kind == SpeechCandidateKind.BargeIn &&
                    _detector.IsConfirmationMet;
            }
        }
    }

    /// <summary>
    /// Begins a listening or barge-in candidate for a speech-start fact. Returns the candidate
    /// classification without taking Listening ownership; the Kernel decides that.
    /// </summary>
    public SpeechStartedDecision ObserveSpeechStarted(bool enableBargeIn, long proposedTurnId)
    {
        lock (_lock)
            return ObserveSpeechStartedLocked(enableBargeIn, proposedTurnId);
    }

    /// <summary>
    /// Adds capture audio to an active candidate. Returns false when no candidate owns this audio,
    /// which routes it to the recognition path instead.
    /// </summary>
    public bool TryObserveCandidateAudio(
        ReadOnlySpan<float> audio,
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId,
        out SpeakingDecision decision)
    {
        lock (_lock)
        {
            decision = default;
            if (!_detector.IsCandidateActive)
                return false;

            // 监听候选只在 Idle 态有意义; 状态被外部转走 (如 Stop) 后直接废弃。
            if (_kind == SpeechCandidateKind.Listening &&
                _turnState.ConversationState != AssistantState.Idle)
            {
                ClearLocked();
                return true;
            }

            if (!_detector.AddAudio(audio))
                return true;

            decision = ResolveConfirmedLocked(speakerGateStatus, proposedTurnId);
            return true;
        }
    }

    public SpeakingDecision ResolveSpeakerGate(
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId)
    {
        lock (_lock)
        {
            if (!_detector.IsCandidateActive ||
                _kind != SpeechCandidateKind.BargeIn ||
                !_detector.IsConfirmationMet)
            {
                return default;
            }

            return ResolveConfirmedLocked(speakerGateStatus, proposedTurnId);
        }
    }

    /// <summary>
    /// Produces the discard decision for a speech-end that arrives while a candidate is still
    /// unconfirmed, and drops the candidate.
    /// </summary>
    public bool TryDiscardOnSpeechEnded(TimeSpan duration, out SpeechEndedDecision decision)
    {
        lock (_lock)
        {
            decision = default;
            if (!_detector.IsCandidateActive)
                return false;

            SpeechEndedDecisionKind discardedKind = _kind == SpeechCandidateKind.Listening
                ? SpeechEndedDecisionKind.DiscardedListeningCandidate
                : SpeechEndedDecisionKind.DiscardedBargeInCandidate;
            decision = new SpeechEndedDecision(
                discardedKind,
                AsrSessionId: null,
                TurnId: _turnState.TurnId,
                SpeechDuration: duration,
                _detector.CandidateSamples,
                _detector.ComputeRms(),
                _detector.ConfirmationSamples,
                _detector.ComputeConfirmationRms(),
                DiscardedBargeInDiagnostics: default);
            ClearLocked();
            return true;
        }
    }

    /// <summary>
    /// True when a near-end window may open a fresh barge-in: no candidate accumulation is in
    /// flight and the policy has settings for the current state.
    /// </summary>
    public bool TryGetNearEndSettings(AssistantState state, out BargeInConfirmationSettings settings)
    {
        lock (_lock)
        {
            settings = default;
            return !_detector.IsCandidateActive && _policy.TryGetSettings(state, out settings);
        }
    }

    /// <summary>A candidate of this kind exists and has already met its confirmation threshold.</summary>
    public bool CanMarkConfirmed(SpeechCandidateKind expectedKind)
    {
        lock (_lock)
        {
            return _detector.IsCandidateActive &&
                _kind == expectedKind &&
                _detector.IsConfirmationMet;
        }
    }

    /// <summary>Retires a confirmed candidate while retaining its audio for the new ASR session.</summary>
    public bool TryMarkConfirmed(SpeechCandidateKind expectedKind)
    {
        lock (_lock)
        {
            if (!_detector.IsCandidateActive ||
                _kind != expectedKind ||
                !_detector.IsConfirmationMet)
            {
                return false;
            }

            _detector.MarkConfirmedKeepingPendingAudio();
            _kind = SpeechCandidateKind.None;
            return true;
        }
    }

    /// <summary>
    /// Retires the candidate slot for a near-end confirmation, which carries its own pre-roll and
    /// therefore never accumulated a candidate here.
    /// </summary>
    public void MarkNearEndConfirmed()
    {
        lock (_lock)
            _kind = SpeechCandidateKind.None;
    }

    public void Clear()
    {
        lock (_lock)
            ClearLocked();
    }

    /// <summary>
    /// Hands the candidate pre-roll to the recognition session in capture order and releases it.
    /// Candidate audio always precedes audio buffered after confirmation.
    /// </summary>
    public void DrainPendingAudioInto(List<float[]> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        lock (_lock)
        {
            destination.AddRange(_detector.PendingAudio);
            _detector.ClearPendingAudio();
        }
    }

    private SpeechStartedDecision ObserveSpeechStartedLocked(bool enableBargeIn, long proposedTurnId)
    {
        AssistantState state = _turnState.ConversationState;
        long activeTurnId = _turnState.TurnId;

        if (state == AssistantState.Idle)
        {
            // Idle 态先累积候选音频, 达到时长和能量门槛后再开始对话,
            // 避免 TTS 回声残差或环境噪声触发无意义的 Listening 状态与 ASR 会话。
            if (_listeningGate is { } gate)
            {
                BeginCandidateLocked(gate, SpeechCandidateKind.Listening);
                return new SpeechStartedDecision(
                    SpeechStartedDecisionKind.ListeningCandidate,
                    state,
                    state,
                    activeTurnId);
            }

            if (proposedTurnId <= 0)
                throw new ArgumentOutOfRangeException(nameof(proposedTurnId));

            return new SpeechStartedDecision(
                SpeechStartedDecisionKind.StartListening,
                state,
                AssistantState.Listening,
                proposedTurnId);
        }

        // Speaking authorization belongs to the render-aware DTD window; main VAD events
        // must never create an independent playback interruption candidate.
        if (enableBargeIn &&
            state != AssistantState.Speaking &&
            _policy.TryGetSettings(state, out BargeInConfirmationSettings settings))
        {
            BeginCandidateLocked(settings, SpeechCandidateKind.BargeIn);
            return new SpeechStartedDecision(
                SpeechStartedDecisionKind.BargeInCandidate,
                state,
                state,
                activeTurnId);
        }

        return new SpeechStartedDecision(
            SpeechStartedDecisionKind.Ignored,
            state,
            state,
            activeTurnId);
    }

    private SpeakingDecision ResolveConfirmedLocked(
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId)
    {
        long confirmedCandidateSamples = _detector.CandidateSamples;
        double confirmedCandidateRms = _detector.ComputeRms();
        long confirmedConfirmationSamples = _detector.ConfirmationSamples;
        double confirmedConfirmationRms = _detector.ComputeConfirmationRms();
        BargeInConfirmationSettings confirmationSettings = _detector.CurrentSettings;
        AssistantState state = _turnState.ConversationState;

        if (_kind == SpeechCandidateKind.BargeIn &&
            speakerGateStatus != BargeInSpeakerGateStatus.Match)
        {
            if (speakerGateStatus == BargeInSpeakerGateStatus.Pending)
                return default;

            long rejectedTurnId = _turnState.TurnId;
            ClearLocked();
            return new SpeakingDecision(
                SpeakingDecisionKind.RejectedBySpeaker,
                AsrSessionId: null,
                ResponseToCancel: null,
                InterruptedTurnId: rejectedTurnId,
                OldState: state,
                TurnId: rejectedTurnId,
                confirmedCandidateSamples,
                confirmedCandidateRms,
                confirmedConfirmationSamples,
                confirmedConfirmationRms,
                confirmationSettings.MinDurationMs,
                confirmationSettings.MinRms,
                confirmationSettings.InitialSuppressionMs,
                ConfirmedBargeInDiagnostics: default);
        }

        if (proposedTurnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(proposedTurnId));

        if (_kind == SpeechCandidateKind.Listening)
        {
            return new SpeakingDecision(
                SpeakingDecisionKind.ConfirmedListening,
                AsrSessionId: null,
                ResponseToCancel: null,
                InterruptedTurnId: 0,
                state,
                proposedTurnId,
                confirmedCandidateSamples,
                confirmedCandidateRms,
                confirmedConfirmationSamples,
                confirmedConfirmationRms,
                confirmationSettings.MinDurationMs,
                confirmationSettings.MinRms,
                confirmationSettings.InitialSuppressionMs,
                ConfirmedBargeInDiagnostics: default);
        }

        return new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            _detector.ResponseCts,
            _detector.InterruptedTurnId,
            state,
            proposedTurnId,
            confirmedCandidateSamples,
            confirmedCandidateRms,
            confirmedConfirmationSamples,
            confirmedConfirmationRms,
            confirmationSettings.MinDurationMs,
            confirmationSettings.MinRms,
            confirmationSettings.InitialSuppressionMs,
            ConfirmedBargeInDiagnostics: default);
    }

    private void BeginCandidateLocked(BargeInConfirmationSettings settings, SpeechCandidateKind kind)
    {
        ClearLocked();
        // 监听候选在 Idle 态产生, 没有可打断的活跃回复, 不携带 ResponseCts。
        _detector.BeginCandidate(
            _turnState.TurnId,
            kind == SpeechCandidateKind.BargeIn
                ? _resources.GetOrCreate(_turnState.TurnId).ResponseCts
                : null,
            settings);
        _kind = kind;
    }

    private void ClearLocked()
    {
        _detector.Clear();
        _kind = SpeechCandidateKind.None;
    }
}
