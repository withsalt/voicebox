using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Extensions;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Modules.Recognition.Controllers;

/// <summary>
/// Drives one recognition operation from session start to the finish signal. It owns the deferred
/// recognition-ready ledger and the supervised session tasks, and returns typed semantic plans
/// for the control-loop router to commit instead of applying global transitions itself.
/// </summary>
internal sealed class RecognitionTurnCoordinator :
    IRecognitionPreparationPort,
    IRecognitionSessionCleanupPort
{
    private readonly RecognitionModule _recognition;
    private readonly RecognitionTurnOwner _recognitionTurn;
    private readonly ITurnStateView _turnState;
    private readonly TurnContextStore _turnContexts;
    private readonly ControlOperationRegistry _operations;
    private readonly IBargeInTurnSupportPort _bargeIn;
    private readonly INearEndCaptureControlPort _nearEndCapture;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly RuntimeRunScope _runScope;
    private readonly InputGate _gate;
    private readonly ILogger _logger;
    private readonly HashSet<long> _deferredReadyTurns = [];
    private readonly object _deferredReadyLock = new();
    // Null exactly when barge-in is not installed; a validation turn is then unreachable.
    private IBargeInRecognitionOutcomePort? _recognitionOutcome;
    // The whole-utterance recognition result of a single-talk turn whose speaker verdict is still
    // pending. Control-loop owned: written and taken only on the control consumer.
    private TurnSignal? _parkedCompletion;

    public RecognitionTurnCoordinator(
        RecognitionModule recognition,
        RecognitionTurnOwner recognitionTurn,
        ITurnStateView turnState,
        TurnContextStore turnContexts,
        ControlOperationRegistry operations,
        IBargeInTurnSupportPort bargeIn,
        INearEndCaptureControlPort nearEndCapture,
        AssistantTaskSupervisor tasks,
        RuntimeRunScope runScope,
        InputGate gate,
        ILogger logger)
    {
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _recognitionTurn = recognitionTurn ?? throw new ArgumentNullException(nameof(recognitionTurn));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _bargeIn = bargeIn ?? throw new ArgumentNullException(nameof(bargeIn));
        _nearEndCapture = nearEndCapture ?? throw new ArgumentNullException(nameof(nearEndCapture));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Binds the recognition-outcome runner once both owners exist. Pass null when barge-in is not installed.
    /// </summary>
    public void Bind(IBargeInRecognitionOutcomePort? recognitionOutcome)
    {
        _recognitionOutcome = recognitionOutcome;
    }

    public void MarkDeferred(long turnId)
    {
        lock (_deferredReadyLock)
            _deferredReadyTurns.Add(turnId);
    }

    public bool TakeDeferred(long turnId)
    {
        lock (_deferredReadyLock)
            return _deferredReadyTurns.Remove(turnId);
    }

    public bool IsDeferred(long turnId)
    {
        lock (_deferredReadyLock)
            return _deferredReadyTurns.Contains(turnId);
    }

    public int DeferredCount
    {
        get
        {
            lock (_deferredReadyLock)
                return _deferredReadyTurns.Count;
        }
    }

    public void ClearDeferred()
    {
        lock (_deferredReadyLock)
            _deferredReadyTurns.Clear();
        _parkedCompletion = null;
    }

    /// <summary>
    /// Decides how a whole-utterance recognition result meets the current turn's identity gate.
    /// A plain turn (<c>NotRequired</c>/<c>Authorized</c>) proceeds to the Kernel; a pending
    /// single-talk verdict parks the result; a denied verdict drops it and the turn completes
    /// without a response. ASR text never substitutes for the speaker verdict.
    /// </summary>
    public RecognitionCompletionGate ObserveCompletionGate(long turnId)
    {
        TurnSnapshot snapshot = _turnState.Snapshot;
        if (snapshot.TurnId != turnId ||
            snapshot.ValidationTurnId != 0 ||
            snapshot.ConversationState is not (AssistantState.Listening or AssistantState.Recognizing))
        {
            return RecognitionCompletionGate.Proceed;
        }

        if (snapshot.Authorization == TurnAuthorizationStatus.PendingSpeaker)
            return RecognitionCompletionGate.Park;
        return snapshot.Authorization.IsDenied()
            ? RecognitionCompletionGate.Drop
            : RecognitionCompletionGate.Proceed;
    }

    /// <summary>Parks the Kernel projection of a completed recognition until the single-talk verdict.</summary>
    public void ParkCompletion(TurnSignal completion)
    {
        if (completion.Kind != TurnSignalKind.RecognitionCompleted)
            throw new ArgumentException("Only a RecognitionCompleted signal can be parked.", nameof(completion));
        if (_parkedCompletion is { } existing && existing.TurnId != completion.TurnId)
            _logger.LogWarning("丢弃过期的单讲识别结果: turnId={TurnId}。", existing.TurnId);
        _parkedCompletion = completion;
        _logger.LogDebug(
            "单讲识别结果已暂存, 等待说话人裁决: turnId={TurnId}, text=\"{Text}\"。",
            completion.TurnId,
            completion.Text);
    }

    public bool TryTakeParkedCompletion(long turnId, out TurnSignal completion)
    {
        if (_parkedCompletion is { } parked && parked.TurnId == turnId)
        {
            _parkedCompletion = null;
            completion = parked;
            return true;
        }

        completion = default;
        return false;
    }

    public bool HasParkedCompletion(long turnId)
        => _parkedCompletion is { } parked && parked.TurnId == turnId;

    public void PrepareRecognitionSession(long turnId)
        => Track(
            _recognition.PrepareAsync(
                turnId,
                _operations.NextOperationId(),
                _runScope.Token),
            "recognition.start",
            turnId);

    public Task StartAudioPump(CancellationToken cancellationToken)
        => Track(_recognition.StartAudioPump(cancellationToken), "recognition.audio-feed-pump", ownerTurnId: null);

    public ValueTask DrainAudioAsync() => _recognition.DrainAudioAsync();

    public Task CleanupAbandonedSessionAsync(string sessionId)
        => _recognition.CleanupAbandonedAsrSessionAsync(sessionId);

    public string? TakeAbandonedSession() => _recognitionTurn.TakeActiveSessionAndClearPending();

    /// <summary>
    /// Attaches a provider session the module just opened and replays its buffered pre-roll. The
    /// returned result lets the control-loop router decide whether to enter Recognizing.
    /// </summary>
    public RecognitionSessionStartResult ObserveSessionStarted(AssistantSignal signal)
    {
        string sessionId = signal.Text ?? string.Empty;
        bool recognitionReadyWasDeferred = IsDeferred(signal.TurnId);
        bool attached = _recognitionTurn.TryAttachStartedSession(
            signal.TurnId,
            sessionId,
            out bool speechAlreadyEnded,
            out TimeSpan speechDuration);

        return new RecognitionSessionStartResult(
            TurnId: signal.TurnId,
            OperationId: signal.OperationId,
            Attached: attached,
            IsValidationTurn: _bargeIn.IsValidating(signal.TurnId),
            RecognitionReadyWasDeferred: recognitionReadyWasDeferred,
            SpeechAlreadyEnded: speechAlreadyEnded,
            SpeechDuration: speechDuration,
            SessionId: sessionId);
    }

    /// <summary>
    /// Applies the non-semantic tail of a started-session observation after the control loop has
    /// projected/committed the corresponding recognition-ready signal.  Keeping this after the
    /// commit prevents a speech-end-before-ready path from recursively committing the same
    /// recognition-ready transition through the module's request port.
    /// </summary>
    public RecognitionSemanticPlan CommitSessionStarted(
        RecognitionSessionStartResult started,
        KernelCommitResult kernel)
    {
        if (!started.Attached)
        {
            TakeDeferred(started.TurnId);
            _recognition.ReleaseCaptureTurn(started.TurnId);
            if (!string.IsNullOrWhiteSpace(started.SessionId))
            {
                Track(
                    _recognition.CleanupAbandonedAsrSessionAsync(started.SessionId),
                    "recognition.cleanup-unattached-session",
                    started.TurnId);
            }
            return default;
        }

        TakeDeferred(started.TurnId);

        if (!started.IsValidationTurn &&
            !started.RecognitionReadyWasDeferred &&
            !kernel.Accepted)
        {
            _recognition.ReleaseCaptureTurn(started.TurnId);
            return default;
        }

        if (!_turnContexts.TryAttachAsrSession(
                started.TurnId,
                started.OperationId,
                started.SessionId))
        {
            _recognition.ReleaseCaptureTurn(started.TurnId);
            Track(
                _recognition.CleanupAbandonedAsrSessionAsync(started.SessionId),
                "recognition.cleanup-untracked-session",
                started.TurnId);
            return default;
        }
        _logger.LogDebug("ASR 会话已启动: {SessionId}", started.SessionId);
        if (!started.SpeechAlreadyEnded)
            return default;

        return BeginRecognition(started.TurnId, started.SessionId, started.SpeechDuration);
    }

    public RecognitionSemanticPlan BeginRecognition(long turnId, string asrSessionId, TimeSpan speechDuration)
    {
        AssistantState conversationState = _turnState.ConversationState;
        if (!_recognitionTurn.CanEnterRecognizing(turnId, asrSessionId, out bool isValidationTurn))
        {
            _logger.LogDebug("忽略 OnSpeechEnded: 当前状态非 Listening。");
            _bargeIn.ForgetDecisionContext(turnId);
            if ((_turnState.IsCurrentTurn(turnId) || _bargeIn.IsValidating(turnId)) && _runScope.CanOpenInput)
                _gate.Open();
            Track(
                _recognition.CleanupAbandonedAsrSessionAsync(asrSessionId),
                "recognition.cleanup-abandoned-session",
                turnId);
            return default;
        }

        if (isValidationTurn)
        {
            // An interruption turn can only exist when barge-in is installed. Fail deterministically
            // rather than dereferencing an unbound runner.
            IBargeInRecognitionOutcomePort recognitionOutcome = _recognitionOutcome
                ?? throw new InvalidOperationException(
                    "A playback interruption turn was observed without an installed barge-in recognition outcome runner.");
            _gate.Close();
            if (!_recognitionTurn.CommitEnterRecognizing(
                    turnId,
                    asrSessionId,
                    isValidationTurn: true))
            {
                throw new InvalidOperationException(
                    "Interruption recognition observation could not be committed.");
            }
            _bargeIn.ClearValidationPendingAudio();
            _bargeIn.ClearCandidates();
            Track(
                Task.Run(
                    () => recognitionOutcome.RunRecognitionOutcomeAsync(
                        turnId,
                        asrSessionId,
                        speechDuration,
                        _runScope.Token),
                    _runScope.Token),
                "recognition.barge-in-outcome",
                turnId);
            return default;
        }

        if (conversationState == AssistantState.Listening)
        {
            return RecognitionSemanticPlan.SessionReady(
                turnId,
                asrSessionId,
                speechDuration,
                TurnSignal.RecognitionSessionReady(
                    turnId,
                    _operations.NextOperationId()));
        }

        // RecognitionStarted already crossed the Kernel boundary before an endpoint that raced
        // session startup was replayed. Commit the capture owner's private transition now.
        _gate.Close();
        if (!_recognitionTurn.CommitEnterRecognizing(
                turnId,
                asrSessionId,
                isValidationTurn: false))
        {
            throw new InvalidOperationException(
                "Recognition observation could not be committed after session-ready acceptance.");
        }
        _bargeIn.ClearCandidates();

        return CreateSpeechEndedPlan(turnId, asrSessionId, speechDuration);
    }

    private RecognitionSemanticPlan CreateSpeechEndedPlan(
        long turnId,
        string asrSessionId,
        TimeSpan speechDuration)
    {

        if (!_recognition.TryGetProviderSession(turnId, out _, out string? attachedSessionId) ||
            !StringComparer.Ordinal.Equals(attachedSessionId, asrSessionId))
        {
            _bargeIn.ForgetDecisionContext(turnId);
            Track(
                _recognition.CleanupAbandonedAsrSessionAsync(asrSessionId),
                "recognition.cleanup-abandoned-session",
                turnId);
            return default;
        }

        long finishOperationId = _operations.NextOperationId();
        return RecognitionSemanticPlan.SpeechEnded(
            turnId,
            asrSessionId,
            speechDuration,
            TurnSignal.SpeechEnded(turnId, finishOperationId, speechDuration, asrSessionId));
    }

    public RecognitionSemanticPlan CommitSemanticPlan(
        RecognitionSemanticPlan plan,
        KernelCommitResult kernel)
    {
        if (plan.Kind == RecognitionSemanticPlanKind.None)
            return default;

        if (!kernel.Accepted)
        {
            _recognition.ReleaseCaptureTurn(plan.TurnId);
            Track(
                _recognition.CleanupAbandonedAsrSessionAsync(plan.SessionId),
                "recognition.cleanup-abandoned-session",
                plan.TurnId);
            _bargeIn.ForgetDecisionContext(plan.TurnId);
            return default;
        }

        if (plan.Kind == RecognitionSemanticPlanKind.SpeechEnded)
        {
            _turnContexts.TrySetRecognitionMetadata(
                plan.TurnId,
                rawText: null,
                plan.SpeechDuration);
            return default;
        }

        _gate.Close();
        if (!_recognitionTurn.CommitEnterRecognizing(
                plan.TurnId,
                plan.SessionId,
                isValidationTurn: false))
        {
            throw new InvalidOperationException(
                "Recognition observation could not be committed after Kernel acceptance.");
        }
        _bargeIn.ClearCandidates();

        return CreateSpeechEndedPlan(plan.TurnId, plan.SessionId, plan.SpeechDuration);
    }

    /// <summary>
    /// Unwinds a turn whose ASR session never started. A validation turn additionally has to
    /// resolve its interruption as rejected so the interrupted playback state is restored.
    /// </summary>
    public RecognitionResetPlan ObserveResetAfterAsrStartFailure(long turnId)
    {
        bool isValidationTurn = _bargeIn.IsValidating(turnId);
        AssistantState restored;
        if (isValidationTurn)
        {
            if (!_bargeIn.TryObserveAsrStartFailure(turnId, out restored))
                return default;
        }
        else
        {
            if (!_recognitionTurn.CanObserveActiveAsrStartFailure(turnId))
                return default;
            restored = AssistantState.Idle;
        }

        bool alreadyAccepted = isValidationTurn && _turnState.IsCurrentTurn(turnId);
        if (isValidationTurn && !alreadyAccepted)
        {
            return RecognitionResetPlan.ValidationRejected(
                turnId,
                restored,
                TurnSignal.InterruptionResolved(
                    turnId,
                    _operations.NextOperationId(),
                    accepted: false,
                    reason: "asr-start-failed",
                    restoreState: restored));
        }

        return RecognitionResetPlan.FinalizeOnly(turnId, isValidationTurn, alreadyAccepted, restored);
    }

    public void CommitResetAfterAsrStartFailure(
        RecognitionResetPlan plan,
        KernelCommitResult kernel,
        CancellationToken cancellationToken)
    {
        if (!plan.Observed)
            return;

        bool isValidationTurn = plan.IsValidationTurn;
        if (plan.HasSemanticSignal && !kernel.Accepted)
            return;

        TakeDeferred(plan.TurnId);
        if (isValidationTurn)
            _nearEndCapture.StopCapture(plan.TurnId, startCooldown: true);

        if (isValidationTurn && plan.HasSemanticSignal)
        {
            if (!_bargeIn.CommitAsrStartFailure(plan.TurnId))
            {
                throw new InvalidOperationException(
                    "ASR-start failure observation could not be committed after Kernel acceptance.");
            }

            _bargeIn.CompleteDecisionContext(plan.TurnId);
        }
        else if (isValidationTurn)
        {
            if (!_bargeIn.CommitAsrStartFailure(plan.TurnId))
                return;
        }
        else
        {
            // FinishAsync already retired capture on the session-failed path. Commit is then a
            // no-op, but deferred ready, journal, input gate and turn context still need the
            // same unwind as a true ASR-start failure.
            _ = _recognitionTurn.CommitAsrStartFailure(plan.TurnId);
            _bargeIn.ClearCandidates();
        }

        _bargeIn.ForgetDecisionContext(plan.TurnId);
        if (isValidationTurn && !plan.AlreadyAccepted)
            _bargeIn.HandleRejectedStateRestored(plan.RestoredState);
        if (!cancellationToken.IsCancellationRequested && _runScope.CanOpenInput)
            _gate.Open();
        _recognition.ReleaseCaptureTurn(plan.TurnId);
        _turnContexts.Release(plan.TurnId);
    }

    private Task Track(Task task, string taskName, long? ownerTurnId)
        => _tasks.Track(task, taskName, ownerTurnId, AssistantTaskFaultPolicy.ObserveAndLog);
}
