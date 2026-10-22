using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.TurnContext;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Test-only owner that drives the three production turn-observation owners with the same state
/// projection a successful TurnKernel decision would publish. It stands in for the control loop
/// and the Kernel, and adds no decision logic of its own.
/// </summary>
internal sealed class AssistantTurnObservationStoreHarness
{
    private readonly HarnessTurnState _turnState = new();
    private readonly TurnContextStore _turnContexts;
    private readonly ITurnResourcePort _resources;
    private readonly SpeechCandidateOwner _candidates;
    private readonly BargeInValidationTurnOwner _validation;
    private readonly TestRecognitionCaptureOwner _recognitionCapture;
    private readonly RecognitionTurnOwner _recognitionTurn;
    private readonly ConfirmedBargeInApplier _applier;
    private readonly TurnObservationRelease _release;
    private long _nextTurnId = 1;
    private long _currentTurnId;

    public AssistantTurnObservationStoreHarness(
        BargeInDetector bargeIn,
        BargeInConfirmationSettings? listeningGate = null)
        : this(bargeIn, CreateUniformPolicy(bargeIn), listeningGate)
    {
    }

    private static BargeInConfirmationPolicy CreateUniformPolicy(BargeInDetector bargeIn)
    {
        ArgumentNullException.ThrowIfNull(bargeIn);
        return new BargeInConfirmationPolicy(bargeIn.DefaultSettings, bargeIn.DefaultSettings);
    }

    public AssistantTurnObservationStoreHarness(
        BargeInDetector bargeIn,
        BargeInConfirmationPolicy policy,
        BargeInConfirmationSettings? listeningGate = null)
    {
        ArgumentNullException.ThrowIfNull(bargeIn);
        ArgumentNullException.ThrowIfNull(policy);
        _turnContexts = new TurnContextStore(new OperationCorrelationGate());
        _resources = _turnContexts;
        _candidates = new SpeechCandidateOwner(bargeIn, policy, listeningGate, _turnState, _turnContexts);
        _validation = new BargeInValidationTurnOwner(_turnState);
        _recognitionCapture = new TestRecognitionCaptureOwner();
        _recognitionTurn = new RecognitionTurnOwner(
            _recognitionCapture,
            _turnState,
            _validation,
            _validation,
            _candidates);
        _applier = new ConfirmedBargeInApplier(
            _turnState,
            _candidates,
            _validation,
            _recognitionTurn,
            _turnContexts);
        _release = new TurnObservationRelease(
            _turnState,
            _recognitionTurn,
            _validation,
            _candidates,
            _turnContexts);
        _turnState.PublishNextOrdinal(_nextTurnId);
    }

    public AssistantState State => _turnState.ConversationState;

    /// <summary>Ownership identifiers the white-box mutation tests assert against directly.</summary>
    public long ActiveTurnId => _turnState.TurnId;

    public long ValidationTurnId => _validation.ValidationTurnId;

    public ITurnResourcePort Resources => _resources;

    /// <summary>
    /// Forces validation ownership the way a mutation test needs it, without going through the
    /// acoustic path that would normally establish it.
    /// </summary>
    public void SetValidationTurnId(long turnId)
        => SetValidation(turnId, _validation.InterruptedTurnId);

    public void SetValidationInterruptedTurnId(long interruptedTurnId)
        => SetValidation(_validation.ValidationTurnId, interruptedTurnId);

    private void SetValidation(long turnId, long interruptedTurnId)
    {
        if (turnId == 0)
        {
            _validation.ClearOwnership();
            return;
        }

        _validation.Begin(
            turnId,
            interruptedTurnId,
            _validation.ResponseCts,
            AssistantState.Speaking,
            initialAudio: null);
    }

    public bool IsIdleWithoutCandidate
        => State == AssistantState.Idle && !_candidates.IsCandidateActive && !_validation.HasValidation;

    public bool HasPendingBargeInCandidateOrValidation
        => _candidates.IsCandidateActive || _validation.HasValidation;

    public bool IsSpeakingWithoutPendingCandidate
        => State == AssistantState.Speaking && !_candidates.IsCandidateActive && !_validation.HasValidation;

    public bool IsBargeInAwaitingSpeaker => _candidates.IsAwaitingSpeaker;

    public void SynchronizeKernelState(AssistantState state, long turnId) => Synchronize(state, turnId);

    /// <summary>
    /// Observes a speech start without taking Listening ownership, the way the candidate owner is
    /// driven before the Kernel has ruled.
    /// </summary>
    public SpeechStartedDecision ObserveSpeechStartedWithoutCommit(bool enableBargeIn, long proposedTurnId)
    {
        _nextTurnId = proposedTurnId;
        _turnState.PublishNextOrdinal(proposedTurnId);
        return _validation.HasValidation
            ? new SpeechStartedDecision(
                SpeechStartedDecisionKind.Ignored,
                State,
                State,
                _validation.ValidationTurnId)
            : _candidates.ObserveSpeechStarted(enableBargeIn, proposedTurnId);
    }

    /// <summary>Overloads that pin the proposed turn id the control loop would have supplied.</summary>
    public SpeechStartedDecision OnSpeechStarted(bool enableBargeIn, long proposedTurnId)
    {
        _nextTurnId = proposedTurnId;
        _turnState.PublishNextOrdinal(proposedTurnId);
        return OnSpeechStarted(enableBargeIn);
    }

    public SpeakingDecision OnSpeaking(
        ReadOnlySpan<float> audio,
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId)
    {
        _nextTurnId = proposedTurnId;
        _turnState.PublishNextOrdinal(proposedTurnId);
        return OnSpeaking(audio, speakerGateStatus);
    }

    public bool TryObserveBargeInFromNearEndWindow(
        IReadOnlyList<float[]> initialAudio,
        BargeInDecision decision,
        long proposedTurnId,
        out SpeakingDecision confirmed)
    {
        _nextTurnId = proposedTurnId;
        _turnState.PublishNextOrdinal(proposedTurnId);
        return TryObserveBargeInFromNearEndWindow(initialAudio, decision, out confirmed);
    }

    public bool TryObserveWakeupInterrupt(long operationOwnerTurnId, out WakeupInterruptDecision decision)
        => _release.TryObserveWakeupInterrupt(operationOwnerTurnId, out decision);

    public SpeechStartedDecision OnSpeechStarted(bool enableBargeIn)
    {
        SpeechStartedDecision decision = _validation.HasValidation
            ? new SpeechStartedDecision(
                SpeechStartedDecisionKind.Ignored,
                State,
                State,
                _validation.ValidationTurnId)
            : _candidates.ObserveSpeechStarted(enableBargeIn, _nextTurnId);
        if (decision.Kind != SpeechStartedDecisionKind.StartListening)
            return decision;

        Advance(decision.TurnId);
        Synchronize(AssistantState.Listening, decision.TurnId);
        PrepareKernelListeningTurn(decision.TurnId);
        return decision;
    }

    public void PrepareKernelListeningTurn(long turnId)
    {
        if (!_turnState.IsCurrentTurn(turnId) || State != AssistantState.Listening)
            throw new InvalidOperationException("Kernel listening turn projection is not current.");

        _recognitionTurn.PrepareTurn();
        _validation.ClearOwnership();
        _candidates.Clear();
    }

    public SpeakingDecision OnSpeaking(
        ReadOnlySpan<float> audio,
        BargeInSpeakerGateStatus speakerGateStatus = BargeInSpeakerGateStatus.Match)
    {
        SpeakingDecision decision = _candidates.TryObserveCandidateAudio(
            audio,
            speakerGateStatus,
            _nextTurnId,
            out SpeakingDecision candidateDecision)
            ? candidateDecision
            : _recognitionTurn.ObserveAudio(audio);
        Apply(decision);
        CommitObservedDecision(decision);
        return decision;
    }

    public SpeakingDecision ResolveBargeInSpeakerGate(BargeInSpeakerGateStatus speakerGateStatus)
    {
        SpeakingDecision decision = _candidates.ResolveSpeakerGate(speakerGateStatus, _nextTurnId);
        Apply(decision);
        CommitObservedDecision(decision);
        return decision;
    }

    public bool TryObserveBargeInFromNearEndWindow(
        IReadOnlyList<float[]> initialAudio,
        BargeInDecision decision,
        out SpeakingDecision confirmed,
        bool identityAuthorized = false)
        => _applier.TryConfirmFromNearEndWindow(initialAudio, decision, identityAuthorized, out confirmed);

    public bool CommitConfirmedBargeIn(
        SpeakingDecision decision,
        IReadOnlyList<float[]>? initialAudio = null)
        => _applier.Apply(decision, initialAudio);

    public bool TryConfirmBargeInFromNearEndWindow(
        IReadOnlyList<float[]> initialAudio,
        BargeInDecision decision,
        out SpeakingDecision confirmed,
        bool identityAuthorized = false)
    {
        bool accepted = TryObserveBargeInFromNearEndWindow(initialAudio, decision, out confirmed, identityAuthorized);
        if (accepted)
            Apply(confirmed);
        if (accepted)
            accepted = _applier.Apply(confirmed, initialAudio);
        return accepted;
    }

    public bool TryObserveWakeupInterrupt(out WakeupInterruptDecision decision)
        => _release.TryObserveWakeupInterrupt(_nextTurnId, out decision);

    public bool CommitWakeupInterrupt(WakeupInterruptDecision decision)
        => _release.CommitWakeupInterrupt(decision);

    public bool TryInterruptForWakeup(out WakeupInterruptDecision decision)
    {
        bool accepted = TryObserveWakeupInterrupt(out decision);
        if (accepted)
            accepted = CommitWakeupInterrupt(decision);
        if (!accepted)
            return false;

        Advance(decision.TurnId);
        Synchronize(AssistantState.Idle, 0);
        return true;
    }

    public SpeechEndedDecision OnSpeechEnded(TimeSpan duration)
        => _candidates.TryDiscardOnSpeechEnded(duration, out SpeechEndedDecision discarded)
            ? discarded
            : _recognitionTurn.ObserveSpeechEnded(duration);

    public bool TryAttachStartedAsrSession(
        long turnId,
        string sessionId,
        Action<float[]> feedAudio,
        out bool speechAlreadyEnded,
        out TimeSpan speechDuration)
    {
        _recognitionCapture.BootstrapSink = feedAudio;
        try
        {
            return _recognitionTurn.TryAttachStartedSession(
                turnId,
                sessionId,
                out speechAlreadyEnded,
                out speechDuration);
        }
        finally
        {
            _recognitionCapture.BootstrapSink = null;
        }
    }

    public bool TryEnterRecognizing(
        long turnId,
        string sessionId,
        out AssistantState oldState,
        out bool isValidationTurn)
    {
        oldState = State;
        bool accepted = _recognitionTurn.CanEnterRecognizing(
            turnId,
            sessionId,
            out isValidationTurn);
        if (!accepted)
            return false;
        accepted = _recognitionTurn.CommitEnterRecognizing(
            turnId,
            sessionId,
            isValidationTurn);
        if (!accepted)
            return false;

        if (isValidationTurn)
            _validation.ClearPendingAudio();
        _candidates.Clear();
        if (!isValidationTurn)
            Synchronize(AssistantState.Recognizing, turnId);
        return true;
    }

    public bool ResetListeningAfterAsrStartFailure(
        long turnId,
        out AssistantState oldState,
        out AssistantState restoredState,
        out bool isValidationTurn)
    {
        oldState = State;
        isValidationTurn = _validation.IsValidating(turnId);
        bool accepted;
        if (isValidationTurn)
        {
            accepted = _validation.TryObserveAsrStartFailure(turnId, out restoredState) &&
                _validation.CommitAsrStartFailure(turnId);
            if (accepted)
                _candidates.Clear();
        }
        else
        {
            restoredState = AssistantState.Idle;
            accepted = _recognitionTurn.CanObserveActiveAsrStartFailure(turnId) &&
                _recognitionTurn.CommitAsrStartFailure(turnId);
            if (accepted)
                _candidates.Clear();
        }

        if (accepted)
            Synchronize(restoredState, isValidationTurn ? _currentTurnId : turnId);
        return accepted;
    }

    public bool TrySetResponseCtsForTurn(long turnId, CancellationTokenSource responseCts)
    {
        if (!_turnState.IsCurrentTurn(turnId))
            return false;

        _resources.GetOrCreate(turnId).ResponseCts = responseCts;
        return true;
    }

    public CancellationTokenSource? GetResponseCtsForTurn(long turnId)
        => _turnState.IsCurrentTurn(turnId) ? _resources.GetOrCreate(turnId).ResponseCts : null;

    public bool TrySetResponseCtsAndClearValidatedBargeIn(long turnId, CancellationTokenSource responseCts)
    {
        ArgumentNullException.ThrowIfNull(responseCts);
        if (!_turnState.IsCurrentTurn(turnId) || !_validation.IsValidating(turnId))
            return false;

        _resources.GetOrCreate(turnId).ResponseCts = responseCts;
        _validation.ClearOwnership();
        _validation.ClearPendingAudio();
        _recognitionCapture.ClearCapturePending(turnId);
        return true;
    }

    public bool TryCompleteAcceptedBargeInValidation(long turnId, out CancellationTokenSource? responseCts)
    {
        bool completed = _validation.TryCompleteAccepted(turnId, out responseCts);
        if (completed && responseCts is null)
            responseCts = _resources.GetOrCreate(turnId).ResponseCts;
        return completed;
    }

    public void CancelResponseCts()
    {
        CancellationTokenSource? responseCts = _resources.GetOrCreate(_turnState.TurnId).ResponseCts;
        try { _ = responseCts?.CancelAsync(); }
        catch (ObjectDisposedException) { }
    }

    public void DisposeResponseCts(CancellationTokenSource responseCts)
    {
        _resources.TryRemoveResponseCts(responseCts);
        try { responseCts.Dispose(); }
        catch { }
    }

    public bool TryClearResponseCtsForTurn(long turnId, CancellationTokenSource responseCts)
    {
        ArgumentNullException.ThrowIfNull(responseCts);
        TurnResourceSlot slot = _resources.GetOrCreate(turnId);
        if (!_turnState.IsCurrentTurn(turnId) || !ReferenceEquals(slot.ResponseCts, responseCts))
            return false;

        slot.ResponseCts = null;
        return true;
    }

    public string? TakeActiveAsrSessionAndClearPending()
    {
        string? active = _recognitionTurn.TakeActiveSessionAndClearPending();
        _validation.Clear();
        _candidates.Clear();
        return active;
    }

    public bool IsCurrentTurn(long turnId) => _turnState.IsCurrentTurn(turnId);

    public bool IsCurrentOrValidatingTurn(long turnId)
        => _turnState.IsCurrentTurn(turnId) || _validation.IsValidating(turnId);

    public bool IsValidatingBargeInTurn(long turnId) => _validation.IsValidating(turnId);

    public bool TryTransitionForTurn(
        long turnId,
        AssistantState to,
        out AssistantState oldState,
        out bool changed)
    {
        oldState = State;
        changed = false;
        if (!_turnState.IsCurrentTurn(turnId))
            return false;
        if (oldState == to)
            return true;

        changed = true;
        Synchronize(to, turnId);
        return true;
    }

    public bool TryCompleteTurn(
        long turnId,
        out AssistantState oldState,
        out bool changed,
        out bool completedInterruptedTurn)
    {
        bool accepted = _release.TryReleaseCompletedTurn(
            turnId,
            out oldState,
            out changed,
            out completedInterruptedTurn);
        if (accepted && changed)
            Synchronize(AssistantState.Idle, turnId);
        return accepted;
    }

    public bool TransitionTo(AssistantState to, out AssistantState oldState)
    {
        oldState = State;
        if (oldState == to)
            return false;

        Synchronize(to, _currentTurnId);
        return true;
    }

    public bool TryObserveValidatedBargeIn(
        long turnId,
        out AssistantState oldState,
        out CancellationTokenSource? responseToCancel,
        out long interruptedTurnId)
        => _validation.TryObserveValidated(turnId, out oldState, out responseToCancel, out interruptedTurnId);

    public bool TryAcceptValidatedBargeIn(
        long turnId,
        CancellationTokenSource? responseCts,
        bool clearValidationContext,
        out AssistantState oldState,
        out CancellationTokenSource? responseToCancel,
        out long interruptedTurnId)
    {
        bool accepted = TryObserveValidatedBargeIn(
            turnId,
            out oldState,
            out responseToCancel,
            out interruptedTurnId);
        if (!accepted)
            return false;

        Synchronize(AssistantState.Listening, turnId);
        _resources.GetOrCreate(turnId).ResponseCts = responseCts;
        accepted = _validation.CommitAccepted(turnId, responseCts, clearValidationContext);
        if (accepted)
            _recognitionTurn.ClearActiveSessionId();
        return accepted;
    }

    public bool TryObserveValidatedBargeInRejection(
        long turnId,
        out AssistantState oldState,
        out AssistantState restoredState)
        => _validation.TryObserveRejection(turnId, out oldState, out restoredState);

    public bool CommitValidatedBargeInRejected(long turnId)
    {
        if (!_validation.CommitRejected(turnId))
            return false;

        _candidates.Clear();
        return true;
    }

    public bool TryRejectValidatedBargeIn(
        long turnId,
        out AssistantState oldState,
        out AssistantState restoredState)
    {
        bool accepted = TryObserveValidatedBargeInRejection(turnId, out oldState, out restoredState) &&
            CommitValidatedBargeInRejected(turnId);
        if (accepted)
            Synchronize(restoredState, _currentTurnId);
        return accepted;
    }

    private void Apply(in SpeakingDecision decision)
    {
        if (decision.Kind is not (
                SpeakingDecisionKind.ConfirmedListening or
                SpeakingDecisionKind.ConfirmedBargeIn))
        {
            return;
        }

        Advance(decision.TurnId);
        if (decision.Kind == SpeakingDecisionKind.ConfirmedListening ||
            decision.OldState != AssistantState.Speaking)
        {
            Synchronize(AssistantState.Listening, decision.TurnId);
        }
    }

    private void CommitObservedDecision(in SpeakingDecision decision)
    {
        bool committed = decision.Kind switch
        {
            SpeakingDecisionKind.ConfirmedListening => _applier.ApplyConfirmedListening(decision),
            SpeakingDecisionKind.ConfirmedBargeIn => _applier.Apply(decision),
            _ => true,
        };
        if (!committed)
            throw new InvalidOperationException($"Could not commit observed decision '{decision.Kind}'.");
    }

    private void Advance(long turnId)
    {
        _nextTurnId = Math.Max(_nextTurnId, checked(turnId + 1));
        // The control loop derives the next proposed turn from the committed snapshot, so the
        // harness has to keep that ordinal in step with the ids it hands out.
        _turnState.PublishNextOrdinal(_nextTurnId);
    }

    private void Synchronize(AssistantState state, long turnId)
    {
        _currentTurnId = turnId;
        _turnState.Publish(state, turnId);
    }

    internal object? GetRecognitionTestField(string fieldName)
    {
        object? captureValue = _recognitionCapture.GetTestField(fieldName, _validation, _turnState.TurnId);
        if (fieldName == "_validatingBargeInPendingAudio")
        {
            var combined = new List<float[]>();
            if (GetPrivateValidationField("_pendingAudio") is List<float[]> validationPending)
                combined.AddRange(validationPending);
            if (captureValue is List<float[]> capturePending)
                combined.AddRange(capturePending);
            return combined;
        }

        return captureValue;
    }

    internal void SetRecognitionTestField(string fieldName, object? value)
        => _recognitionCapture.SetTestField(fieldName, value, _validation, _turnState.TurnId);

    internal object? GetValidationTestField(string fieldName)
        => fieldName switch
        {
            "_validatingBargeInResponseCts" => GetPrivateValidationField("_responseCts"),
            "_validatingBargeInPendingAudio" => GetPrivateValidationField("_pendingAudio"),
            "_validatingBargeInInterruptedTurnId" => _validation.InterruptedTurnId,
            _ => null,
        };

    internal void SetValidationTestField(string fieldName, object? value)
    {
        switch (fieldName)
        {
            case "_validatingBargeInResponseCts":
                SetPrivateValidationField("_responseCts", value as CancellationTokenSource);
                break;
            case "_validatingBargeInPendingAudio":
                List<float[]> pending = Assert.IsType<List<float[]>>(value);
                List<float[]> target = Assert.IsType<List<float[]>>(GetPrivateValidationField("_pendingAudio"));
                target.Clear();
                target.AddRange(pending);
                break;
            default:
                throw new InvalidOperationException($"Unknown validation test field '{fieldName}'.");
        }
    }

    private object? GetPrivateValidationField(string fieldName)
        => typeof(BargeInValidationTurnOwner)
            .GetField(fieldName, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?.GetValue(_validation);

    private void SetPrivateValidationField(string fieldName, object? value)
        => typeof(BargeInValidationTurnOwner)
            .GetField(fieldName, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?.SetValue(_validation, value);

    private sealed class TestRecognitionCaptureOwner : IRecognitionCaptureOwner
    {
        private readonly Dictionary<long, CaptureState> _states = [];

        public Action<float[]>? BootstrapSink { get; set; }

        public string? GetCaptureSessionId(long activeTurnId, long validationTurnId)
            => GetOrNull(activeTurnId)?.SessionId ??
                (validationTurnId != 0 ? GetOrNull(validationTurnId)?.SessionId : null);

        public SpeakingDecision ObserveCaptureAudio(
            long activeTurnId,
            AssistantState conversationState,
            ReadOnlySpan<float> audio,
            BargeInValidationCaptureSnapshot validation)
        {
            if (audio.IsEmpty)
                return default;
            long validationTurnId = validation.TurnId;
            long turnId = validationTurnId != 0 ? validationTurnId : activeTurnId;
            if (turnId <= 0)
                return default;
            CaptureState state = Get(turnId);
            if (state.SessionId is not null)
            {
                bool isValidation = validationTurnId != 0;
                return new SpeakingDecision(
                    SpeakingDecisionKind.FeedAsr,
                    state.SessionId,
                    isValidation ? validation.ResponseCancellation : null,
                    isValidation ? validation.InterruptedTurnId : 0,
                    conversationState,
                    isValidation ? validationTurnId : activeTurnId,
                    0, 0, 0, 0, 0, 0, 0,
                    default);
            }

            if (!state.SpeechEnded &&
                (validationTurnId != 0 || conversationState == AssistantState.Listening))
                state.PendingAudio.Add(audio.ToArray());
            return default;
        }

        public SpeechEndedDecision ObserveCaptureSpeechEnded(
            long activeTurnId,
            long validationTurnId,
            AssistantState conversationState,
            TimeSpan duration)
        {
            long turnId = validationTurnId != 0 ? validationTurnId : activeTurnId;
            if (turnId <= 0)
                return default;
            CaptureState state = Get(turnId);
            if (state.SessionId is not null)
                return Ready(state.SessionId, turnId, duration);
            if (validationTurnId == 0 && conversationState != AssistantState.Listening)
                return default;
            state.IsValidation = validationTurnId != 0;
            state.SpeechEnded = true;
            state.SpeechDuration = duration;
            return Waiting(turnId, duration);
        }

        public bool TryAttachCaptureSession(
            long turnId,
            string sessionId,
            bool isValidationTurn,
            IReadOnlyList<float[]> bootstrapAudio,
            out bool speechAlreadyEnded,
            out TimeSpan speechDuration)
        {
            speechAlreadyEnded = false;
            speechDuration = TimeSpan.Zero;
            CaptureState state = Get(turnId);
            if (state.SessionId is not null)
                return false;

            state.BootstrapAudio.AddRange(bootstrapAudio);
            while (true)
            {
                state.BootstrapAudio.AddRange(state.PendingAudio);
                state.PendingAudio.Clear();
                if (state.BootstrapAudio.Count == 0)
                    break;

                float[][] batch = [.. state.BootstrapAudio];
                state.BootstrapAudio.Clear();
                foreach (float[] audio in batch)
                    BootstrapSink?.Invoke(audio);
            }

            state.SessionId = sessionId;
            state.IsValidation = isValidationTurn;
            speechAlreadyEnded = state.SpeechEnded;
            speechDuration = state.SpeechDuration;
            state.SpeechEnded = false;
            state.SpeechDuration = TimeSpan.Zero;
            return true;
        }

        public bool CanEnterCaptureRecognizing(
            long turnId,
            string asrSessionId,
            AssistantState conversationState,
            out bool isValidationTurn)
        {
            isValidationTurn = false;
            CaptureState? state = GetOrNull(turnId);
            if (state is null || !StringComparer.Ordinal.Equals(state.SessionId, asrSessionId))
                return false;
            if (state.IsValidation)
            {
                isValidationTurn = true;
                return true;
            }
            if (conversationState is not (AssistantState.Listening or AssistantState.Recognizing))
                return false;
            return true;
        }

        public bool CommitEnterCaptureRecognizing(
            long turnId,
            string asrSessionId,
            bool isValidationTurn)
        {
            CaptureState? state = GetOrNull(turnId);
            if (state is null ||
                !StringComparer.Ordinal.Equals(state.SessionId, asrSessionId) ||
                state.IsValidation != isValidationTurn)
            {
                return false;
            }

            state.SessionId = null;
            if (!isValidationTurn)
            {
                state.PendingAudio.Clear();
                state.BootstrapAudio.Clear();
                state.SpeechEnded = false;
                state.SpeechDuration = TimeSpan.Zero;
            }
            return true;
        }

        public bool CanObserveCaptureStartFailure(long turnId, AssistantState conversationState)
            => turnId > 0 && conversationState is AssistantState.Listening or AssistantState.Idle;

        public bool CommitCaptureStartFailure(long turnId)
        {
            CaptureState? state = GetOrNull(turnId);
            if (state is null)
                return false;
            state.SessionId = null;
            state.PendingAudio.Clear();
            state.BootstrapAudio.Clear();
            state.SpeechEnded = false;
            state.SpeechDuration = TimeSpan.Zero;
            return true;
        }

        public void PrepareCaptureTurn(long turnId)
        {
            if (turnId <= 0)
                return;
            CaptureState state = Get(turnId);
            state.SessionId = null;
            state.PendingAudio.Clear();
            state.BootstrapAudio.Clear();
            state.SpeechEnded = false;
            state.SpeechDuration = TimeSpan.Zero;
        }

        public void ClearCaptureSessionId(long turnId)
        {
            if (GetOrNull(turnId) is { } state)
                state.SessionId = null;
        }

        public string? TakeAbandonedCaptureSession(long activeTurnId, long validationTurnId)
        {
            long selectedTurnId = activeTurnId != 0 &&
                GetOrNull(activeTurnId)?.SessionId is not null
                ? activeTurnId
                : validationTurnId != 0 ? validationTurnId : activeTurnId;
            CaptureState? state = GetOrNull(selectedTurnId);
            if (state is null)
                return null;
            string? session = state.SessionId;
            state.SessionId = null;
            state.PendingAudio.Clear();
            state.BootstrapAudio.Clear();
            state.SpeechEnded = false;
            state.SpeechDuration = TimeSpan.Zero;
            return session;
        }

        public void ClearCapturePending(long turnId)
        {
            if (GetOrNull(turnId) is not { } state)
                return;
            state.PendingAudio.Clear();
            state.BootstrapAudio.Clear();
        }

        public object? GetTestField(
            string fieldName,
            BargeInValidationTurnOwner validation,
            long activeTurnId)
        {
            if (fieldName == "_validatingBargeInAsrSessionId")
                return GetOrNull(validation.ValidationTurnId)?.SessionId;
            long turnId = fieldName.StartsWith("_validating", StringComparison.Ordinal)
                ? validation.ValidationTurnId
                : activeTurnId;
            CaptureState state = Get(turnId);
            return fieldName switch
            {
                "_pendingAsrAudio" or "_validatingBargeInPendingAudio" =>
                    PendingSnapshot(state),
                "_currentAsrSessionId" => GetOrNull(activeTurnId)?.SessionId,
                "_speechEndedBeforeAsrReady" or "_validatingBargeInSpeechEndedBeforeAsrReady" => state.SpeechEnded,
                "_pendingSpeechDuration" or "_validatingBargeInSpeechDuration" => state.SpeechDuration,
                _ => null,
            };
        }

        public void SetTestField(
            string fieldName,
            object? value,
            BargeInValidationTurnOwner validation,
            long activeTurnId)
        {
            if (fieldName == "_validatingBargeInAsrSessionId")
            {
                Get(validation.ValidationTurnId).SessionId = value as string;
                return;
            }
            long turnId = fieldName.StartsWith("_validating", StringComparison.Ordinal)
                ? validation.ValidationTurnId
                : activeTurnId;
            CaptureState state = Get(turnId);
            switch (fieldName)
            {
                case "_pendingAsrAudio":
                case "_validatingBargeInPendingAudio":
                    state.PendingAudio.Clear();
                    state.PendingAudio.AddRange(Assert.IsType<List<float[]>>(value));
                    break;
                case "_currentAsrSessionId": state.SessionId = value as string; break;
                case "_speechEndedBeforeAsrReady":
                case "_validatingBargeInSpeechEndedBeforeAsrReady":
                    state.SpeechEnded = Assert.IsType<bool>(value); break;
                case "_pendingSpeechDuration":
                case "_validatingBargeInSpeechDuration":
                    state.SpeechDuration = Assert.IsType<TimeSpan>(value); break;
            }
        }

        private CaptureState Get(long turnId)
        {
            if (!_states.TryGetValue(turnId, out CaptureState? state))
                _states.Add(turnId, state = new CaptureState());
            return state;
        }

        private CaptureState? GetOrNull(long turnId)
            => _states.TryGetValue(turnId, out CaptureState? state) ? state : null;

        private static List<float[]> PendingSnapshot(CaptureState state)
        {
            var pending = new List<float[]>(state.BootstrapAudio.Count + state.PendingAudio.Count);
            pending.AddRange(state.BootstrapAudio);
            pending.AddRange(state.PendingAudio);
            return pending;
        }

        private static SpeechEndedDecision Waiting(long turnId, TimeSpan duration)
            => new(SpeechEndedDecisionKind.WaitingForAsrStart, null, turnId, duration, 0, 0, 0, 0, default);

        private static SpeechEndedDecision Ready(string sessionId, long turnId, TimeSpan duration)
            => new(SpeechEndedDecisionKind.ReadyForRecognition, sessionId, turnId, duration, 0, 0, 0, 0, default);

        private sealed class CaptureState
        {
            public List<float[]> PendingAudio { get; } = [];
            public List<float[]> BootstrapAudio { get; } = [];
            public string? SessionId { get; set; }
            public bool IsValidation { get; set; }
            public bool SpeechEnded { get; set; }
            public TimeSpan SpeechDuration { get; set; }
        }
    }

    /// <summary>Stands in for the control loop's committed snapshot.</summary>
    private sealed class HarnessTurnState : ITurnStateView
    {
        private TurnSnapshot _snapshot = TurnSnapshot.Initial;

        public TurnSnapshot Snapshot => Volatile.Read(ref _snapshot);

        public AssistantState ConversationState => Snapshot.ConversationState;

        public long TurnId => Snapshot.TurnId;

        public long NextTurnOrdinal => Snapshot.NextTurnOrdinal;

        public long ValidationTurnId => Snapshot.ValidationTurnId;

        public bool IsCurrentTurn(long turnId) => Snapshot.TurnId == turnId;

        public void Publish(AssistantState state, long turnId)
            => Volatile.Write(
                ref _snapshot,
                Snapshot with { ConversationState = state, TurnId = turnId });

        public void PublishNextOrdinal(long nextTurnOrdinal)
            => Volatile.Write(ref _snapshot, Snapshot with { NextTurnOrdinal = nextTurnOrdinal });
    }
}
