using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Runtime.TurnContext;

/// <summary>
/// Releases the per-turn observation context held across the three owners when a turn ends, for
/// both exits that abandon one: normal completion and a wake-word interruption. It reads the
/// committed snapshot and never decides a transition.
/// </summary>
internal sealed class TurnObservationRelease
{
    private readonly ITurnStateView _turnState;
    private readonly RecognitionTurnOwner _recognitionTurn;
    private readonly BargeInValidationTurnOwner _validation;
    private readonly SpeechCandidateOwner _candidates;
    private readonly ITurnResourcePort _resources;

    public TurnObservationRelease(
        ITurnStateView turnState,
        RecognitionTurnOwner recognitionTurn,
        BargeInValidationTurnOwner validation,
        SpeechCandidateOwner candidates,
        ITurnResourcePort resources)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _recognitionTurn = recognitionTurn ?? throw new ArgumentNullException(nameof(recognitionTurn));
        _validation = validation ?? throw new ArgumentNullException(nameof(validation));
        _candidates = candidates ?? throw new ArgumentNullException(nameof(candidates));
        _resources = resources ?? throw new ArgumentNullException(nameof(resources));
    }

    public bool TryReleaseCompletedTurn(
        long turnId,
        out AssistantState oldState,
        out bool changed,
        out bool completedInterruptedTurn)
    {
        oldState = _turnState.ConversationState;
        changed = false;
        completedInterruptedTurn = false;
        if (!_turnState.IsCurrentTurn(turnId))
            return false;

        // Playback completion and the response pipeline finally block can converge on the same
        // turn. If the first completion already moved the state to Idle while a barge-in
        // validation is still outstanding, the idempotent second completion must retain that
        // context so the runtime does not prematurely return to wake-word mode.
        completedInterruptedTurn = _validation.TryMarkInterruptedTurnCompleted(turnId);
        if (oldState == AssistantState.Idle)
            return true;

        _recognitionTurn.PrepareTurn();
        if (!completedInterruptedTurn)
        {
            if (_validation.IsValidating(turnId))
                _validation.ClearOwnership();
            _validation.ClearPendingAudio();
            _candidates.Clear();
        }

        changed = true;
        return true;
    }

    /// <summary>
    /// Snapshots what a wake-word interruption would abandon. Wake word is a high-confidence
    /// command to start over, so it applies even while regular barge-in is enabled.
    /// </summary>
    public bool TryObserveWakeupInterrupt(long operationOwnerTurnId, out WakeupInterruptDecision decision)
    {
        decision = default;
        AssistantState oldState = _turnState.ConversationState;
        if (oldState is not (AssistantState.Listening or AssistantState.Recognizing or
                AssistantState.Thinking or AssistantState.Speaking) &&
            !_validation.HasValidation)
        {
            return false;
        }

        if (operationOwnerTurnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationOwnerTurnId));

        long validationInterrupted = _validation.InterruptedTurnId;
        decision = new WakeupInterruptDecision(
            oldState,
            operationOwnerTurnId,
            validationInterrupted != 0 ? validationInterrupted : _turnState.TurnId,
            _validation.ResponseCts ?? PeekResponseCts(_turnState.TurnId),
            _recognitionTurn.ActiveSessionId);
        return true;
    }

    public bool CommitWakeupInterrupt(WakeupInterruptDecision decision)
    {
        if (decision.TurnId <= 0 ||
            (!_turnState.IsCurrentTurn(decision.InterruptedTurnId) &&
             _validation.InterruptedTurnId != decision.InterruptedTurnId))
        {
            return false;
        }

        if (decision.ResponseToCancel is not null)
            _resources.TryRemoveResponseCts(decision.ResponseToCancel);
        if (_resources.TryGet(_turnState.TurnId, out TurnResourceSlot slot))
            slot.ResponseCts = null;
        _recognitionTurn.PrepareTurn();
        _validation.Clear();
        _candidates.Clear();
        return true;
    }

    private CancellationTokenSource? PeekResponseCts(long turnId)
        => _resources.TryGet(turnId, out TurnResourceSlot slot) ? slot.ResponseCts : null;
}
