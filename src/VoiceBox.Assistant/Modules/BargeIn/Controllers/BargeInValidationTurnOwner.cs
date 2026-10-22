using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Owns the playback barge-in validation turn: which turn is being validated, which turn it would
/// interrupt, and the state to restore if validation is rejected. The Kernel owns the public state
/// transition; this owner only keeps the module-private ownership context around it.
/// </summary>
internal sealed class BargeInValidationTurnOwner :
    IBargeInValidationTurnView,
    IBargeInValidationOwnershipPort,
    IValidationCapturePreRollPort
{
    private readonly object _lock = new();
    private readonly ITurnStateView _turnState;
    private long _validationTurnId;
    private long _interruptedTurnId;
    private AssistantState _previousState;
    private CancellationTokenSource? _responseCts;
    private readonly List<float[]> _pendingAudio = [];

    public BargeInValidationTurnOwner(ITurnStateView turnState)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
    }

    public long ValidationTurnId
    {
        get { lock (_lock) return _validationTurnId; }
    }

    public BargeInValidationCaptureSnapshot CaptureSnapshot
    {
        get
        {
            lock (_lock)
            {
                return new BargeInValidationCaptureSnapshot(
                    _validationTurnId,
                    _interruptedTurnId,
                    _responseCts);
            }
        }
    }

    public bool HasValidation
    {
        get { lock (_lock) return _validationTurnId != 0; }
    }

    public bool IsValidating(long turnId)
    {
        lock (_lock)
            return _validationTurnId == turnId;
    }

    public long InterruptedTurnId
    {
        get { lock (_lock) return _interruptedTurnId; }
    }

    public CancellationTokenSource? ResponseCts
    {
        get { lock (_lock) return _responseCts; }
    }

    /// <summary>
    /// 播放期候选先进入验证回合: 保留候选音频给 ASR, 但不抢占当前活跃回复。
    /// ASR 证明文本有效时立即升级；验证超时时也会停止旧回复并取得回合所有权，
    /// 但保留验证上下文，继续等待当前 ASR 的最终结果。
    /// </summary>
    public void Begin(
        long validationTurnId,
        long interruptedTurnId,
        CancellationTokenSource? responseCts,
        AssistantState previousState,
        IReadOnlyList<float[]>? initialAudio)
    {
        lock (_lock)
        {
            _validationTurnId = validationTurnId;
            _interruptedTurnId = interruptedTurnId;
            _responseCts = responseCts;
            _previousState = previousState;
            ClearPendingAudioLocked();
            if (initialAudio is null)
                return;

            foreach (float[] chunk in initialAudio)
            {
                if (chunk.Length > 0)
                    _pendingAudio.Add(chunk.ToArray());
            }
        }
    }

    /// <summary>Observes a validated barge-in outcome without changing local ownership.</summary>
    public bool TryObserveValidated(
        long turnId,
        out AssistantState oldState,
        out CancellationTokenSource? responseToCancel,
        out long interruptedTurnId)
    {
        lock (_lock)
        {
            oldState = _turnState.ConversationState;
            responseToCancel = null;
            interruptedTurnId = 0;
            if (_validationTurnId != turnId)
                return false;

            responseToCancel = _responseCts;
            interruptedTurnId = _interruptedTurnId;
            return true;
        }
    }

    /// <summary>Applies an accepted validation outcome after Kernel commit.</summary>
    public bool CommitAccepted(long turnId, CancellationTokenSource? responseCts, bool clearValidationContext)
    {
        lock (_lock)
        {
            if (_validationTurnId != turnId)
                return false;

            _responseCts = responseCts;
            _interruptedTurnId = 0;
            _previousState = default;
            if (clearValidationContext)
            {
                ClearOwnershipLocked();
                ClearPendingAudioLocked();
            }

            return true;
        }
    }

    /// <summary>Observes a rejected validation outcome without clearing local ownership.</summary>
    public bool TryObserveRejection(long turnId, out AssistantState oldState, out AssistantState restoredState)
    {
        lock (_lock)
        {
            oldState = _turnState.ConversationState;
            restoredState = oldState;
            if (_validationTurnId != turnId)
                return false;

            restoredState = _previousState;
            return true;
        }
    }

    /// <summary>Applies a rejected validation outcome after Kernel commit.</summary>
    public bool CommitRejected(long turnId)
    {
        lock (_lock)
        {
            if (_validationTurnId != turnId)
                return false;

            ClearOwnershipLocked();
            ClearPendingAudioLocked();
            return true;
        }
    }

    /// <summary>
    /// Releases validation ownership for a turn the Kernel already accepted, returning the
    /// response source the runtime must keep cancelling.
    /// </summary>
    public bool TryCompleteAccepted(long turnId, out CancellationTokenSource? responseCts)
    {
        lock (_lock)
        {
            responseCts = null;
            if (!_turnState.IsCurrentTurn(turnId))
                return false;

            responseCts = _responseCts;
            if (_validationTurnId == turnId)
            {
                ClearOwnershipLocked();
                ClearPendingAudioLocked();
            }

            return true;
        }
    }

    /// <summary>
    /// Restores the previous state for an ASR-start failure on the validation turn.
    /// </summary>
    public bool TryObserveAsrStartFailure(long turnId, out AssistantState restoredState)
    {
        lock (_lock)
        {
            restoredState = _turnState.ConversationState;
            if (_validationTurnId != turnId)
                return false;

            restoredState = _previousState;
            return true;
        }
    }

    public bool CommitAsrStartFailure(long turnId)
    {
        lock (_lock)
        {
            if (_validationTurnId != turnId)
                return false;

            ClearOwnershipLocked();
            ClearPendingAudioLocked();
            return true;
        }
    }

    /// <summary>
    /// Marks that the completing turn is the one this validation interrupted, so wake-word return
    /// is suppressed until the validation itself resolves.
    /// </summary>
    public bool TryMarkInterruptedTurnCompleted(long turnId)
    {
        lock (_lock)
        {
            if (_validationTurnId == 0 || _interruptedTurnId != turnId)
                return false;

            _previousState = AssistantState.Idle;
            return true;
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            ClearOwnershipLocked();
            ClearPendingAudioLocked();
        }
    }

    /// <summary>Releases validation ownership without touching the buffered validation audio.</summary>
    public void ClearOwnership()
    {
        lock (_lock)
            ClearOwnershipLocked();
    }

    public void ClearPendingAudio()
    {
        lock (_lock)
            ClearPendingAudioLocked();
    }

    /// <summary>
    /// Transfers validation pre-roll to the RecognitionModule at the session ownership boundary.
    /// The returned buffers are removed before provider admission, so the validation owner never
    /// retains a second PCM queue after attach.
    /// </summary>
    public bool TryTakePendingAudio(out List<float[]> audio)
    {
        lock (_lock)
        {
            audio = [];
            if (_pendingAudio.Count == 0)
                return false;

            audio.AddRange(_pendingAudio);
            _pendingAudio.Clear();
            return true;
        }
    }

    private void ClearOwnershipLocked()
    {
        _validationTurnId = 0;
        _interruptedTurnId = 0;
        _responseCts = null;
        _previousState = default;
    }

    private void ClearPendingAudioLocked()
    {
        _pendingAudio.Clear();
    }
}
