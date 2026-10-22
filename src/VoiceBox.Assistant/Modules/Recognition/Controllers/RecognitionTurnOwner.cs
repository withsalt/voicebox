using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Modules.Recognition.Controllers;

/// <summary>
/// Routes committed turn observations to the recognition module. This owner deliberately has no
/// ASR/session/PCM storage of its own; <see cref="RecognitionModule"/> is the sole capture owner.
/// </summary>
internal sealed class RecognitionTurnOwner : IRecognitionBoundaryPort
{
    private readonly IRecognitionCaptureOwner _capture;
    private readonly ITurnStateView _turnState;
    private readonly IBargeInValidationTurnView _validation;
    private readonly IValidationCapturePreRollPort _validationPreRoll;
    private readonly ICandidatePreRollSource _candidatePreRoll;

    public RecognitionTurnOwner(
        IRecognitionCaptureOwner capture,
        ITurnStateView turnState,
        IBargeInValidationTurnView validation,
        IValidationCapturePreRollPort validationPreRoll,
        ICandidatePreRollSource candidatePreRoll)
    {
        _capture = capture ?? throw new ArgumentNullException(nameof(capture));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _validation = validation ?? throw new ArgumentNullException(nameof(validation));
        _validationPreRoll = validationPreRoll ?? throw new ArgumentNullException(nameof(validationPreRoll));
        _candidatePreRoll = candidatePreRoll ?? throw new ArgumentNullException(nameof(candidatePreRoll));
    }

    public string? ActiveSessionId
    {
        get
        {
            TurnSnapshot turn = _turnState.Snapshot;
            BargeInValidationCaptureSnapshot validation = _validation.CaptureSnapshot;
            return _capture.GetCaptureSessionId(turn.TurnId, validation.TurnId);
        }
    }

    /// <summary>
    /// Routes capture audio that no candidate owns to the recognition module. The module decides
    /// whether to feed an admitted session or retain bounded pre-session PCM.
    /// </summary>
    public SpeakingDecision ObserveAudio(ReadOnlySpan<float> audio)
    {
        TurnSnapshot turn = _turnState.Snapshot;
        BargeInValidationCaptureSnapshot validation = _validation.CaptureSnapshot;
        return _capture.ObserveCaptureAudio(
            turn.TurnId,
            turn.ConversationState,
            audio,
            validation);
    }

    public SpeechEndedDecision ObserveSpeechEnded(TimeSpan duration)
        => _capture.ObserveCaptureSpeechEnded(
            _turnState.TurnId,
            _validation.ValidationTurnId,
            _turnState.ConversationState,
            duration);

    /// <summary>
    /// Attaches a provider session after its RecognitionStarted signal is admitted. Validation
    /// pre-roll crosses this boundary exactly once and is then owned by RecognitionModule.
    /// </summary>
    public bool TryAttachStartedSession(
        long turnId,
        string sessionId,
        out bool speechAlreadyEnded,
        out TimeSpan speechDuration)
    {
        speechAlreadyEnded = false;
        speechDuration = TimeSpan.Zero;

        TurnSnapshot turn = _turnState.Snapshot;
        BargeInValidationCaptureSnapshot validation = _validation.CaptureSnapshot;
        bool isValidationTurn = validation.TurnId == turnId;
        if (!isValidationTurn &&
            (turn.TurnId != turnId || turn.ConversationState != AssistantState.Listening))
        {
            return false;
        }

        List<float[]> bootstrapAudio = [];
        if (isValidationTurn && _validationPreRoll.TryTakePendingAudio(out List<float[]> validationAudio))
            bootstrapAudio.AddRange(validationAudio);
        _candidatePreRoll.DrainPendingAudioInto(bootstrapAudio);

        return _capture.TryAttachCaptureSession(
            turnId,
            sessionId,
            isValidationTurn,
            bootstrapAudio,
            out speechAlreadyEnded,
            out speechDuration);
    }

    public bool CanEnterRecognizing(long turnId, string asrSessionId, out bool isValidationTurn)
    {
        if (!_turnState.IsCurrentTurn(turnId) && !_validation.IsValidating(turnId))
        {
            isValidationTurn = false;
            return false;
        }

        return _capture.CanEnterCaptureRecognizing(
            turnId,
            asrSessionId,
            _turnState.ConversationState,
            out isValidationTurn);
    }

    public bool CommitEnterRecognizing(
        long turnId,
        string asrSessionId,
        bool isValidationTurn)
        => _capture.CommitEnterCaptureRecognizing(turnId, asrSessionId, isValidationTurn);

    public bool CanObserveActiveAsrStartFailure(long turnId)
        => _turnState.IsCurrentTurn(turnId) &&
            _capture.CanObserveCaptureStartFailure(turnId, _turnState.ConversationState);

    public bool CommitAsrStartFailure(long turnId)
        => _turnState.IsCurrentTurn(turnId) && _capture.CommitCaptureStartFailure(turnId);

    public void PrepareTurn()
        => _capture.PrepareCaptureTurn(_turnState.TurnId);

    /// <summary>
    /// Detaches the interrupted active turn's capture session. Once a timeout has promoted the
    /// validation turn to the current turn, its session must remain attached until the endpoint
    /// starts recognition; otherwise that later endpoint would wait forever for a second start.
    /// </summary>
    public void ClearActiveSessionId()
    {
        TurnSnapshot turn = _turnState.Snapshot;
        long validationTurnId = _validation.ValidationTurnId;
        if (validationTurnId != 0 && validationTurnId == turn.TurnId)
            return;

        _capture.ClearCaptureSessionId(turn.TurnId);
    }

    public string? TakeActiveSessionAndClearPending()
        => _capture.TakeAbandonedCaptureSession(
            _turnState.TurnId,
            _validation.ValidationTurnId);
}
