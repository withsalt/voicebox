using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Applies an accepted interruption across the three turn-observation owners: it opens the playback
/// interruption turn whose recognition runs on the whole utterance, or hands a Thinking-state
/// interruption straight to recognition. It reads the committed snapshot and never changes it — the
/// Kernel decision is committed by the caller beforehand.
/// </summary>
internal sealed class ConfirmedBargeInApplier
{
    private readonly ITurnStateView _turnState;
    private readonly SpeechCandidateOwner _candidates;
    private readonly BargeInValidationTurnOwner _validation;
    private readonly IRecognitionTurnLifecyclePort _recognitionTurn;
    private readonly ITurnResourcePort _resources;

    public ConfirmedBargeInApplier(
        ITurnStateView turnState,
        SpeechCandidateOwner candidates,
        BargeInValidationTurnOwner validation,
        IRecognitionTurnLifecyclePort recognitionTurn,
        ITurnResourcePort resources)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _candidates = candidates ?? throw new ArgumentNullException(nameof(candidates));
        _validation = validation ?? throw new ArgumentNullException(nameof(validation));
        _recognitionTurn = recognitionTurn ?? throw new ArgumentNullException(nameof(recognitionTurn));
        _resources = resources ?? throw new ArgumentNullException(nameof(resources));
    }

    /// <summary>
    /// Confirms a render-aware near-end candidate as an interruption of the active playback turn.
    /// With KWS disabled the DTD decision must be an acoustic <c>Interrupt</c>; with KWS enabled the
    /// candidate's own speaker Match is the authorization and the DTD need not have reached
    /// <c>Interrupt</c> — the two are evaluated in parallel on the same window.
    /// </summary>
    public bool TryConfirmFromNearEndWindow(
        IReadOnlyList<float[]> initialAudio,
        BargeInDecision nearEndDecision,
        bool identityAuthorized,
        out SpeakingDecision decision)
    {
        decision = default;
        AssistantState state = _turnState.ConversationState;
        bool acousticallyConfirmed = nearEndDecision.Action == BargeInAction.Interrupt;
        if ((!acousticallyConfirmed && !identityAuthorized) ||
            state != AssistantState.Speaking ||
            _validation.HasValidation ||
            !_candidates.TryGetNearEndSettings(state, out BargeInConfirmationSettings settings))
        {
            return false;
        }

        long proposedTurnId = _turnState.NextTurnOrdinal;
        if (proposedTurnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialAudio));

        long interruptedTurnId = _turnState.TurnId;
        AudioStats stats = ComputeAudioStats(initialAudio);
        decision = new SpeakingDecision(
            SpeakingDecisionKind.ConfirmedBargeIn,
            AsrSessionId: null,
            _resources.GetOrCreate(interruptedTurnId).ResponseCts,
            interruptedTurnId,
            state,
            proposedTurnId,
            stats.SampleCount,
            stats.Rms,
            stats.SampleCount,
            stats.Rms,
            settings.MinDurationMs,
            settings.MinRms,
            settings.InitialSuppressionMs,
            DiagnosticsFromNearEndWindow(nearEndDecision));
        return true;
    }

    public bool Apply(SpeakingDecision decision, IReadOnlyList<float[]>? initialAudio = null)
    {
        if (decision.Kind != SpeakingDecisionKind.ConfirmedBargeIn || decision.TurnId <= 0)
            return false;

        if (decision.OldState != AssistantState.Speaking)
        {
            if (!_candidates.CanMarkConfirmed(SpeechCandidateKind.BargeIn))
                return false;

            // 候选确认后开启新回合; 候选累积的音频保留, 供 ASR session 接管后补喂。
            _recognitionTurn.PrepareTurn();
            _validation.ClearOwnership();
            _candidates.TryMarkConfirmed(SpeechCandidateKind.BargeIn);
            return true;
        }

        if (_validation.IsValidating(decision.TurnId))
            return true;
        if (_turnState.ConversationState != AssistantState.Speaking ||
            _validation.HasValidation ||
            !_turnState.IsCurrentTurn(decision.InterruptedTurnId))
        {
            return false;
        }

        if (initialAudio is not null)
        {
            if (_candidates.IsCandidateActive)
                return false;
            _validation.Begin(
                decision.TurnId,
                decision.InterruptedTurnId,
                decision.ResponseToCancel,
                decision.OldState,
                initialAudio);
            _recognitionTurn.ClearActiveSessionId();
            _candidates.MarkNearEndConfirmed();
            return true;
        }

        if (!_candidates.CanMarkConfirmed(SpeechCandidateKind.BargeIn))
            return false;

        _validation.Begin(
            decision.TurnId,
            decision.InterruptedTurnId,
            decision.ResponseToCancel,
            decision.OldState,
            initialAudio: null);
        _recognitionTurn.ClearActiveSessionId();
        _candidates.TryMarkConfirmed(SpeechCandidateKind.BargeIn);
        return true;
    }

    /// <summary>
    /// Applies a confirmed Idle listening candidate: the turn is already the Kernel's, so only the
    /// observation context moves over.
    /// </summary>
    public bool ApplyConfirmedListening(SpeakingDecision decision)
    {
        if (decision.Kind != SpeakingDecisionKind.ConfirmedListening ||
            decision.TurnId <= 0 ||
            !_candidates.CanMarkConfirmed(SpeechCandidateKind.Listening))
        {
            return false;
        }

        _recognitionTurn.PrepareTurn();
        _validation.ClearOwnership();
        _candidates.TryMarkConfirmed(SpeechCandidateKind.Listening);
        return true;
    }

    private static BargeInDecisionDiagnostics DiagnosticsFromNearEndWindow(BargeInDecision decision)
        => new(
            decision.FrameCount,
            decision.AcceptedFrameCount,
            decision.RejectedFrameCount,
            decision.EchoLikeFrameCount,
            decision.AcceptedDurationMs,
            decision.RequiredDurationMs,
            decision.ScoreDurationMs,
            decision.PeakScoreDurationMs,
            decision.LastFrameRms,
            decision.LastRequiredRms,
            decision.MinRequiredRms,
            decision.MaxRequiredRms,
            decision.Reason);

    private static AudioStats ComputeAudioStats(IReadOnlyList<float[]> chunks)
    {
        long samples = 0;
        double energy = 0;
        foreach (float[] chunk in chunks)
        {
            samples += chunk.Length;
            for (int i = 0; i < chunk.Length; i++)
            {
                double sample = chunk[i];
                energy += sample * sample;
            }
        }

        return samples <= 0 ? default : new AudioStats(samples, Math.Sqrt(energy / samples));
    }

    private readonly record struct AudioStats(long SampleCount, double Rms);
}
