using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.TurnContext;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Applies confirmed interruptions to the conversation. When speaker checks are enabled, only the
/// candidate's Match stops playback; otherwise acoustic confirmation does. Whole-utterance
/// recognition then decides whether to start a response.
/// </summary>
internal sealed class BargeInTurnCoordinator :
    IBargeInTurnSupportPort,
    IActiveNearEndConversationView
{
    private readonly ITurnStateView _turnState;
    private readonly SpeechCandidateOwner _candidates;
    private readonly ConfirmedBargeInApplier _applier;
    private readonly BargeInValidationTurnOwner _validation;
    private readonly IRecognitionTurnLifecyclePort _recognitionTurn;
    private readonly BargeInDiagnosticsCoordinator _diagnostics;
    private readonly BargeInDecisionJournal _journal;
    private readonly IPlaybackInterruptionPort _playbackInterruption;
    private readonly IConversationRollbackPort _history;
    private readonly TurnContextStore _turnContexts;
    private readonly ITurnResourcePort _resources;
    // Bound by composition once the recognition owner exists; see BindRecognition.
    private IRecognitionPreparationPort? _boundRecognitionPreparation;
    private readonly AssistantControlSignalGateway _controlSignals;
    private readonly RuntimeRunScope _runScope;
    private readonly InputGate _gate;
    private readonly IWakeupModule? _wakeup;
    private readonly VoiceAssistantOptions _options;
    private readonly VoiceAssistantConversationOptions _conversationOptions;
    private readonly bool _identityRequired;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;
    private readonly bool _wakeupInstalled;
    private readonly ILogger _logger;
    private TurnCompletionCoordinator? _turnCompletion;

    public const string SpeakerMatchReason = "speaker-match";
    public const string AudioConfirmedReason = "audio-confirmed";

    public BargeInTurnCoordinator(
        ITurnStateView turnState,
        SpeechCandidateOwner candidates,
        ConfirmedBargeInApplier applier,
        BargeInValidationTurnOwner validation,
        IRecognitionTurnLifecyclePort recognitionTurn,
        BargeInDiagnosticsCoordinator diagnostics,
        BargeInDecisionJournal journal,
        IPlaybackInterruptionPort playbackInterruption,
        IConversationRollbackPort history,
        TurnContextStore turnContexts,
        ITurnResourcePort resources,
        AssistantControlSignalGateway controlSignals,
        RuntimeRunScope runScope,
        InputGate gate,
        IWakeupModule? wakeup,
        VoiceAssistantOptions options,
        bool identityRequired,
        ILogger logger,
        VoiceAssistantBenchmarkProbe? benchmarkProbe = null)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _candidates = candidates ?? throw new ArgumentNullException(nameof(candidates));
        _applier = applier ?? throw new ArgumentNullException(nameof(applier));
        _validation = validation ?? throw new ArgumentNullException(nameof(validation));
        _recognitionTurn = recognitionTurn ?? throw new ArgumentNullException(nameof(recognitionTurn));
        _diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        _journal = journal ?? throw new ArgumentNullException(nameof(journal));
        _playbackInterruption = playbackInterruption ?? throw new ArgumentNullException(nameof(playbackInterruption));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _resources = resources ?? throw new ArgumentNullException(nameof(resources));
        _controlSignals = controlSignals ?? throw new ArgumentNullException(nameof(controlSignals));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _wakeup = wakeup;
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _conversationOptions = _options.Conversation;
        _identityRequired = identityRequired;
        _benchmarkProbe = benchmarkProbe;
        _wakeupInstalled = wakeup is not null;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>KWS enabled: every playback interruption needs the current candidate's speaker Match.</summary>
    public bool IsIdentityRequired => _identityRequired;

    /// <summary>
    /// Composition binds the recognition owner's preparation port once it exists. The two owners are mutually
    /// reachable — an accepted interruption defers a recognition transition, and a failed session
    /// start unwinds a validation — so one of the two edges has to be closed here. Both sides of the
    /// cycle are narrow typed ports, never each other's concrete type.
    /// </summary>
    public void BindRecognition(IRecognitionPreparationPort recognitionPreparation)
    {
        _boundRecognitionPreparation = recognitionPreparation
            ?? throw new ArgumentNullException(nameof(recognitionPreparation));
    }

    private IRecognitionPreparationPort _recognitionPreparation
        => _boundRecognitionPreparation ?? throw new InvalidOperationException(
            "BindRecognition must be called before the barge-in owner handles a decision.");

    public void Bind(TurnCompletionCoordinator turnCompletion)
    {
        _turnCompletion = turnCompletion ?? throw new ArgumentNullException(nameof(turnCompletion));
    }

    public BargeInSemanticPlan ObserveConfirmedBargeIn(
        SpeakingDecision decision,
        BargeInTriggerSource source,
        IReadOnlyList<float[]>? initialAudio,
        BargeInIdentityContext identity)
    {
        if (_identityRequired && decision.OldState == AssistantState.Speaking && !identity.PermitsInterruption)
        {
            // Never reaches the Kernel: with KWS enabled an unauthorized candidate cannot even be
            // requested as an interruption, so it cannot produce any playback side effect.
            _logger.LogDebug(
                "播放期候选未获当前窗口 Speaker 授权, 不提交打断: turnId={TurnId}, source={Source}.",
                decision.TurnId,
                source);
            return default;
        }

        TurnAuthorizationStatus authorization = _identityRequired
            ? TurnAuthorizationStatus.PendingSpeaker
            : TurnAuthorizationStatus.NotRequired;
        TurnSignal? signal = decision.OldState switch
        {
            AssistantState.Speaking => TurnSignal.InterruptionRequested(
                decision.InterruptedTurnId,
                decision.TurnId,
                publishHeard: true,
                source,
                decision.ConfirmedBargeInDiagnostics,
                hasBargeInContext: true,
                authorization: authorization,
                candidateId: identity.CandidateId),
            AssistantState.Thinking => TurnSignal.InterruptionRequested(
                decision.InterruptedTurnId,
                decision.TurnId,
                source: source,
                diagnostics: decision.ConfirmedBargeInDiagnostics,
                hasBargeInContext: true),
            _ => null,
        };

        return BargeInSemanticPlan.InterruptionRequested(
            signal,
            decision,
            source,
            initialAudio,
            identity,
            decision.ConfirmedBargeInDiagnostics);
    }

    public BargeInSemanticPlan CommitInterruptionRequested(
        BargeInSemanticPlan plan,
        KernelCommitResult kernel)
    {
        if (!kernel.Accepted)
            return default;

        SpeakingDecision decision = plan.Decision;
        _benchmarkProbe?.ObserveControlTrace(
            "barge-in.confirmed",
            decision.TurnId,
            detail: plan.Source.ToString());
        _diagnostics.RecordConfirmed(decision, plan.Source);
        _wakeup?.MarkUserActivity();
        if (!_applier.Apply(decision, plan.InitialAudio))
        {
            throw new InvalidOperationException(
                "Confirmed barge-in observation could not be committed after Kernel acceptance.");
        }

        if (decision.OldState == AssistantState.Thinking)
        {
            _playbackInterruption.InterruptActiveResponse(decision.InterruptedTurnId, decision.ResponseToCancel);
            _history.RemoveUnansweredUserMessage(decision.InterruptedTurnId);
            _recognitionPreparation.PrepareRecognitionSession(decision.TurnId);
            return default;
        }

        _logger.LogDebug(
            "Barge-in candidate committed: source={Source}, turnId={TurnId}, interruptedTurnId={InterruptedTurnId}, " +
            "identity={IdentityMode}, candidateId={CandidateId}, candidateSamples={CandidateSamples}, " +
            "confirmationSamples={ConfirmationSamples}, minDurationMs={MinDurationMs}, minRms={MinRms:F4}, responseCts={HasResponseCts}.",
            plan.Source,
            decision.TurnId,
            decision.InterruptedTurnId,
            _identityRequired ? "pending-speaker" : "authorization-not-required:confirmation-disabled",
            plan.Identity.CandidateId,
            decision.ConfirmedCandidateSamples,
            decision.ConfirmedConfirmationSamples,
            decision.ConfirmationMinDurationMs,
            decision.ConfirmationMinRms,
            decision.ResponseToCancel is not null);

        if (_identityRequired)
        {
            // The module already holds this candidate's Match; the Kernel now records it as the one
            // legitimate stop trigger and issues the owner-scoped stop effects exactly once.
            return BargeInSemanticPlan.AuthorizationResolved(
                TurnSignal.AuthorizationResolved(
                    decision.TurnId,
                    _controlSignals.NextOperationId(),
                    TurnAuthorizationStatus.Authorized,
                    plan.Identity.CandidateId,
                    SpeakerMatchReason),
                decision.TurnId,
                decision.InterruptedTurnId,
                SpeakerMatchReason);
        }

        _benchmarkProbe?.ObserveControlTrace("physical-interrupt-issued.dtd", decision.TurnId);
        return ObserveValidatedBargeIn(
            decision.TurnId,
            text: null,
            reason: AudioConfirmedReason,
            operationId: _controlSignals.NextOperationId());
    }

    public BargeInSemanticPlan CommitAuthorizationResolved(
        BargeInSemanticPlan plan,
        KernelCommitResult kernel)
    {
        if (!kernel.Accepted)
        {
            _logger.LogWarning(
                "Kernel rejected the speaker authorization of validation turn {TurnId}; the candidate is dropped and playback continues.",
                plan.TurnId);
            return ObserveRejectedBargeIn(plan.TurnId, text: null, reason: "command-denied-speaker-unavailable:kernel-rejected");
        }

        _benchmarkProbe?.ObserveControlTrace("physical-interrupt-issued.speaker-match", plan.TurnId);
        _logger.LogDebug(
            "physical-interrupt-issued:speaker-match: turnId={TurnId}, interruptedTurnId={InterruptedTurnId}.",
            plan.TurnId,
            plan.InterruptedTurnId);
        return ObserveValidatedBargeIn(
            plan.TurnId,
            text: null,
            reason: SpeakerMatchReason,
            operationId: _controlSignals.NextOperationId());
    }

    public bool TryConfirmFromNearEndWindow(
        IReadOnlyList<float[]> initialAudio,
        BargeInDecision nearEndDecision,
        bool identityAuthorized,
        out SpeakingDecision decision)
        => _applier.TryConfirmFromNearEndWindow(initialAudio, nearEndDecision, identityAuthorized, out decision);

    /// <summary>
    /// Observes a recognition-outcome resolution without changing feature-owned state. The returned
    /// plan is committed by the control-loop router after the corresponding Kernel decision. Because
    /// interruptions are accepted at authorization time, this is almost always a late completion of
    /// an already accepted turn and is expected to be rejected by the Kernel while still releasing
    /// the recognition-side context.
    /// </summary>
    public BargeInSemanticPlan ObserveValidationResolution(AssistantSignal signal)
    {
        TurnSignal kernelSignal = TurnSignal.InterruptionResolved(
            signal.TurnId,
            signal.OperationId,
            signal.Flag,
            signal.Reason,
            signal.Text,
            sourceSequence: signal.SourceSequence,
            sourceTick: signal.SourceTimestamp);

        if (_validation.IsValidating(signal.TurnId) &&
            _turnState.IsCurrentTurn(signal.TurnId) &&
            _turnState.ConversationState == AssistantState.Listening)
        {
            return BargeInSemanticPlan.ValidationLateCompletion(kernelSignal, signal.TurnId);
        }

        if (!_validation.IsValidating(signal.TurnId))
            return default;

        string reason = signal.Reason ?? "unknown";
        if (signal.Flag)
        {
            if (!_validation.TryObserveValidated(
                    signal.TurnId,
                    out _,
                    out _,
                    out long interruptedTurnId))
            {
                return default;
            }

            return BargeInSemanticPlan.ValidationAccepted(
                kernelSignal,
                signal.TurnId,
                reason,
                interruptedTurnId,
                textPresent: !string.IsNullOrWhiteSpace(signal.Text));
        }

        if (!_validation.TryObserveRejection(
                signal.TurnId,
                out _,
                out AssistantState restored))
        {
            return default;
        }

        return BargeInSemanticPlan.ValidationRejected(
            kernelSignal,
            signal.TurnId,
            reason,
            restored);
    }

    public BargeInSemanticPlan ObserveValidatedBargeIn(
        long turnId,
        string? text,
        string reason,
        long operationId = 0,
        long sourceSequence = 0,
        long sourceTimestamp = 0)
    {
        if (!_validation.TryObserveValidated(
                turnId,
                out _,
                out _,
                out long interruptedTurnId))
        {
            _logger.LogDebug(
                "忽略已失效的播放期 barge-in 授权结果: turnId={TurnId}, reason={Reason}.",
                turnId,
                reason);
            return default;
        }

        // The Accepted publication is an ordered effect of this Kernel decision, emitted before the
        // StateChanged(→Listening) publish effect (EO-13).
        if (operationId <= 0)
            operationId = _controlSignals.NextOperationId();
        TurnSignal signal = TurnSignal.InterruptionResolved(
            turnId,
            operationId,
            accepted: true,
            reason: reason,
            text: text,
            sourceSequence: sourceSequence,
            sourceTick: sourceTimestamp);
        return BargeInSemanticPlan.ValidationAccepted(
            signal,
            turnId,
            reason,
            interruptedTurnId,
            textPresent: !string.IsNullOrWhiteSpace(text));
    }

    public BargeInSemanticPlan ObserveRejectedBargeIn(
        long turnId,
        string? text,
        string reason,
        long operationId = 0,
        long sourceSequence = 0,
        long sourceTimestamp = 0)
    {
        if (!_validation.TryObserveRejection(turnId, out _, out AssistantState restored))
            return default;

        // The Rejected publication is an ordered effect of this Kernel decision (EO-14).
        if (operationId <= 0)
            operationId = _controlSignals.NextOperationId();
        TurnSignal signal = TurnSignal.InterruptionResolved(
            turnId,
            operationId,
            accepted: false,
            reason: reason,
            text: text,
            restoreState: restored,
            sourceSequence: sourceSequence,
            sourceTick: sourceTimestamp);
        return BargeInSemanticPlan.ValidationRejected(
            signal,
            turnId,
            reason,
            restored);
    }

    public AssistantControlSignalResult CommitValidationPlan(
        BargeInSemanticPlan plan,
        KernelCommitResult kernel)
    {
        if (plan.Kind == BargeInSemanticPlanKind.ValidationLateCompletion)
        {
            bool completed = false;
            if (!kernel.Accepted &&
                _validation.IsValidating(plan.TurnId) &&
                _turnState.IsCurrentTurn(plan.TurnId) &&
                _turnState.ConversationState == AssistantState.Listening)
            {
                completed = _validation.TryCompleteAccepted(plan.TurnId, out _);
                CompleteDecisionContext(plan.TurnId);
            }

            return new AssistantControlSignalResult(completed, false, 0);
        }

        if (!kernel.Accepted)
            return new AssistantControlSignalResult(false, false, 0);

        if (plan.Kind == BargeInSemanticPlanKind.ValidationAccepted)
        {
            _benchmarkProbe?.ObserveControlTrace(
                "validation.accept.begin",
                plan.TurnId,
                detail: plan.Reason);
            _logger.LogDebug(
                "播放期 barge-in 已接受: turnId={TurnId}, interruptedTurnId={InterruptedTurnId}, " +
                "reason={Reason}, textPresent={TextPresent}.",
                plan.TurnId,
                plan.InterruptedTurnId,
                plan.Reason,
                plan.TextPresent);
            _history.RemoveUnansweredUserMessage(plan.InterruptedTurnId);
            _turnContexts.TryCreate(plan.TurnId);
            if (!_validation.CommitAccepted(plan.TurnId, responseCts: null, clearValidationContext: false))
            {
                throw new InvalidOperationException(
                    "Validated barge-in acceptance could not be committed after Kernel acceptance.");
            }

            _recognitionTurn.ClearActiveSessionId();
            // Recognition readiness is deferred for every accepted interruption: the ASR session
            // opened for the candidate keeps collecting until the endpoint, then the whole-utterance
            // outcome decides whether a response follows.
            _recognitionPreparation.MarkDeferred(plan.TurnId);
            CompleteDecisionContext(plan.TurnId);
            return new AssistantControlSignalResult(
                kernel.Accepted,
                kernel.StateChanged,
                kernel.PublicationSequence);
        }

        if (!_validation.CommitRejected(plan.TurnId))
        {
            throw new InvalidOperationException(
                "Validated barge-in rejection could not be committed after Kernel acceptance.");
        }

        // Denied: the candidate is dropped; the interrupted reply was never touched, so nothing is
        // restored. Only the candidate's own capture and turn context are released.
        _candidates.Clear();
        CompleteDecisionContext(plan.TurnId);
        HandleRejectedStateRestored(plan.RestoredState.GetValueOrDefault());
        _recognitionPreparation.TakeDeferred(plan.TurnId);
        _turnContexts.Release(plan.TurnId);
        if (_runScope.CanOpenInput)
            _gate.Open();

        return new AssistantControlSignalResult(
            kernel.Accepted,
            kernel.StateChanged,
            kernel.PublicationSequence);
    }

    public void ProcessCompleteValidationTurn(long turnId, bool endConversationRequested)
    {
        CancellationTokenSource? responseCts = _turnState.IsCurrentTurn(turnId)
            ? _resources.GetOrCreate(turnId).ResponseCts
            : null;
        bool completedCurrentTurn = _turnCompletion!.TryCompleteTurn(
            turnId,
            endConversationRequested,
            publishStateTransition: false);
        if (completedCurrentTurn &&
            responseCts?.IsCancellationRequested != true &&
            _runScope.CanOpenInput &&
            IsIdleWithoutCandidate)
        {
            _gate.Open();
        }

        _history.ForgetCommittedResponseTurn(turnId);
        ForgetDecisionContext(turnId);
        if (responseCts is null)
            _turnContexts.Release(turnId);
    }

    /// <summary>
    /// 是否处于"真正空闲": Idle 且没有正在累积的语音候选。连续对话窗口的超时判定只应在
    /// 该状态下进行, 否则会把候选累积期/回合进行中的用户语音掐断在半截。
    /// </summary>
    public bool IsIdleWithoutCandidate
        => _turnState.ConversationState == AssistantState.Idle &&
            !_candidates.IsCandidateActive &&
            !_validation.HasValidation;

    public AssistantState ConversationState => _turnState.ConversationState;

    public long CurrentTurnId => _turnState.TurnId;

    /// <summary>
    /// 播放中且没有正在累积/验证的语音候选 —— 主动近端 barge-in 触发的前置条件。
    /// </summary>
    public bool IsSpeakingWithoutPendingCandidate
        => _turnState.ConversationState == AssistantState.Speaking &&
            !_candidates.IsCandidateActive &&
            !_validation.HasValidation;

    public bool IsValidating(long turnId) => _validation.IsValidating(turnId);

    public bool TryObserveAsrStartFailure(long turnId, out AssistantState restoredState)
        => _validation.TryObserveAsrStartFailure(turnId, out restoredState);

    public bool CommitAsrStartFailure(long turnId)
    {
        if (!_validation.CommitAsrStartFailure(turnId))
            return false;

        _candidates.Clear();
        return true;
    }

    public void ClearValidationPendingAudio() => _validation.ClearPendingAudio();

    public void ClearCandidates() => _candidates.Clear();

    public void CompleteDecisionContext(long turnId)
    {
        _journal.CompleteContext(turnId);
        _journal.ForgetTriggerSource(turnId);
    }

    public void ForgetDecisionContext(long turnId) => CompleteDecisionContext(turnId);

    public void HandleRejectedStateRestored(AssistantState restoredState)
    {
        if (restoredState != AssistantState.Idle ||
            _conversationOptions.ContinuousConversationEnabled ||
            !_wakeupInstalled ||
            !_runScope.CanOpenInput)
        {
            return;
        }

        WakeupTimeoutDecision decision = _wakeup!.ReturnToWakeWord();
        if (decision.ModeChanged)
            _controlSignals.SubmitWakeModeChanged(decision.OldMode, decision.NewMode);
    }
}
