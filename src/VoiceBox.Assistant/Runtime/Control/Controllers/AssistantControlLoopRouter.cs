using VoiceBox.Assistant.Kernel.Extensions;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Projections;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>
/// The control consumer's dispatch table. Every branch routes one admitted envelope to the single
/// collaborator that owns it and reports the verdict back to the waiting producer. It owns no Turn
/// state, no module state and no policy of its own.
/// </summary>
internal sealed class AssistantControlLoopRouter : IAssistantControlLoopHandler
{
    private readonly AssistantControlLoopHost _loopHost;
    private readonly ControlOperationRegistry _operations;
    private readonly ITurnStateView _turnState;
    private readonly ITurnSemanticCommitPort _kernel;
    private readonly TurnStateProjection _projection;
    private readonly TurnContextStore _turnContexts;
    private readonly SpeechBoundaryIngress _speechIngress;
    private readonly SpeechBoundaryCoordinator _speechBoundary;
    private readonly RecognitionTurnCoordinator _recognition;
    private readonly BargeInTurnCoordinator _bargeInTurns;
    private readonly ActiveNearEndControlHandler _nearEnd;
    private readonly INearEndCaptureControlPort _nearEndCapture;
    private readonly PlaybackStartedControlHandler _playbackStarted;
    private readonly WakeupInterruptOwner _wakeup;
    private readonly VadSpeakerGateHost _speakerGateHost;
    private readonly AssistantFaultConverger _faults;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly RuntimeRunScope _runScope;

    public AssistantControlLoopRouter(
        AssistantControlLoopHost loopHost,
        ControlOperationRegistry operations,
        ITurnStateView turnState,
        ITurnSemanticCommitPort kernel,
        TurnStateProjection projection,
        TurnContextStore turnContexts,
        SpeechBoundaryIngress speechIngress,
        SpeechBoundaryCoordinator speechBoundary,
        RecognitionTurnCoordinator recognition,
        BargeInTurnCoordinator bargeInTurns,
        ActiveNearEndControlHandler nearEnd,
        INearEndCaptureControlPort nearEndCapture,
        PlaybackStartedControlHandler playbackStarted,
        WakeupInterruptOwner wakeup,
        VadSpeakerGateHost speakerGateHost,
        AssistantFaultConverger faults,
        AssistantTaskSupervisor tasks,
        RuntimeRunScope runScope)
    {
        _loopHost = loopHost ?? throw new ArgumentNullException(nameof(loopHost));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _projection = projection ?? throw new ArgumentNullException(nameof(projection));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _speechIngress = speechIngress ?? throw new ArgumentNullException(nameof(speechIngress));
        _speechBoundary = speechBoundary ?? throw new ArgumentNullException(nameof(speechBoundary));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _bargeInTurns = bargeInTurns ?? throw new ArgumentNullException(nameof(bargeInTurns));
        _nearEnd = nearEnd ?? throw new ArgumentNullException(nameof(nearEnd));
        _nearEndCapture = nearEndCapture ?? throw new ArgumentNullException(nameof(nearEndCapture));
        _playbackStarted = playbackStarted ?? throw new ArgumentNullException(nameof(playbackStarted));
        _wakeup = wakeup ?? throw new ArgumentNullException(nameof(wakeup));
        _speakerGateHost = speakerGateHost ?? throw new ArgumentNullException(nameof(speakerGateHost));
        _faults = faults ?? throw new ArgumentNullException(nameof(faults));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
    }

    public ValueTask HandleRuntimeCommandAsync(
        RuntimeCommand command,
        ControlEnvelope envelope,
        CancellationToken cancellationToken)
    {
        _operations.CompleteCommand(command.OperationId);
        return ValueTask.CompletedTask;
    }

    public async ValueTask HandleAssistantSignalAsync(
        AssistantSignal signal,
        ControlEnvelope envelope,
        CancellationToken cancellationToken)
    {
        _loopHost.BeginMutation("semantic work");
        try
        {
            CaptureKernelSignalMetadata(signal);
            // Attach the recognition session and replay pre-roll before the Kernel enters
            // Recognizing. The state transition is still committed below with the other signals.
            RecognitionSessionStartResult sessionStart = default;
            if (signal.Kind == AssistantSignalKind.RecognitionStarted)
                sessionStart = _recognition.ObserveSessionStarted(signal);

            BargeInSemanticPlan validationPlan = default;
            if (signal.Kind == AssistantSignalKind.InterruptionResolved)
                validationPlan = _bargeInTurns.ObserveValidationResolution(signal);

            // A completed single-talk recognition meets the turn's identity gate before it may
            // become a semantic turn: pending verdicts park it, denied verdicts drop it.
            RecognitionCompletionGate completionGate = RecognitionCompletionGate.Proceed;
            if (signal.Kind == AssistantSignalKind.RecognitionCompleted)
                completionGate = _recognition.ObserveCompletionGate(signal.TurnId);

            AssistantSignal projectionSignal = sessionStart is
            { Attached: true, IsValidationTurn: true } ||
                sessionStart.RecognitionReadyWasDeferred ||
                completionGate != RecognitionCompletionGate.Proceed
                ? signal.WithKernelProjectionSuppressed()
                : signal;
            KernelCommitResult kernel = validationPlan.Signal is { } validationSignal
                ? _kernel.Commit(validationSignal)
                : signal.Kind == AssistantSignalKind.WakeupObserved
                    ? default
                    : signal.Kind == AssistantSignalKind.RecognitionStarted && !sessionStart.Attached
                        ? default
                        : ProjectAndCommit(projectionSignal, envelope);
            switch (signal.Kind)
            {
                case AssistantSignalKind.WakeupModeChanged:
                    CompleteFromKernel(signal, kernel);
                    break;

                case AssistantSignalKind.ActiveNearEndInterruptionRequested:
                    ProcessBargeInPlan(_nearEnd.ProcessInterruptionRequested(signal.OperationId));
                    break;

                case AssistantSignalKind.ActiveNearEndSpeechEnded:
                    _ = _tasks.Track(
                        _nearEnd.DrainSpeechAsync(
                            signal.TurnId,
                            TimeSpan.FromTicks(signal.DurationTicks),
                            cancellationToken),
                        "active-near-end.speech-drain",
                        signal.TurnId,
                        AssistantTaskFaultPolicy.ObserveAndLog);
                    break;

                case AssistantSignalKind.ActiveNearEndSpeechDrainCompleted:
                    ProcessSpeechBoundaryObservation(_nearEnd.ProcessSpeechDrainCompleted(
                        signal.TurnId,
                        TimeSpan.FromTicks(signal.DurationTicks)));
                    break;

                case AssistantSignalKind.PlaybackTransportStarted:
                    _playbackStarted.Handle(signal.TurnId);
                    break;

                case AssistantSignalKind.SpeakerGateResolved:
                    if (_speakerGateHost.TryTakeObservation(
                            signal.OperationId,
                            out SpeakerVerificationPurpose gatePurpose,
                            out BargeInSpeakerGateStatus status,
                            out TimeSpan? pendingSpeechEnd))
                    {
                        ProcessSpeechBoundaryObservation(
                            _speechBoundary.ProcessVadSpeakerGateObservation(
                                gatePurpose,
                                status,
                                pendingSpeechEnd,
                                _operations.NextOperationId()));
                    }

                    break;

                case AssistantSignalKind.SpeechStarted:
                    if (!_speechIngress.IsCurrentAdmissionEpoch(signal.AdmissionEpoch))
                    {
                        _speechIngress.SettlePendingSpeechStart(signal.AdmissionEpoch);
                        break;
                    }

                    ProcessSpeechBoundaryObservation(
                        _speechBoundary.ProcessSpeechStarted(
                            signal.Flag,
                            _operations.NextOperationId(),
                            _speechIngress.TryConsumeWakeupReplayAuthorization(signal.AdmissionEpoch)));
                    _speechIngress.SettlePendingSpeechStart(signal.AdmissionEpoch);
                    DrainSpeechIngressAudio();
                    await _recognition.DrainAudioAsync();
                    break;

                case AssistantSignalKind.SpeechAudioAvailable:
                    _speechIngress.ReleaseAudioDrainSchedule();
                    DrainSpeechIngressAudio();
                    if (_speechIngress.HasDeferredSpeakingAudio)
                        _speechIngress.ScheduleDeferredSpeakingDrain();
                    break;

                case AssistantSignalKind.SpeechEnded:
                    if (!_speechIngress.IsCurrentAdmissionEpoch(signal.AdmissionEpoch))
                        break;
                    DrainSpeechIngressAudio();
                    await _recognition.DrainAudioAsync();
                    ProcessSpeechBoundaryObservation(
                        _speechBoundary.ProcessSpeechEnded(
                            TimeSpan.FromTicks(signal.DurationTicks),
                            signal.Flag));
                    break;

                case AssistantSignalKind.WakeupObserved:
                    ProcessWakeupPlan(_wakeup.ProcessObservation(signal.OperationId));
                    break;

                case AssistantSignalKind.RecognitionStarted:
                    ProcessRecognitionPlan(
                        _recognition.CommitSessionStarted(sessionStart, kernel));
                    bool sessionAccepted = sessionStart.Attached &&
                        (sessionStart.IsValidationTurn ||
                         sessionStart.RecognitionReadyWasDeferred ||
                         kernel.Accepted);
                    _operations.CompleteSignal(
                        signal,
                        new AssistantControlSignalResult(
                            sessionAccepted,
                            false,
                            kernel.PublicationSequence));
                    break;

                case AssistantSignalKind.RecognitionFailed:
                    RecognitionResetPlan reset = _recognition.ObserveResetAfterAsrStartFailure(signal.TurnId);
                    if (reset.Signal is { } resetSignal)
                    {
                        KernelCommitResult resetKernel = _kernel.Commit(resetSignal);
                        _recognition.CommitResetAfterAsrStartFailure(reset, resetKernel, _runScope.Token);
                    }
                    else
                    {
                        _recognition.CommitResetAfterAsrStartFailure(reset, default, _runScope.Token);
                    }
                    CompleteAccepted(signal);
                    break;

                case AssistantSignalKind.InterruptionResolved:
                    if (validationPlan.Kind != BargeInSemanticPlanKind.None)
                    {
                        AssistantControlSignalResult validationResult =
                            _bargeInTurns.CommitValidationPlan(validationPlan, kernel);
                        _operations.CompleteSignal(signal, validationResult);
                    }
                    else
                    {
                        CompleteFromKernel(signal, kernel);
                    }
                    break;

                case AssistantSignalKind.RecognitionCompleted:
                    _recognition.TakeDeferred(signal.TurnId);
                    switch (completionGate)
                    {
                        case RecognitionCompletionGate.Park:
                            _recognition.ParkCompletion(TurnSignal.RecognitionCompleted(
                                signal.TurnId,
                                signal.OperationId,
                                signal.Text ?? string.Empty,
                                TimeSpan.FromTicks(signal.DurationTicks)));
                            CompleteAccepted(signal);
                            break;

                        case RecognitionCompletionGate.Drop:
                            CompleteTurnWithoutResponse(signal.TurnId);
                            CompleteAccepted(signal);
                            break;

                        default:
                            CompleteFromKernel(signal, kernel);
                            break;
                    }

                    break;

                case AssistantSignalKind.ResponseStarted:
                    _operations.CompleteSignal(
                        signal,
                        new AssistantControlSignalResult(_turnState.IsCurrentTurn(signal.TurnId), false, 0));
                    break;

                case AssistantSignalKind.BackgroundResponseStarted:
                    if (kernel.Accepted)
                        _turnContexts.TryCreate(signal.TurnId);
                    CompleteFromKernel(signal, kernel);
                    break;

                case AssistantSignalKind.ResponseSpeakableStarted:
                case AssistantSignalKind.ResponseCompleted:
                case AssistantSignalKind.PlaybackCompleted:
                    // The public semantic state is already committed by TurnKernel; there is no
                    // second projection transition here.
                    CompleteFromKernel(signal, kernel);
                    break;

                case AssistantSignalKind.ResponseFailed:
                case AssistantSignalKind.OperationCanceled:
                    _bargeInTurns.ForgetDecisionContext(signal.TurnId);
                    CompleteAccepted(signal);
                    break;

                case AssistantSignalKind.ValidationCancelled:
                    _recognition.TakeDeferred(signal.TurnId);
                    _bargeInTurns.ProcessCompleteValidationTurn(signal.TurnId, endConversationRequested: false);
                    CompleteAccepted(signal);
                    break;

                case AssistantSignalKind.TurnCompleted:
                    _bargeInTurns.ProcessCompleteValidationTurn(signal.TurnId, signal.Flag);
                    CompleteAccepted(signal);
                    break;

                case AssistantSignalKind.RuntimeFault:
                    _faults.Converge(signal.Failure);
                    break;
            }
        }
        finally
        {
            _operations.SettleInterruptionAdmission(signal);
            _projection.Publish(_turnState.ConversationState);
            _loopHost.EndMutation();
        }
    }

    public ValueTask HandleTerminalFaultAsync(
        AssistantFailure failure,
        ControlEnvelope envelope,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _loopHost.BeginMutation("terminal fault work");
        try
        {
            _loopHost.MarkTerminated();
            _kernel.Commit(TurnSignal.RuntimeFaultObserved(
                AssistantSignalKernelProjection.ProjectFailure(failure)));
            _faults.Converge(failure);
            return ValueTask.CompletedTask;
        }
        finally
        {
            _loopHost.EndMutation();
        }
    }

    public ValueTask HandleAbandonedAfterBarrierAsync(
        ControlEnvelope envelope,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (envelope.Kind == ControlEnvelopeKind.RuntimeCommand)
        {
            _operations.CancelCommand(envelope.Command.OperationId, cancellationToken);
        }
        else if (envelope.Kind == ControlEnvelopeKind.AssistantSignal)
        {
            _operations.SettleInterruptionAdmission(envelope.Signal);
            _operations.CompleteSignal(
                envelope.Signal,
                new AssistantControlSignalResult(false, false, 0));
        }

        return ValueTask.CompletedTask;
    }

    private KernelCommitResult ProjectAndCommit(
        AssistantSignal signal,
        ControlEnvelope envelope)
    {
        TurnSignal? projected = AssistantSignalKernelProjection.Project(
            signal,
            _turnState.ConversationState,
            signal.SourceSequence != 0 ? signal.SourceSequence : envelope.IngressSequence,
            signal.SourceTimestamp != 0 ? signal.SourceTimestamp : envelope.SourceTimestamp);
        return projected is { } kernelSignal ? _kernel.Commit(kernelSignal) : default;
    }

    private void ProcessSpeechBoundaryObservation(SpeechBoundaryObservation observation)
    {
        switch (observation.Kind)
        {
            case SpeechBoundaryObservationKind.StartListening:
                {
                    KernelCommitResult kernel = CommitRequired(observation.Signal, "speech-started");
                    _speechBoundary.ApplyStartListening(observation, kernel);
                    break;
                }

            case SpeechBoundaryObservationKind.ConfirmedListening:
                {
                    KernelCommitResult kernel = CommitRequired(observation.Signal, "confirmed-listening");
                    _speechBoundary.CommitListeningObservation(observation, kernel);
                    break;
                }

            case SpeechBoundaryObservationKind.ConfirmedBargeIn:
                {
                    BargeInSemanticPlan plan = _bargeInTurns.ObserveConfirmedBargeIn(
                        observation.SpeakingDecision,
                        observation.BargeInSource,
                        observation.InitialAudio,
                        observation.Identity);
                    ProcessBargeInPlan(plan);
                    if (observation.PendingSpeechEnd is { } duration)
                    {
                        ProcessSpeechBoundaryObservation(_speechBoundary.HandleSpeechEnded(duration));
                    }

                    break;
                }

            case SpeechBoundaryObservationKind.BeginRecognition:
                ProcessRecognitionPlan(
                    _recognition.BeginRecognition(
                        observation.TurnId,
                        observation.AsrSessionId!,
                        observation.SpeechDuration));
                break;

            case SpeechBoundaryObservationKind.SingleTalkAuthorization:
                {
                    // The verdict may arrive after the turn already ended (failure, stop); a
                    // rejected commit then leaves nothing to apply.
                    KernelCommitResult kernel = _kernel.Commit(observation.Signal!.Value);
                    if (kernel.Accepted)
                        ApplySingleTalkAuthorization(observation.TurnId);
                    break;
                }
        }
    }

    /// <summary>
    /// Applies the recorded single-talk verdict to a parked whole-utterance result: Authorized
    /// releases it to the Kernel as the canonical RecognitionCompleted; a denied verdict discards
    /// the command and completes the turn without a response. Without a parked result the
    /// recognition completion itself reads the recorded verdict when it arrives.
    /// </summary>
    private void ApplySingleTalkAuthorization(long turnId)
    {
        TurnSnapshot snapshot = _turnState.Snapshot;
        if (snapshot.TurnId != turnId || !_recognition.HasParkedCompletion(turnId))
            return;

        if (snapshot.Authorization == TurnAuthorizationStatus.Authorized)
        {
            if (_recognition.TryTakeParkedCompletion(turnId, out TurnSignal completion))
                CommitRequired(completion, "single-talk-authorized-recognition");
            return;
        }

        if (snapshot.Authorization.IsDenied())
        {
            _recognition.TryTakeParkedCompletion(turnId, out _);
            CompleteTurnWithoutResponse(turnId);
        }
    }

    /// <summary>
    /// Ends the current turn with no semantic outcome, mirroring the TurnCompleted path: the Kernel
    /// leaves Listening/Recognizing for Idle and the turn owners release their per-turn context.
    /// </summary>
    private void CompleteTurnWithoutResponse(long turnId)
    {
        _kernel.Commit(TurnSignal.CancelRequested(turnId, _operations.NextOperationId(), "turn-completed"));
        _bargeInTurns.ProcessCompleteValidationTurn(turnId, endConversationRequested: false);
    }

    private void DrainSpeechIngressAudio()
    {
        while (_speechIngress.TryReadAdmittedAudio(out float[] audio))
        {
            ProcessSpeechBoundaryObservation(
                _speechBoundary.ProcessSpeaking(audio, _operations.NextOperationId()));
        }
    }

    private void ProcessRecognitionPlan(RecognitionSemanticPlan plan)
    {
        while (plan.Kind != RecognitionSemanticPlanKind.None)
        {
            KernelCommitResult kernel = plan.Signal is { } signal
                ? CommitRequired(signal, $"recognition-{plan.Kind}")
                : default;

            plan = _recognition.CommitSemanticPlan(plan, kernel);
        }
    }

    private void ProcessBargeInPlan(BargeInSemanticPlan plan)
    {
        while (plan.Kind != BargeInSemanticPlanKind.None)
        {
            KernelCommitResult kernel = plan.Signal is { } signal
                ? plan.Kind is BargeInSemanticPlanKind.ValidationLateCompletion
                    or BargeInSemanticPlanKind.AuthorizationResolved
                    ? _kernel.Commit(signal)
                    : CommitRequired(signal, $"barge-in-{plan.Kind}")
                : default;
            plan = plan.Kind switch
            {
                BargeInSemanticPlanKind.InterruptionRequested =>
                    _bargeInTurns.CommitInterruptionRequested(plan, kernel),
                BargeInSemanticPlanKind.AuthorizationResolved =>
                    _bargeInTurns.CommitAuthorizationResolved(plan, kernel),
                BargeInSemanticPlanKind.ValidationAccepted or
                BargeInSemanticPlanKind.ValidationRejected =>
                    CommitValidationPlanAndFinish(plan, kernel),
                _ => default,
            };
        }
    }

    private BargeInSemanticPlan CommitValidationPlanAndFinish(
        BargeInSemanticPlan plan,
        KernelCommitResult kernel)
    {
        _bargeInTurns.CommitValidationPlan(plan, kernel);
        if (kernel.Accepted && plan.Kind == BargeInSemanticPlanKind.ValidationRejected)
            _nearEndCapture.StopCapture(plan.TurnId, startCooldown: true);
        return default;
    }

    private void ProcessWakeupPlan(WakeupSemanticPlan plan)
    {
        while (plan.Kind != WakeupSemanticPlanKind.None)
        {
            KernelCommitResult kernel = plan.Signal is { } signal
                ? CommitRequired(signal, $"wakeup-{plan.Kind}")
                : default;
            plan = plan.Kind switch
            {
                WakeupSemanticPlanKind.Observed => _wakeup.CommitObservation(plan, kernel),
                WakeupSemanticPlanKind.Interrupt =>
                    CommitWakeupInterruptAndFinish(plan, kernel),
                _ => default,
            };
        }
    }

    private WakeupSemanticPlan CommitWakeupInterruptAndFinish(
        WakeupSemanticPlan plan,
        KernelCommitResult kernel)
    {
        _wakeup.CommitInterrupt(plan, kernel);
        return default;
    }

    private KernelCommitResult CommitRequired(TurnSignal? signal, string source)
    {
        if (signal is not { } value)
            return default;

        KernelCommitResult result = _kernel.Commit(value);
        if (!result.Accepted)
        {
            throw new InvalidOperationException(
                $"TurnKernel rejected a runtime-owned semantic decision from '{source}'.");
        }

        return result;
    }

    private void CaptureKernelSignalMetadata(AssistantSignal signal)
    {
        if (signal.Kind != AssistantSignalKind.RecognitionCompleted)
            return;

        _turnContexts.TryGetRecognitionMetadata(signal.TurnId, out string? rawText, out _);
        _turnContexts.TrySetRecognitionMetadata(
            signal.TurnId,
            rawText,
            TimeSpan.FromTicks(signal.DurationTicks));
    }

    private void CompleteFromKernel(AssistantSignal signal, KernelCommitResult kernel)
        => _operations.CompleteSignal(
            signal,
            new AssistantControlSignalResult(kernel.Accepted, kernel.StateChanged, kernel.PublicationSequence));

    private void CompleteAccepted(AssistantSignal signal)
        => _operations.CompleteSignal(signal, new AssistantControlSignalResult(true, false, 0));
}
