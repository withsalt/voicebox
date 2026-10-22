using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Extensions;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;

namespace VoiceBox.Assistant.Kernel;

/// <summary>
/// Pure TurnKernel boundary. The runtime owns the current snapshot; this type performs only
/// deterministic Snapshot + Signal -> Decision reduction and has no mutable state or I/O.
/// </summary>
internal static class TurnKernel
{
    public static TurnDecision Reduce(TurnSnapshot current, TurnSignal signal)
    {
        ArgumentNullException.ThrowIfNull(current);

        return signal.Kind switch
        {
            TurnSignalKind.InputGateChanged => ReduceInputGate(current, signal),
            TurnSignalKind.WakeupDetected => ReduceWakeupDetected(current, signal),
            TurnSignalKind.SpeechStarted => ReduceSpeechStarted(current, signal),
            TurnSignalKind.RecognitionSessionReady => ReduceRecognitionReady(current, signal),
            TurnSignalKind.SpeechEnded => ReduceSpeechEnded(current, signal),
            TurnSignalKind.RecognitionCompleted => ReduceRecognitionCompleted(current, signal),
            TurnSignalKind.ResponseGenerated => ReduceResponseGenerated(current, signal),
            TurnSignalKind.BackgroundResponseStarted => ReduceBackgroundResponseStarted(current, signal),
            TurnSignalKind.PlaybackStarted => ReducePlaybackStarted(current, signal),
            TurnSignalKind.PlaybackCompleted => ReducePlaybackCompleted(current, signal),
            TurnSignalKind.InterruptionRequested => ReduceInterruptionRequested(current, signal),
            TurnSignalKind.InterruptionResolved => ReduceInterruptionResolved(current, signal),
            TurnSignalKind.RecognitionFailed => ReduceRecognitionFailed(current, signal),
            TurnSignalKind.OperationFailed => ReduceOperationFailed(current, signal),
            TurnSignalKind.CancelRequested => ReduceCancelled(current, signal),
            TurnSignalKind.RuntimeFaultObserved => ReduceRuntimeFault(current, signal),
            TurnSignalKind.AuthorizationResolved => ReduceAuthorizationResolved(current, signal),
            _ => Rejected(current),
        };
    }

    /// <summary>
    /// The one physical-stop gate. Emits the interrupted turn's cancel/stop/clear effects exactly
    /// once per candidate and records who legitimately issued them; every later request for the
    /// same owner turn is a no-op at this level and again at the playback handler's owner check.
    /// </summary>
    private static bool TryIssuePhysicalStop(
        TurnSnapshot current,
        long operationId,
        PhysicalStopTrigger trigger,
        List<AssistantEffect> effects)
    {
        if (current.PhysicalStopIssued || current.InterruptedTurnId <= 0)
            return false;

        // Device audio first, then the response pipeline: the owner-scoped stop must find the
        // reply still active so its queued PCM is removed deterministically, before cancellation
        // of the response source can retire the playback request on its own.
        effects.Add(new StopPlaybackEffect(current.InterruptedTurnId, operationId));
        effects.Add(new ClearPlaybackQueueEffect(current.InterruptedTurnId, operationId));
        effects.Add(new CancelResponseEffect(current.InterruptedTurnId, operationId));
        return true;
    }

    private static TurnDecision ReduceInputGate(TurnSnapshot current, TurnSignal signal)
    {
        if (current.InputGateOpen == signal.Flag)
            return Rejected(current);

        return Accepted(
            current with { InputGateOpen = signal.Flag },
            new SetInputGateEffect(current.TurnId, signal.Flag));
    }

    private static TurnDecision ReduceWakeupDetected(TurnSnapshot current, TurnSignal signal)
    {
        KernelWakeupObservation observation = signal.Wakeup;
        if ((!observation.HasWakeup && !observation.ModeChanged) || signal.OperationId <= 0)
            return Rejected(current);

        var effects = new List<AssistantEffect>(2);
        if (observation.ModeChanged)
        {
            effects.Add(new PublishWakeModeChangedEffect(
                current.TurnId,
                observation.FromMode,
                observation.ToMode,
                signal.OperationId));
        }

        if (observation.HasWakeup)
            effects.Add(new PublishWakeupDetectedEffect(current.TurnId, observation, signal.OperationId));

        return Accepted(
            current with { WakeMode = observation.ToMode },
            effects.ToArray());
    }

    /// <summary>
    /// Opens the Listening turn. With the effective SingleTalk purpose the turn enters as
    /// <c>PendingSpeaker</c>: its ASR session and speaker query run in parallel, and
    /// <see cref="ReduceRecognitionCompleted"/> only admits the utterance once
    /// <see cref="ReduceAuthorizationResolved"/> recorded the owner's Match.
    /// </summary>
    private static TurnDecision ReduceSpeechStarted(TurnSnapshot current, TurnSignal signal)
    {
        if (current.ConversationState != AssistantState.Idle ||
            signal.TurnId <= 0 ||
            signal.TurnId != current.NextTurnOrdinal)
            return Rejected(current);
        if (signal.Authorization is not (TurnAuthorizationStatus.NotRequired or TurnAuthorizationStatus.PendingSpeaker))
            return Rejected(current);

        TurnSnapshot next = current with
        {
            ConversationState = AssistantState.Listening,
            TurnId = signal.TurnId,
            NextTurnOrdinal = checked(signal.TurnId + 1),
            InterruptedTurnId = 0,
            ValidationTurnId = 0,
            ResponseLifecycle = ResponseLifecycle.None,
            Authorization = signal.Authorization,
            CandidateId = signal.CandidateId,
            PhysicalStopIssued = false,
            PhysicalStopTrigger = PhysicalStopTrigger.None,
        };
        return Accepted(
            next,
            new PublishStateChangedEffect(
                signal.TurnId,
                AssistantState.Idle,
                AssistantState.Listening),
            new PrepareRecognitionEffect(signal.TurnId, signal.OperationId));
    }

    private static TurnDecision ReduceRecognitionReady(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState != AssistantState.Listening ||
            signal.OperationId <= 0)
            return Rejected(current);

        return Accepted(
            current with { ConversationState = AssistantState.Recognizing },
            new PublishStateChangedEffect(
                signal.TurnId,
                AssistantState.Listening,
                AssistantState.Recognizing,
                signal.OperationId));
    }

    private static TurnDecision ReduceSpeechEnded(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState != AssistantState.Recognizing ||
            signal.OperationId <= 0 ||
            string.IsNullOrEmpty(signal.Text))
        {
            return Rejected(current);
        }

        return Accepted(
            current,
            new FinishRecognitionEffect(signal.TurnId, signal.OperationId, signal.Text));
    }

    private static TurnDecision ReduceRecognitionCompleted(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState is not (AssistantState.Listening or AssistantState.Recognizing) ||
            signal.OperationId <= 0)
            return Rejected(current);
        // A single-talk utterance whose speaker verdict is still pending or was denied never becomes a
        // semantic turn: the module parks the result until the Match, or drops it and completes the
        // turn without a response.
        if (current.ValidationTurnId == 0 &&
            (current.Authorization == TurnAuthorizationStatus.PendingSpeaker || current.Authorization.IsDenied()))
            return Rejected(current);

        return Accepted(
            current with
            {
                ConversationState = AssistantState.Thinking,
                ResponseLifecycle = ResponseLifecycle.Generating,
            },
            new PublishUserSpokeEffect(
                signal.TurnId,
                signal.Text ?? string.Empty,
                TimeSpan.FromTicks(signal.DurationTicks),
                signal.OperationId),
            new PublishStateChangedEffect(
                signal.TurnId,
                current.ConversationState,
                AssistantState.Thinking,
                signal.OperationId),
            new GenerateResponseEffect(signal.TurnId, signal.OperationId, signal.Text ?? string.Empty));
    }

    private static TurnDecision ReduceBackgroundResponseStarted(TurnSnapshot current, TurnSignal signal)
    {
        if (current.ConversationState != AssistantState.Idle ||
            current.ValidationTurnId != 0 ||
            signal.TurnId <= 0 || signal.TurnId != current.NextTurnOrdinal ||
            signal.OperationId <= 0)
        {
            return Rejected(current);
        }

        return Accepted(
            current with
            {
                ConversationState = AssistantState.Thinking,
                TurnId = signal.TurnId,
                NextTurnOrdinal = checked(signal.TurnId + 1),
                InterruptedTurnId = 0,
                ValidationTurnId = 0,
                ResponseLifecycle = ResponseLifecycle.Generated,
                Authorization = TurnAuthorizationStatus.NotRequired,
                CandidateId = 0,
                PhysicalStopIssued = false,
                PhysicalStopTrigger = PhysicalStopTrigger.None,
            },
            new PublishStateChangedEffect(
                signal.TurnId, AssistantState.Idle, AssistantState.Thinking, signal.OperationId));
    }

    private static TurnDecision ReduceResponseGenerated(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState is not (AssistantState.Thinking or AssistantState.Speaking) ||
            signal.OperationId <= 0)
        {
            return Rejected(current);
        }

        return Accepted(
            current with { ResponseLifecycle = ResponseLifecycle.Generated },
            new CommitHistoryEffect(signal.TurnId, signal.Text ?? string.Empty, signal.OperationId),
            new PublishAssistantResponseGeneratedEffect(
                signal.TurnId,
                signal.Text ?? string.Empty,
                signal.OperationId));
    }

    private static TurnDecision ReducePlaybackStarted(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState is not (AssistantState.Thinking or AssistantState.Speaking) ||
            signal.OperationId <= 0)
        {
            return Rejected(current);
        }

        bool stateChanged = current.ConversationState != AssistantState.Speaking;
        TurnSnapshot next = current with
        {
            ConversationState = AssistantState.Speaking,
            ResponseLifecycle = ResponseLifecycle.Playing,
        };
        return stateChanged
            ? Accepted(
                next,
                new PublishStateChangedEffect(
                    signal.TurnId,
                    AssistantState.Thinking,
                    AssistantState.Speaking,
                    signal.OperationId))
            : Accepted(next);
    }

    private static TurnDecision ReducePlaybackCompleted(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState != AssistantState.Speaking ||
            signal.OperationId <= 0)
            return Rejected(current);

        return Accepted(
            current with
            {
                ConversationState = AssistantState.Idle,
                ResponseLifecycle = ResponseLifecycle.Completed,
            },
            new PublishAssistantRepliedEffect(
                signal.TurnId,
                signal.Text ?? string.Empty,
                TimeSpan.FromTicks(signal.DurationTicks),
                signal.OperationId),
            new CompleteTurnEffect(signal.TurnId, signal.OperationId),
            new PublishStateChangedEffect(
                signal.TurnId,
                AssistantState.Speaking,
                AssistantState.Idle,
                signal.OperationId));
    }

    private static TurnDecision ReduceInterruptionRequested(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            signal.OperationId <= 0 ||
            signal.OperationId != current.NextTurnOrdinal)
            return Rejected(current);

        if (current.ConversationState == AssistantState.Thinking)
        {
            // The interrupting candidate already carried its own identity gate (BargeInCommand
            // when KWS is enabled); the new Listening turn starts with no pending authorization.
            return Accepted(
                current with
                {
                    ConversationState = AssistantState.Listening,
                    TurnId = signal.OperationId,
                    NextTurnOrdinal = checked(signal.OperationId + 1),
                    InterruptedTurnId = 0,
                    ValidationTurnId = 0,
                    ResponseLifecycle = ResponseLifecycle.None,
                    Authorization = TurnAuthorizationStatus.NotRequired,
                    CandidateId = 0,
                    PhysicalStopIssued = false,
                    PhysicalStopTrigger = PhysicalStopTrigger.None,
                },
                new CancelResponseEffect(current.TurnId, signal.OperationId),
                new RollbackHistoryEffect(current.TurnId, signal.OperationId),
                new PublishStateChangedEffect(
                    signal.OperationId,
                    AssistantState.Thinking,
                    AssistantState.Listening,
                    signal.OperationId),
                new PrepareRecognitionEffect(signal.OperationId, signal.OperationId));
        }

        if (current.ConversationState != AssistantState.Speaking)
            return Rejected(current);
        if (current.ValidationTurnId != 0)
            return Rejected(current);
        if (signal.Authorization is not (TurnAuthorizationStatus.NotRequired or TurnAuthorizationStatus.PendingSpeaker))
            return Rejected(current);

        TurnSnapshot validating = current with
        {
            InterruptedTurnId = current.TurnId,
            ValidationTurnId = signal.OperationId,
            NextTurnOrdinal = checked(signal.OperationId + 1),
            Authorization = signal.Authorization,
            CandidateId = signal.CandidateId,
            PhysicalStopIssued = false,
            PhysicalStopTrigger = PhysicalStopTrigger.None,
        };

        var effects = new List<AssistantEffect>(5);
        if (signal.Authorization == TurnAuthorizationStatus.NotRequired)
        {
            // KWS disabled: the acoustic interrupt is the whole authorization. Stop now, once.
            TryIssuePhysicalStop(validating, signal.OperationId, PhysicalStopTrigger.AcousticInterrupt, effects);
            validating = validating with
            {
                PhysicalStopIssued = true,
                PhysicalStopTrigger = PhysicalStopTrigger.AcousticInterrupt,
            };
        }
        // PendingSpeaker: the owner's reply keeps playing at its normal gain; no duck, no stop, no
        // clear until AuthorizationResolved(Authorized) for this very candidate.

        effects.Add(new PrepareRecognitionEffect(signal.OperationId, signal.OperationId));
        // The Heard publication is decided here (EO-15: no state transition accompanies it).
        if (signal.Flag)
        {
            effects.Add(new PublishBargeInValidationEffect(
                signal.OperationId,
                BargeInValidationKind.Heard,
                Text: null,
                Reason: null,
                signal.OperationId,
                signal.HasBargeInContext,
                signal.BargeInSource,
                signal.BargeInDiagnostics));
        }

        return Accepted(validating, effects.ToArray());
    }

    /// <summary>
    /// Speaker verdict for the pending candidate. Only a Match of the same candidate turns
    /// <c>PendingSpeaker</c> into <c>Authorized</c> and issues the single stop; DTD progress is
    /// never required. Denied verdicts only record the denial: the interrupted reply continues.
    /// Without a validation turn the verdict belongs to the current single-talk Listening/Recognizing
    /// turn and only records the status; the recognition path acts on it.
    /// </summary>
    private static TurnDecision ReduceAuthorizationResolved(TurnSnapshot current, TurnSignal signal)
    {
        if (current.ValidationTurnId == 0)
            return ReduceSingleTalkAuthorizationResolved(current, signal);

        if (signal.TurnId != current.ValidationTurnId ||
            signal.OperationId <= 0 ||
            current.Authorization != TurnAuthorizationStatus.PendingSpeaker ||
            (current.CandidateId != 0 && signal.CandidateId != current.CandidateId))
        {
            return Rejected(current);
        }

        switch (signal.Authorization)
        {
            case TurnAuthorizationStatus.Authorized:
            {
                var effects = new List<AssistantEffect>(3);
                bool issued = current.ConversationState == AssistantState.Speaking &&
                    TryIssuePhysicalStop(current, signal.OperationId, PhysicalStopTrigger.SpeakerMatch, effects);
                return Accepted(
                    current with
                    {
                        Authorization = TurnAuthorizationStatus.Authorized,
                        PhysicalStopIssued = current.PhysicalStopIssued || issued,
                        PhysicalStopTrigger = issued ? PhysicalStopTrigger.SpeakerMatch : current.PhysicalStopTrigger,
                    },
                    effects.ToArray());
            }

            case TurnAuthorizationStatus.DeniedMismatch:
            case TurnAuthorizationStatus.DeniedInsufficientEvidence:
            case TurnAuthorizationStatus.DeniedUnavailable:
                return Accepted(current with { Authorization = signal.Authorization });

            default:
                return Rejected(current);
        }
    }

    private static TurnDecision ReduceSingleTalkAuthorizationResolved(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            signal.OperationId <= 0 ||
            current.ConversationState is not (AssistantState.Listening or AssistantState.Recognizing) ||
            current.Authorization != TurnAuthorizationStatus.PendingSpeaker ||
            (current.CandidateId != 0 && signal.CandidateId != current.CandidateId))
        {
            return Rejected(current);
        }

        return signal.Authorization switch
        {
            TurnAuthorizationStatus.Authorized or
            TurnAuthorizationStatus.DeniedMismatch or
            TurnAuthorizationStatus.DeniedInsufficientEvidence or
            TurnAuthorizationStatus.DeniedUnavailable
                => Accepted(current with { Authorization = signal.Authorization }),
            _ => Rejected(current),
        };
    }

    private static TurnDecision ReduceInterruptionResolved(TurnSnapshot current, TurnSignal signal)
    {
        if (current.ValidationTurnId == 0 ||
            signal.TurnId != current.ValidationTurnId ||
            signal.OperationId <= 0)
            return Rejected(current);

        if (!signal.Flag)
        {
            var rejected = new List<AssistantEffect>
            {
                new PublishBargeInValidationEffect(
                    signal.TurnId,
                    BargeInValidationKind.Rejected,
                    signal.Text,
                    signal.Reason,
                    signal.OperationId),
            };
            // EO-14: Rejected ≺ StateChanged only when the restored conversation state differs.
            // Nothing is restored on the playback side: a pending or denied candidate never
            // changed the interrupted reply or its gain in the first place.
            if (signal.RestoreState is { } restored && restored != current.ConversationState)
            {
                rejected.Add(new PublishStateChangedEffect(
                    signal.TurnId,
                    current.ConversationState,
                    restored,
                    signal.OperationId));
            }

            return Accepted(
                current with
                {
                    ValidationTurnId = 0,
                    InterruptedTurnId = 0,
                    Authorization = TurnAuthorizationStatus.NotRequired,
                    CandidateId = 0,
                    PhysicalStopIssued = false,
                    PhysicalStopTrigger = PhysicalStopTrigger.None,
                },
                rejected.ToArray());
        }

        // A candidate still waiting for its speaker verdict cannot be accepted; the identity gate
        // is the only thing that turns PendingSpeaker into an interruption.
        if (current.Authorization == TurnAuthorizationStatus.PendingSpeaker || current.Authorization.IsDenied())
            return Rejected(current);

        var accepted = new List<AssistantEffect>(5);
        // Defensive: every accepted path has already issued its stop through the identity gate;
        // this only closes a gap for a NotRequired candidate whose request predates the gate.
        TryIssuePhysicalStop(current, signal.OperationId, PhysicalStopTrigger.AcousticInterrupt, accepted);
        // The interrupted turn's playback can already have completed while this validation was
        // still outstanding, which leaves the snapshot in Idle with the validation turn intact.
        // The published transition therefore has to read the state being left instead of assuming
        // Speaking. EO-13 publishes it only when Listening is a real change, matching the
        // rejected branch above.
        if (current.ConversationState != AssistantState.Listening)
        {
            accepted.Add(new PublishStateChangedEffect(
                signal.TurnId,
                current.ConversationState,
                AssistantState.Listening,
                signal.OperationId));
        }

        accepted.Add(new PublishBargeInValidationEffect(
            signal.TurnId,
            BargeInValidationKind.Accepted,
            signal.Text,
            signal.Reason,
            signal.OperationId));
        return Accepted(
            current with
            {
                ConversationState = AssistantState.Listening,
                TurnId = signal.TurnId,
                ValidationTurnId = 0,
                ResponseLifecycle = ResponseLifecycle.None,
                Authorization = TurnAuthorizationStatus.NotRequired,
                CandidateId = 0,
                PhysicalStopIssued = false,
                PhysicalStopTrigger = PhysicalStopTrigger.None,
            },
            accepted.ToArray());
    }

    private static TurnDecision ReduceRecognitionFailed(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) ||
            current.ConversationState is not (AssistantState.Listening or AssistantState.Recognizing) ||
            signal.OperationId <= 0)
        {
            return Rejected(current);
        }

        return Accepted(
            current with
            {
                ConversationState = AssistantState.Idle,
                ResponseLifecycle = ResponseLifecycle.Failed,
                Authorization = current.ValidationTurnId == 0 ? TurnAuthorizationStatus.NotRequired : current.Authorization,
                CandidateId = current.ValidationTurnId == 0 ? 0 : current.CandidateId,
            },
            new RollbackHistoryEffect(signal.TurnId, signal.OperationId),
            new PublishStateChangedEffect(
                signal.TurnId,
                current.ConversationState,
                AssistantState.Idle,
                signal.OperationId));
    }

    private static TurnDecision ReduceOperationFailed(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) || signal.OperationId <= 0)
            return Rejected(current);

        return Accepted(
            current with { ConversationState = AssistantState.Idle, ResponseLifecycle = ResponseLifecycle.Failed },
            new RollbackHistoryEffect(signal.TurnId, signal.OperationId),
            new CancelOperationEffect(signal.TurnId, signal.OperationId),
            new PublishStateChangedEffect(
                signal.TurnId,
                current.ConversationState,
                AssistantState.Idle,
                signal.OperationId));
    }

    private static TurnDecision ReduceCancelled(TurnSnapshot current, TurnSignal signal)
    {
        if (!IsCurrent(current, signal) || signal.OperationId <= 0)
            return Rejected(current);

        bool wakeupInterrupt = string.Equals(
            signal.Reason,
            "wakeup-interrupt",
            StringComparison.Ordinal);
        if (wakeupInterrupt && signal.OperationId != current.NextTurnOrdinal)
            return Rejected(current);

        bool turnCompleted = string.Equals(
            signal.Reason,
            "turn-completed",
            StringComparison.Ordinal);
        if (turnCompleted)
        {
            if (current.ConversationState == AssistantState.Idle)
                return Rejected(current);

            return Accepted(
                current with
                {
                    ConversationState = AssistantState.Idle,
                    ResponseLifecycle = ResponseLifecycle.Completed,
                    // A denied or still-pending single-talk turn completes without a response; a
                    // concurrent interruption candidate keeps its own identity facts.
                    Authorization = current.ValidationTurnId == 0 ? TurnAuthorizationStatus.NotRequired : current.Authorization,
                    CandidateId = current.ValidationTurnId == 0 ? 0 : current.CandidateId,
                },
                new CompleteTurnEffect(signal.TurnId, signal.OperationId),
                new PublishStateChangedEffect(
                    signal.TurnId,
                    current.ConversationState,
                    AssistantState.Idle,
                    signal.OperationId));
        }

        var cancelled = new List<AssistantEffect>(6)
        {
            new CancelOperationEffect(signal.TurnId, signal.OperationId),
        };
        if (wakeupInterrupt && current.ConversationState is AssistantState.Speaking or AssistantState.Thinking)
        {
            // An authorized in-session wake word stops the interrupted reply through the same
            // owner-scoped playback effects as a barge-in; there is no module-side stop path.
            cancelled.Add(new StopPlaybackEffect(signal.TurnId, signal.OperationId));
            cancelled.Add(new ClearPlaybackQueueEffect(signal.TurnId, signal.OperationId));
        }
        cancelled.Add(new RollbackHistoryEffect(signal.TurnId, signal.OperationId));
        cancelled.Add(new CompleteTurnEffect(signal.TurnId, signal.OperationId));
        cancelled.Add(new PublishStateChangedEffect(
            signal.TurnId,
            current.ConversationState,
            AssistantState.Idle,
            signal.OperationId));
        return Accepted(
            current with
            {
                ConversationState = AssistantState.Idle,
                NextTurnOrdinal = wakeupInterrupt
                    ? checked(signal.OperationId + 1)
                    : current.NextTurnOrdinal,
                ResponseLifecycle = ResponseLifecycle.Cancelled,
                InterruptedTurnId = 0,
                ValidationTurnId = 0,
                Authorization = TurnAuthorizationStatus.NotRequired,
                CandidateId = 0,
                PhysicalStopIssued = false,
                PhysicalStopTrigger = PhysicalStopTrigger.None,
            },
            cancelled.ToArray());
    }

    private static TurnDecision ReduceRuntimeFault(TurnSnapshot current, TurnSignal signal)
        => Accepted(
            current with { ResponseLifecycle = ResponseLifecycle.Failed },
            new RollbackHistoryEffect(current.TurnId, signal.OperationId),
            new PublishFaultedEffect(current.TurnId, signal.Failure, signal.OperationId));

    private static bool IsCurrent(TurnSnapshot current, TurnSignal signal)
        => signal.TurnId > 0 && signal.TurnId == current.TurnId;

    private static TurnDecision Rejected(TurnSnapshot current)
        => new(current, Array.Empty<AssistantEffect>(), Accepted: false);

    private static TurnDecision Accepted(TurnSnapshot next, params AssistantEffect[] effects)
        => new(next, Array.AsReadOnly(effects), Accepted: true);
}
