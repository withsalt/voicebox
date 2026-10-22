using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AssistantTurnObservationStoreMutationTests
{
    [Fact]
    public void CandidatePredicates_RequireTheExactStateKindAndConfirmationCombination()
    {
        var idle = CreateTurns();
        Assert.False(idle.IsSpeakingWithoutPendingCandidate);

        var listeningDetector = new BargeInDetector(1, 0, 1000, 1);
        var listening = new AssistantTurnObservationStoreHarness(
            listeningDetector,
            new BargeInConfirmationSettings(1, 0));
        Assert.Equal(SpeechStartedDecisionKind.ListeningCandidate, listening.OnSpeechStarted(enableBargeIn: true).Kind);
        Assert.True(listeningDetector.AddAudio([0.5f]));
        Assert.False(listening.IsBargeInAwaitingSpeaker);
        Assert.Equal(SpeakingDecisionKind.None, listening.ResolveBargeInSpeakerGate(BargeInSpeakerGateStatus.Match).Kind);

        var bargeDetector = new BargeInDetector(100, 0, 1000, 1);
        var barge = new AssistantTurnObservationStoreHarness(bargeDetector);
        long turnId = barge.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(barge.TryTransitionForTurn(turnId, AssistantState.Thinking, out _, out _));
        Assert.Equal(SpeechStartedDecisionKind.BargeInCandidate, barge.OnSpeechStarted(enableBargeIn: true).Kind);
        Assert.False(barge.IsBargeInAwaitingSpeaker);
        Assert.Equal(SpeakingDecisionKind.None, barge.ResolveBargeInSpeakerGate(BargeInSpeakerGateStatus.Match).Kind);
        Assert.Equal(
            SpeechEndedDecisionKind.DiscardedBargeInCandidate,
            barge.OnSpeechEnded(TimeSpan.FromMilliseconds(10)).Kind);
    }

    [Fact]
    public void OnSpeaking_BuffersOnlyAnOpenListeningOrValidationWindow()
    {
        var wrongState = CreateTurns();
        long wrongStateTurn = wrongState.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(wrongState.TryTransitionForTurn(wrongStateTurn, AssistantState.Thinking, out _, out _));
        wrongState.OnSpeaking([0.25f]);
        Assert.True(wrongState.TryTransitionForTurn(wrongStateTurn, AssistantState.Listening, out _, out _));
        var wrongStateFeed = new List<float[]>();
        Assert.True(wrongState.TryAttachStartedAsrSession(wrongStateTurn, "wrong-state", wrongStateFeed.Add, out _, out _));
        Assert.Empty(wrongStateFeed);

        var afterEndpoint = CreateTurns();
        long endedTurn = afterEndpoint.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.Equal(SpeechEndedDecisionKind.WaitingForAsrStart, afterEndpoint.OnSpeechEnded(TimeSpan.FromMilliseconds(25)).Kind);
        afterEndpoint.OnSpeaking([0.5f]);
        var endedFeed = new List<float[]>();
        Assert.True(afterEndpoint.TryAttachStartedAsrSession(
            endedTurn,
            "ended",
            endedFeed.Add,
            out bool speechAlreadyEnded,
            out TimeSpan duration));
        Assert.True(speechAlreadyEnded);
        Assert.Equal(TimeSpan.FromMilliseconds(25), duration);
        Assert.Empty(endedFeed);
    }

    [Fact]
    public void OnSpeaking_RoutesANormalAsrSessionWithTheActiveTurnIdentity()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryAttachStartedAsrSession(turnId, "normal", _ => { }, out _, out _));

        SpeakingDecision decision = turns.OnSpeaking([0.2f]);

        Assert.Equal(SpeakingDecisionKind.FeedAsr, decision.Kind);
        Assert.Equal("normal", decision.AsrSessionId);
        Assert.Equal(turnId, decision.TurnId);
        Assert.Equal(0, decision.InterruptedTurnId);
        Assert.Null(decision.ResponseToCancel);
    }

    [Fact]
    public void ValidationTurn_IgnoresNewSpeechAndRoutesAudioOnlyToItsOwnAsrSession()
    {
        var turns = CreateTurns();
        using var response = new CancellationTokenSource();
        long active = StartSpeaking(turns, response);
        SpeakingDecision validation = ConfirmNearEnd(turns, [0.4f]);

        SpeechStartedDecision ignored = turns.OnSpeechStarted(enableBargeIn: true);
        Assert.Equal(SpeechStartedDecisionKind.Ignored, ignored.Kind);
        Assert.Equal(validation.TurnId, ignored.TurnId);
        Assert.True(turns.IsCurrentTurn(active));
        Assert.True(turns.IsValidatingBargeInTurn(validation.TurnId));
        Assert.True(turns.HasPendingBargeInCandidateOrValidation);
        Assert.False(turns.IsSpeakingWithoutPendingCandidate);

        SpeakingDecision pendingAudio = turns.OnSpeaking([0.5f]);
        Assert.Equal(SpeakingDecisionKind.None, pendingAudio.Kind);
        Assert.True(turns.TryAttachStartedAsrSession(
            validation.TurnId,
            "validation-asr",
            _ => { },
            out bool endedBeforeAttach,
            out _));
        Assert.False(endedBeforeAttach);

        SpeakingDecision feed = turns.OnSpeaking([0.6f]);
        Assert.Equal(SpeakingDecisionKind.FeedAsr, feed.Kind);
        Assert.Equal("validation-asr", feed.AsrSessionId);
        Assert.Equal(validation.TurnId, feed.TurnId);
        Assert.Equal(active, feed.InterruptedTurnId);
        Assert.Same(response, feed.ResponseToCancel);
    }

    [Fact]
    public void ValidationTurn_SpeechEndBeforeAndAfterAsrAttach_UsesTheCorrectRecognitionDecision()
    {
        var beforeAttach = CreateTurns();
        long beforeActive = StartSpeaking(beforeAttach, null);
        SpeakingDecision beforeValidation = ConfirmNearEnd(beforeAttach, [0.1f]);
        SpeechEndedDecision waiting = beforeAttach.OnSpeechEnded(TimeSpan.FromMilliseconds(123));

        Assert.Equal(SpeechEndedDecisionKind.WaitingForAsrStart, waiting.Kind);
        Assert.Equal(beforeValidation.TurnId, waiting.TurnId);
        Assert.True(beforeAttach.TryAttachStartedAsrSession(
            beforeValidation.TurnId,
            "before",
            _ => { },
            out bool speechAlreadyEnded,
            out TimeSpan duration));
        Assert.True(speechAlreadyEnded);
        Assert.Equal(TimeSpan.FromMilliseconds(123), duration);
        Assert.True(beforeAttach.IsCurrentTurn(beforeActive));

        var afterAttach = CreateTurns();
        StartSpeaking(afterAttach, null);
        SpeakingDecision afterValidation = ConfirmNearEnd(afterAttach, [0.2f]);
        Assert.True(afterAttach.TryAttachStartedAsrSession(afterValidation.TurnId, "after", _ => { }, out _, out _));
        SpeechEndedDecision ready = afterAttach.OnSpeechEnded(TimeSpan.FromMilliseconds(456));

        Assert.Equal(SpeechEndedDecisionKind.ReadyForRecognition, ready.Kind);
        Assert.Equal("after", ready.AsrSessionId);
        Assert.Equal(afterValidation.TurnId, ready.TurnId);
        Assert.Equal(TimeSpan.FromMilliseconds(456), ready.SpeechDuration);
    }

    [Fact]
    public void AsrAttachAndRecognizing_RejectMismatchesAndClearOnlyTheMatchingSession()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;

        Assert.False(turns.TryAttachStartedAsrSession(turnId + 1, "wrong", _ => { }, out _, out _));
        Assert.True(turns.TryAttachStartedAsrSession(turnId, "session", _ => { }, out _, out _));
        Assert.False(turns.TryEnterRecognizing(turnId, "other", out AssistantState wrongOld, out bool wrongValidation));
        Assert.Equal(AssistantState.Listening, wrongOld);
        Assert.False(wrongValidation);
        Assert.True(turns.TryEnterRecognizing(turnId, "session", out AssistantState old, out bool validation));
        Assert.Equal(AssistantState.Listening, old);
        Assert.False(validation);
        Assert.Equal(AssistantState.Recognizing, turns.State);
        Assert.False(turns.TryAttachStartedAsrSession(turnId, "late", _ => { }, out _, out _));

        var validationTurns = CreateTurns();
        StartSpeaking(validationTurns, null);
        SpeakingDecision candidate = ConfirmNearEnd(validationTurns, [0.3f]);
        Assert.True(validationTurns.TryAttachStartedAsrSession(candidate.TurnId, "validation", _ => { }, out _, out _));
        Assert.False(validationTurns.TryEnterRecognizing(candidate.TurnId, "wrong", out _, out _));
        Assert.True(validationTurns.TryEnterRecognizing(candidate.TurnId, "validation", out AssistantState validationOld, out bool isValidation));
        Assert.Equal(AssistantState.Speaking, validationOld);
        Assert.True(isValidation);
        // Validation ownership remains until its accepted response owns the turn.
        Assert.True(validationTurns.IsValidatingBargeInTurn(candidate.TurnId));
    }

    [Fact]
    public void AsrAttach_InitializesOutputsRejectsDuplicatesAndConsumesEndpointFlags()
    {
        var mismatched = CreateTurns();
        long active = mismatched.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.False(mismatched.TryAttachStartedAsrSession(
            active + 1,
            "wrong",
            _ => { },
            out bool mismatchedEnded,
            out TimeSpan mismatchedDuration));
        Assert.False(mismatchedEnded);
        Assert.Equal(TimeSpan.Zero, mismatchedDuration);

        var normal = CreateTurns();
        long normalTurn = normal.OnSpeechStarted(enableBargeIn: false).TurnId;
        normal.OnSpeechEnded(TimeSpan.FromMilliseconds(31));
        Assert.True(normal.TryAttachStartedAsrSession(normalTurn, "normal", _ => { }, out bool normalEnded, out _));
        Assert.True(normalEnded);
        Assert.False(GetField<bool>(normal, "_speechEndedBeforeAsrReady"));

        var validation = CreateTurns();
        StartSpeaking(validation, null);
        SpeakingDecision candidate = ConfirmNearEnd(validation, [0.3f]);
        validation.OnSpeechEnded(TimeSpan.FromMilliseconds(47));
        Assert.True(validation.TryAttachStartedAsrSession(candidate.TurnId, "validation", _ => { }, out bool validationEnded, out _));
        Assert.True(validationEnded);
        Assert.False(GetField<bool>(validation, "_validatingBargeInSpeechEndedBeforeAsrReady"));
        Assert.False(validation.TryAttachStartedAsrSession(candidate.TurnId, "duplicate", _ => { }, out _, out _));
    }

    [Fact]
    public void ResetAfterAsrStartFailure_RestoresOnlyTheApplicableTurn()
    {
        var normal = CreateTurns();
        long normalTurn = normal.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.False(normal.ResetListeningAfterAsrStartFailure(normalTurn + 1, out _, out _, out _));
        Assert.True(normal.ResetListeningAfterAsrStartFailure(normalTurn, out AssistantState normalOld, out AssistantState normalRestored, out bool normalValidation));
        Assert.Equal(AssistantState.Listening, normalOld);
        Assert.Equal(AssistantState.Idle, normalRestored);
        Assert.False(normalValidation);
        Assert.Equal(AssistantState.Idle, normal.State);

        var validation = CreateTurns();
        long active = StartSpeaking(validation, null);
        SpeakingDecision candidate = ConfirmNearEnd(validation, [0.4f]);
        Assert.True(validation.ResetListeningAfterAsrStartFailure(candidate.TurnId, out AssistantState validationOld, out AssistantState validationRestored, out bool isValidation));
        Assert.Equal(AssistantState.Speaking, validationOld);
        Assert.Equal(AssistantState.Speaking, validationRestored);
        Assert.True(isValidation);
        Assert.Equal(AssistantState.Speaking, validation.State);
        Assert.True(validation.IsCurrentTurn(active));
        Assert.False(validation.IsValidatingBargeInTurn(candidate.TurnId));
    }

    [Fact]
    public void ResetAfterAsrStartFailure_ClearsAllPendingAudioOwners()
    {
        var normal = CreateTurns();
        long normalTurn = normal.OnSpeechStarted(enableBargeIn: false).TurnId;
        normal.OnSpeaking([0.1f, 0.2f]);
        Assert.NotEmpty(GetPendingAudio(normal, "_pendingAsrAudio"));
        Assert.True(normal.ResetListeningAfterAsrStartFailure(normalTurn, out _, out _, out _));
        Assert.Empty(GetPendingAudio(normal, "_pendingAsrAudio"));

        var validation = CreateTurns();
        StartSpeaking(validation, null);
        SpeakingDecision candidate = ConfirmNearEnd(validation, [0.3f]);
        validation.OnSpeaking([0.4f]);
        Assert.NotEmpty(GetPendingAudio(validation, "_validatingBargeInPendingAudio"));
        Assert.True(validation.ResetListeningAfterAsrStartFailure(candidate.TurnId, out _, out _, out _));
        Assert.Empty(GetPendingAudio(validation, "_validatingBargeInPendingAudio"));
        Assert.False(validation.HasPendingBargeInCandidateOrValidation);
    }

    [Fact]
    public void TwoPhaseInterruptionObservations_DoNotMutateBeforeCommit()
    {
        var turns = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(0, 0, 16000, 1));
        using var response = new CancellationTokenSource();
        turns.SynchronizeKernelState(AssistantState.Speaking, 1);
        Assert.True(turns.TrySetResponseCtsForTurn(1, response));

        Assert.True(turns.TryObserveBargeInFromNearEndWindow(
            [[0.4f, 0.5f]],
            ConfirmedWindow(),
            proposedTurnId: 2,
            out SpeakingDecision confirmation));
        Assert.False(turns.IsValidatingBargeInTurn(2));
        Assert.Same(response, turns.GetResponseCtsForTurn(1));

        Assert.True(turns.CommitConfirmedBargeIn(confirmation, [[0.4f, 0.5f]]));
        Assert.True(turns.IsValidatingBargeInTurn(2));

        Assert.True(turns.TryObserveValidatedBargeInRejection(
            2,
            out AssistantState oldState,
            out AssistantState restoredState));
        Assert.Equal(AssistantState.Speaking, oldState);
        Assert.Equal(AssistantState.Speaking, restoredState);
        Assert.True(turns.IsValidatingBargeInTurn(2));

        Assert.True(turns.CommitValidatedBargeInRejected(2));
        Assert.False(turns.IsValidatingBargeInTurn(2));

        Assert.True(turns.TryObserveWakeupInterrupt(3, out WakeupInterruptDecision wakeup));
        Assert.Same(response, turns.GetResponseCtsForTurn(1));
        Assert.True(turns.CommitWakeupInterrupt(wakeup));
        Assert.Null(turns.GetResponseCtsForTurn(1));
    }

    [Fact]
    public void ResponseCtsOwnership_CancelDisposeAndQueryOnlyTheCurrentTurn()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        using var current = new CancellationTokenSource();
        using var unrelated = new CancellationTokenSource();

        Assert.False(turns.TrySetResponseCtsForTurn(turnId + 1, current));
        Assert.True(turns.TrySetResponseCtsForTurn(turnId, current));
        Assert.Same(current, turns.GetResponseCtsForTurn(turnId));
        Assert.Null(turns.GetResponseCtsForTurn(turnId + 1));
        turns.CancelResponseCts();
        Assert.True(current.IsCancellationRequested);

        turns.DisposeResponseCts(unrelated);
        Assert.Same(current, turns.GetResponseCtsForTurn(turnId));
        turns.DisposeResponseCts(current);
        Assert.Null(turns.GetResponseCtsForTurn(turnId));
        Assert.Throws<ObjectDisposedException>(() => _ = current.Token);
        turns.DisposeResponseCts(current);
    }

    [Fact]
    public void ValidatedResponseOwnership_RejectsPrematureAssignmentAndClearsPendingOnCompletion()
    {
        var turns = CreateTurns();
        long interrupted = StartSpeaking(turns, null);
        SpeakingDecision validation = ConfirmNearEnd(turns, [0.2f]);
        using var response = new CancellationTokenSource();

        Assert.False(turns.TrySetResponseCtsAndClearValidatedBargeIn(validation.TurnId, response));
        Assert.True(turns.TryAcceptValidatedBargeIn(
            validation.TurnId,
            response,
            clearValidationContext: false,
            out _,
            out _,
            out long interruptedTurn));
        Assert.Equal(interrupted, interruptedTurn);
        Assert.NotEmpty(GetPendingAudio(turns, "_validatingBargeInPendingAudio"));

        Assert.True(turns.TryCompleteAcceptedBargeInValidation(validation.TurnId, out CancellationTokenSource? completed));
        Assert.Same(response, completed);
        Assert.False(turns.IsValidatingBargeInTurn(validation.TurnId));
        Assert.Empty(GetPendingAudio(turns, "_validatingBargeInPendingAudio"));
    }

    [Fact]
    public void ValidatedBargeIn_AcceptanceAndCompletionMaintainSingleCurrentOwner()
    {
        var turns = CreateTurns();
        using var interruptedResponse = new CancellationTokenSource();
        long interrupted = StartSpeaking(turns, interruptedResponse);
        SpeakingDecision validation = ConfirmNearEnd(turns, [0.4f, 0.5f]);
        using var validationResponse = new CancellationTokenSource();
        using var replacementResponse = new CancellationTokenSource();

        Assert.False(turns.TryAcceptValidatedBargeIn(validation.TurnId + 1, validationResponse, clearValidationContext: false, out _, out _, out _));
        Assert.False(turns.TryCompleteAcceptedBargeInValidation(validation.TurnId, out _));
        Assert.True(turns.TryAcceptValidatedBargeIn(
            validation.TurnId,
            validationResponse,
            clearValidationContext: false,
            out AssistantState old,
            out CancellationTokenSource? responseToCancel,
            out long interruptedTurn));
        Assert.Equal(AssistantState.Speaking, old);
        Assert.Same(interruptedResponse, responseToCancel);
        Assert.Equal(interrupted, interruptedTurn);
        Assert.True(turns.IsCurrentTurn(validation.TurnId));
        Assert.True(turns.IsCurrentOrValidatingTurn(validation.TurnId));
        Assert.True(turns.IsValidatingBargeInTurn(validation.TurnId));

        Assert.False(turns.TrySetResponseCtsAndClearValidatedBargeIn(validation.TurnId + 1, replacementResponse));
        Assert.True(turns.TrySetResponseCtsAndClearValidatedBargeIn(validation.TurnId, replacementResponse));
        Assert.Same(replacementResponse, turns.GetResponseCtsForTurn(validation.TurnId));
        Assert.False(turns.IsValidatingBargeInTurn(validation.TurnId));
        Assert.Empty(GetPendingAudio(turns, "_validatingBargeInPendingAudio"));
        Assert.True(turns.TryCompleteAcceptedBargeInValidation(validation.TurnId, out CancellationTokenSource? completedResponse));
        Assert.Same(replacementResponse, completedResponse);
    }

    [Fact]
    public void WakeupAndTakeActiveSession_ClearCandidatesSessionsAndValidationOwnership()
    {
        var idle = CreateTurns();
        Assert.False(idle.TryInterruptForWakeup(out WakeupInterruptDecision none));
        Assert.Equal(default, none);

        var normal = CreateTurns();
        long normalTurn = normal.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(normal.TryAttachStartedAsrSession(normalTurn, "normal", _ => { }, out _, out _));
        Assert.Equal("normal", normal.TakeActiveAsrSessionAndClearPending());
        Assert.Null(normal.TakeActiveAsrSessionAndClearPending());
        Assert.True(normal.TryTransitionForTurn(normalTurn, AssistantState.Thinking, out _, out _));
        Assert.True(normal.TryInterruptForWakeup(out WakeupInterruptDecision normalInterrupt));
        Assert.Equal(AssistantState.Thinking, normalInterrupt.OldState);
        Assert.Equal(normalTurn, normalInterrupt.InterruptedTurnId);
        Assert.Null(normalInterrupt.AsrSessionIdToCleanup);
        Assert.Equal(AssistantState.Idle, normal.State);

        var sessionAtWakeup = CreateTurns();
        long sessionTurn = sessionAtWakeup.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(sessionAtWakeup.TryAttachStartedAsrSession(sessionTurn, "cleanup", _ => { }, out _, out _));
        Assert.True(sessionAtWakeup.TryInterruptForWakeup(out WakeupInterruptDecision sessionInterrupt));
        Assert.Equal("cleanup", sessionInterrupt.AsrSessionIdToCleanup);

        var validation = CreateTurns();
        using var response = new CancellationTokenSource();
        long active = StartSpeaking(validation, response);
        SpeakingDecision candidate = ConfirmNearEnd(validation, [0.7f]);
        Assert.True(validation.TryAttachStartedAsrSession(candidate.TurnId, "validation", _ => { }, out _, out _));
        Assert.Equal("validation", validation.TakeActiveAsrSessionAndClearPending());
        Assert.False(validation.IsValidatingBargeInTurn(candidate.TurnId));
        Assert.True(validation.TryInterruptForWakeup(out WakeupInterruptDecision validationInterrupt));
        Assert.Equal(active, validationInterrupt.InterruptedTurnId);
        Assert.Same(response, validationInterrupt.ResponseToCancel);
    }

    [Fact]
    public void SessionAndCancellationPrecedence_IsExplicitWhenBothOwnersArePresent()
    {
        var speaking = CreateTurns();
        long activeTurn = speaking.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(speaking.TryAttachStartedAsrSession(activeTurn, "current", _ => { }, out _, out _));
        using var currentResponse = new CancellationTokenSource();
        using var validationResponse = new CancellationTokenSource();
        Assert.True(speaking.TrySetResponseCtsForTurn(activeTurn, currentResponse));
        SetField(speaking, "_validatingBargeInTurnId", activeTurn + 1);
        SetField(speaking, "_validatingBargeInInterruptedTurnId", activeTurn);
        SetField(speaking, "_validatingBargeInAsrSessionId", "validation");
        SetField(speaking, "_validatingBargeInResponseCts", validationResponse);

        SpeakingDecision feed = speaking.OnSpeaking([0.1f]);
        Assert.Equal("validation", feed.AsrSessionId);
        Assert.Equal(activeTurn + 1, feed.TurnId);
        Assert.Same(validationResponse, feed.ResponseToCancel);

        Assert.True(speaking.TryInterruptForWakeup(out WakeupInterruptDecision wakeup));
        Assert.Same(validationResponse, wakeup.ResponseToCancel);
        Assert.Equal("current", wakeup.AsrSessionIdToCleanup);

        var take = CreateTurns();
        long takeTurn = take.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(take.TryAttachStartedAsrSession(takeTurn, "current-first", _ => { }, out _, out _));
        SetField(take, "_validatingBargeInTurnId", takeTurn + 1);
        SetField(take, "_validatingBargeInAsrSessionId", "validation-second");
        Assert.Equal("current-first", take.TakeActiveAsrSessionAndClearPending());
    }

    [Fact]
    public void CorrelationPredicates_RequireTheExactCurrentOrValidationOwner()
    {
        var turns = CreateTurns();
        long current = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        using var response = new CancellationTokenSource();
        using var other = new CancellationTokenSource();
        Assert.True(turns.TrySetResponseCtsForTurn(current, response));

        Assert.False(turns.TryClearResponseCtsForTurn(current + 1, response));
        Assert.False(turns.TryClearResponseCtsForTurn(current, other));
        Assert.True(turns.TryClearResponseCtsForTurn(current, response));

        SetField(turns, "_validatingBargeInTurnId", current + 1);
        Assert.True(turns.IsCurrentOrValidatingTurn(current));
        Assert.True(turns.IsCurrentOrValidatingTurn(current + 1));
        Assert.False(turns.IsCurrentOrValidatingTurn(current + 2));
    }

    [Fact]
    public void CompletedInterruptedTurn_RequiresBothValidationAndInterruptedCorrelation()
    {
        var turns = CreateTurns();
        long current = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TransitionTo(AssistantState.Idle, out _));

        SetField(turns, "_validatingBargeInTurnId", current + 1);
        SetField(turns, "_validatingBargeInInterruptedTurnId", current + 2);
        Assert.True(turns.TryCompleteTurn(current, out _, out _, out bool wrongInterrupted));
        Assert.False(wrongInterrupted);

        SetField(turns, "_validatingBargeInInterruptedTurnId", current);
        Assert.True(turns.TryCompleteTurn(current, out _, out _, out bool matched));
        Assert.True(matched);

        SetField(turns, "_validatingBargeInTurnId", 0L);
        Assert.True(turns.TryCompleteTurn(current, out _, out _, out bool missingValidation));
        Assert.False(missingValidation);
    }

    [Fact]
    public void ValidationAsrStartFailure_ClearsOnlyTheValidationContext()
    {
        var turns = CreateTurns();
        long interrupted = StartSpeaking(turns, null);
        SpeakingDecision validation = ConfirmNearEnd(turns, [0.2f]);

        Assert.True(turns.ResetListeningAfterAsrStartFailure(
            validation.TurnId,
            out AssistantState oldState,
            out AssistantState restoredState,
            out bool isValidationTurn));

        Assert.True(isValidationTurn);
        Assert.Equal(AssistantState.Speaking, oldState);
        Assert.Equal(AssistantState.Speaking, restoredState);
        Assert.True(turns.IsCurrentTurn(interrupted));
        Assert.False(turns.IsValidatingBargeInTurn(validation.TurnId));
    }

    [Fact]
    public void Wakeup_ClearsNormalValidationAndCandidateAudioOwners()
    {
        var normal = CreateTurns();
        normal.OnSpeechStarted(enableBargeIn: false);
        normal.OnSpeaking([0.1f]);
        Assert.True(normal.TryInterruptForWakeup(out _));
        Assert.Empty(GetPendingAudio(normal, "_pendingAsrAudio"));

        var validation = CreateTurns();
        StartSpeaking(validation, null);
        ConfirmNearEnd(validation, [0.2f]);
        validation.OnSpeaking([0.3f]);
        Assert.True(validation.TryInterruptForWakeup(out _));
        Assert.Empty(GetPendingAudio(validation, "_validatingBargeInPendingAudio"));

        var candidate = CreateTurns();
        long candidateTurn = candidate.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(candidate.TryTransitionForTurn(candidateTurn, AssistantState.Thinking, out _, out _));
        Assert.Equal(SpeechStartedDecisionKind.BargeInCandidate, candidate.OnSpeechStarted(enableBargeIn: true).Kind);
        Assert.True(candidate.HasPendingBargeInCandidateOrValidation);
        Assert.True(candidate.TryInterruptForWakeup(out _));
        Assert.False(candidate.HasPendingBargeInCandidateOrValidation);
    }

    [Fact]
    public void TakeActiveSession_ClearsEveryPendingAudioOwnerAndCandidate()
    {
        var normal = CreateTurns();
        normal.OnSpeechStarted(enableBargeIn: false);
        normal.OnSpeaking([0.1f]);
        Assert.Null(normal.TakeActiveAsrSessionAndClearPending());
        Assert.Empty(GetPendingAudio(normal, "_pendingAsrAudio"));

        var validation = CreateTurns();
        StartSpeaking(validation, null);
        ConfirmNearEnd(validation, [0.2f]);
        validation.OnSpeaking([0.3f]);
        Assert.Null(validation.TakeActiveAsrSessionAndClearPending());
        Assert.Empty(GetPendingAudio(validation, "_validatingBargeInPendingAudio"));
        Assert.False(validation.HasPendingBargeInCandidateOrValidation);

        var candidate = CreateTurns();
        long turnId = candidate.OnSpeechStarted(enableBargeIn: false).TurnId;
        candidate.TryTransitionForTurn(turnId, AssistantState.Thinking, out _, out _);
        candidate.OnSpeechStarted(enableBargeIn: true);
        Assert.Null(candidate.TakeActiveAsrSessionAndClearPending());
        Assert.False(candidate.HasPendingBargeInCandidateOrValidation);
    }

    [Fact]
    public void TransitionsAndCompletion_RejectLateOwnersAndPreserveTheInterruptedValidationContext()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryTransitionForTurn(turnId, AssistantState.Listening, out AssistantState sameOld, out bool sameChanged));
        Assert.Equal(AssistantState.Listening, sameOld);
        Assert.False(sameChanged);
        Assert.False(turns.TryTransitionForTurn(turnId + 1, AssistantState.Thinking, out _, out _));
        Assert.True(turns.TransitionTo(AssistantState.Thinking, out AssistantState transitionOld));
        Assert.Equal(AssistantState.Listening, transitionOld);
        Assert.False(turns.TransitionTo(AssistantState.Thinking, out _));

        Assert.True(turns.TryCompleteTurn(turnId, out _, out bool changed, out bool interrupted));
        Assert.True(changed);
        Assert.False(interrupted);
        Assert.True(turns.TryCompleteTurn(turnId, out _, out bool duplicateChanged, out _));
        Assert.False(duplicateChanged);
        long successor = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.False(turns.TryCompleteTurn(turnId, out _, out _, out bool lateInterrupted));
        Assert.False(lateInterrupted);
        Assert.Equal(turnId + 1, successor);

        var validating = CreateTurns();
        long active = StartSpeaking(validating, null);
        SpeakingDecision validation = ConfirmNearEnd(validating, [0.2f]);
        Assert.True(validating.TryCompleteTurn(active, out _, out _, out bool completedInterrupted));
        Assert.True(completedInterrupted);
        Assert.True(validating.TryRejectValidatedBargeIn(validation.TurnId, out _, out AssistantState restored));
        Assert.Equal(AssistantState.Idle, restored);
    }

    [Fact]
    public void TurnCompletion_ClearsNormalAndAcceptedValidationResources()
    {
        var normal = CreateTurns();
        long normalTurn = normal.OnSpeechStarted(enableBargeIn: false).TurnId;
        normal.OnSpeaking([0.1f]);
        Assert.True(normal.TryTransitionForTurn(normalTurn, AssistantState.Thinking, out _, out _));
        Assert.True(normal.TryCompleteTurn(normalTurn, out _, out _, out _));
        Assert.Empty(GetPendingAudio(normal, "_pendingAsrAudio"));

        var validation = CreateTurns();
        StartSpeaking(validation, null);
        SpeakingDecision candidate = ConfirmNearEnd(validation, [0.2f]);
        Assert.True(validation.TryAcceptValidatedBargeIn(
            candidate.TurnId,
            responseCts: null,
            clearValidationContext: false,
            out _,
            out _,
            out _));
        Assert.True(validation.TryCompleteTurn(candidate.TurnId, out _, out _, out bool completedInterrupted));
        Assert.False(completedInterrupted);
        Assert.False(validation.IsValidatingBargeInTurn(candidate.TurnId));
        Assert.Empty(GetPendingAudio(validation, "_validatingBargeInPendingAudio"));
    }

    [Fact]
    public void BeginningANewIdleTurn_DropsResourcesFromAnExternallyEndedTurn()
    {
        var pending = CreateTurns();
        long pendingTurn = pending.OnSpeechStarted(enableBargeIn: false).TurnId;
        pending.OnSpeaking([0.1f]);
        Assert.True(pending.TransitionTo(AssistantState.Idle, out AssistantState pendingOld));
        Assert.Equal(AssistantState.Listening, pendingOld);
        long successor = pending.OnSpeechStarted(enableBargeIn: false).TurnId;
        var feed = new List<float[]>();
        Assert.True(pending.TryAttachStartedAsrSession(successor, "successor", feed.Add, out _, out _));
        Assert.Empty(feed);

        var candidate = CreateTurns();
        long candidateTurn = candidate.OnSpeechStarted(enableBargeIn: false).TurnId;
        candidate.TryTransitionForTurn(candidateTurn, AssistantState.Thinking, out _, out _);
        candidate.OnSpeechStarted(enableBargeIn: true);
        Assert.True(candidate.HasPendingBargeInCandidateOrValidation);
        Assert.True(candidate.TransitionTo(AssistantState.Idle, out _));
        Assert.Equal(SpeechStartedDecisionKind.StartListening, candidate.OnSpeechStarted(enableBargeIn: false).Kind);
        Assert.False(candidate.HasPendingBargeInCandidateOrValidation);
    }

    [Fact]
    public void VadCandidate_EnteringSpeakingCreatesAValidationTurnWithOrderedAudio()
    {
        var turns = CreateTurns();
        using var response = new CancellationTokenSource();
        long interrupted = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TrySetResponseCtsForTurn(interrupted, response));
        Assert.True(turns.TryTransitionForTurn(interrupted, AssistantState.Thinking, out _, out _));
        Assert.Equal(SpeechStartedDecisionKind.BargeInCandidate, turns.OnSpeechStarted(enableBargeIn: true).Kind);
        Assert.True(turns.TryTransitionForTurn(interrupted, AssistantState.Speaking, out _, out _));

        SpeakingDecision validation = turns.OnSpeaking([0.4f]);
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, validation.Kind);
        Assert.Equal(interrupted + 1, validation.TurnId);
        Assert.Equal(interrupted, validation.InterruptedTurnId);
        Assert.Same(response, validation.ResponseToCancel);

        var feed = new List<float[]>();
        Assert.True(turns.TryAttachStartedAsrSession(validation.TurnId, "validation", feed.Add, out bool ended, out _));
        Assert.False(ended);
        Assert.Single(feed);
        Assert.Equal([0.4f], feed[0]);
        SpeechEndedDecision endpoint = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(20));
        Assert.Equal(SpeechEndedDecisionKind.ReadyForRecognition, endpoint.Kind);
        Assert.Equal("validation", endpoint.AsrSessionId);
    }

    [Fact]
    public void NearEndValidation_FiltersEmptyChunksAndReportsZeroStatsForEmptyAudio()
    {
        var filtered = CreateTurns();
        StartSpeaking(filtered, null);
        SpeakingDecision validation = ConfirmNearEnd(filtered, [], [0.5f]);
        var feed = new List<float[]>();
        Assert.True(filtered.TryAttachStartedAsrSession(validation.TurnId, "filtered", feed.Add, out _, out _));
        Assert.Single(feed);
        Assert.Equal([0.5f], feed[0]);

        var empty = CreateTurns();
        StartSpeaking(empty, null);
        SpeakingDecision emptyDecision = ConfirmNearEnd(empty, []);
        Assert.Equal(0, emptyDecision.ConfirmedCandidateSamples);
        Assert.Equal(0, emptyDecision.ConfirmedCandidateRms);
        Assert.Equal(0, emptyDecision.ConfirmedConfirmationSamples);
        Assert.Equal(0, emptyDecision.ConfirmedConfirmationRms);
    }

    [Fact]
    public void AcceptAndRejectValidation_ClearTheirOwnedPendingAudio()
    {
        var accepted = CreateTurns();
        StartSpeaking(accepted, null);
        SpeakingDecision acceptedValidation = ConfirmNearEnd(accepted, [0.2f]);
        Assert.True(accepted.TryAcceptValidatedBargeIn(
            acceptedValidation.TurnId,
            responseCts: null,
            clearValidationContext: true,
            out _,
            out _,
            out _));
        Assert.False(accepted.IsValidatingBargeInTurn(acceptedValidation.TurnId));
        Assert.Empty(GetPendingAudio(accepted, "_validatingBargeInPendingAudio"));

        var rejected = CreateTurns();
        StartSpeaking(rejected, null);
        SpeakingDecision rejectedValidation = ConfirmNearEnd(rejected, [0.3f]);
        Assert.True(rejected.TryRejectValidatedBargeIn(rejectedValidation.TurnId, out _, out _));
        Assert.False(rejected.HasPendingBargeInCandidateOrValidation);
        Assert.Empty(GetPendingAudio(rejected, "_validatingBargeInPendingAudio"));
    }

    [Fact]
    public void NullContracts_ThrowTheDocumentedArgumentExceptions()
    {
        var turns = CreateTurns();
        Assert.Throws<ArgumentNullException>(() => turns.TrySetResponseCtsAndClearValidatedBargeIn(1, null!));
        Assert.Throws<ArgumentNullException>(() => new AssistantTurnObservationStoreHarness(null!));
    }

    [Fact]
    public void KernelOwnedTurnProposals_RejectZeroAndNegativeIdentities()
    {
        var immediate = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(0, 0, 1000, 1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => immediate.OnSpeechStarted(enableBargeIn: false, proposedTurnId: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => immediate.OnSpeechStarted(enableBargeIn: false, proposedTurnId: -1));

        var listening = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(100, 0, 1000, 1),
            new BargeInConfirmationSettings(100, 0));
        Assert.Equal(
            SpeechStartedDecisionKind.ListeningCandidate,
            listening.OnSpeechStarted(enableBargeIn: false, proposedTurnId: 1).Kind);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => listening.OnSpeaking(
                Enumerable.Repeat(0.1f, 100).ToArray(),
                BargeInSpeakerGateStatus.Match,
                proposedTurnId: 0));

        var thinking = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(0, 0, 1000, 1));
        thinking.SynchronizeKernelState(AssistantState.Thinking, 1);
        Assert.Equal(
            SpeechStartedDecisionKind.BargeInCandidate,
            thinking.OnSpeechStarted(enableBargeIn: true, proposedTurnId: 2).Kind);
        Assert.Throws<ArgumentOutOfRangeException>(
            () => thinking.OnSpeaking(
                [0.1f],
                BargeInSpeakerGateStatus.Match,
                proposedTurnId: 0));

        var speaking = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(0, 0, 1000, 1));
        speaking.SynchronizeKernelState(AssistantState.Speaking, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            speaking.TryObserveBargeInFromNearEndWindow(
                [[0.1f]],
                ConfirmedWindow(),
                proposedTurnId: 0,
                out _));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => speaking.TryObserveWakeupInterrupt(operationOwnerTurnId: 0, out _));
    }

    private static AssistantTurnObservationStoreHarness CreateTurns()
        => new(new BargeInDetector(0, 0, 16000, 1));

    private static long StartSpeaking(AssistantTurnObservationStoreHarness turns, CancellationTokenSource? response)
    {
        long turnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        if (response is not null)
            Assert.True(turns.TrySetResponseCtsForTurn(turnId, response));
        Assert.True(turns.TryTransitionForTurn(turnId, AssistantState.Speaking, out _, out _));
        return turnId;
    }

    private static SpeakingDecision ConfirmNearEnd(AssistantTurnObservationStoreHarness turns, params float[][] audio)
    {
        Assert.True(turns.TryConfirmBargeInFromNearEndWindow(audio, ConfirmedWindow(), out SpeakingDecision decision));
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, decision.Kind);
        return decision;
    }

    private static BargeInDecision ConfirmedWindow()
        => new(
            Action: BargeInAction.Interrupt,
            FrameCount: 1,
            AcceptedFrameCount: 1,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 20,
            AcceptedDurationMs: 20,
            RequiredDurationMs: 0,
            ScoreDurationMs: 20,
            PeakScoreDurationMs: 20,
            LastFrameRms: 0.1,
            LastRequiredRms: 0,
            MinRequiredRms: 0,
            MaxRequiredRms: 0,
            Reason: "mutation-contract");

    private static List<float[]> GetPendingAudio(AssistantTurnObservationStoreHarness turns, string fieldName)
        => GetField<List<float[]>>(turns, fieldName);

    private static T GetField<T>(AssistantTurnObservationStoreHarness turns, string fieldName)
    {
        if (fieldName == "_responseCts")
            return Assert.IsType<T>(turns.Resources.GetOrCreate(turns.ActiveTurnId).ResponseCts);

        if (fieldName is "_validatingBargeInTurnId" or "_validatingBargeInInterruptedTurnId")
        {
            if (fieldName == "_validatingBargeInTurnId")
                return Assert.IsType<T>(turns.ValidationTurnId);
            return Assert.IsType<T>(turns.GetValidationTestField("_validatingBargeInInterruptedTurnId"));
        }

        if (fieldName == "_validatingBargeInAsrSessionId" ||
            fieldName is "_validatingBargeInSpeechEndedBeforeAsrReady" or
                "_validatingBargeInSpeechDuration" or
                "_validatingBargeInPendingAudio")
            return Assert.IsType<T>(turns.GetRecognitionTestField(fieldName));

        if (fieldName is "_validatingBargeInResponseCts")
            return Assert.IsType<T>(turns.GetValidationTestField(fieldName));

        if (fieldName is "_pendingAsrAudio" or "_currentAsrSessionId" or
            "_speechEndedBeforeAsrReady" or "_pendingSpeechDuration")
            return Assert.IsType<T>(turns.GetRecognitionTestField(fieldName));

        if (fieldName == "_validatingBargeInTurnId")
            return Assert.IsType<T>(turns.ValidationTurnId);
        if (fieldName == "_activeTurnId")
            return Assert.IsType<T>(turns.ActiveTurnId);

        throw new InvalidOperationException(
            $"'{fieldName}' is not a turn-resource field; the owner that holds it must expose it explicitly.");
    }

    private static void SetField<T>(
        AssistantTurnObservationStoreHarness turns,
        string fieldName,
        T value)
    {
        if (fieldName == "_responseCts")
        {
            turns.Resources.GetOrCreate(turns.ActiveTurnId).ResponseCts = value as CancellationTokenSource;
            return;
        }

        if (fieldName is "_validatingBargeInTurnId" or "_validatingBargeInInterruptedTurnId")
        {
            if (fieldName == "_validatingBargeInTurnId")
            {
                turns.SetValidationTurnId(Assert.IsType<long>(value));
                return;
            }

            turns.SetValidationInterruptedTurnId(Assert.IsType<long>(value));
            return;
        }

        if (fieldName == "_validatingBargeInAsrSessionId" ||
            fieldName is "_validatingBargeInSpeechEndedBeforeAsrReady" or
                "_validatingBargeInSpeechDuration" or
                "_validatingBargeInPendingAudio")
        {
            turns.SetRecognitionTestField(fieldName, value);
            return;
        }

        if (fieldName is "_validatingBargeInResponseCts")
        {
            turns.SetValidationTestField(fieldName, value);
            return;
        }

        if (fieldName is "_pendingAsrAudio" or "_currentAsrSessionId" or
            "_speechEndedBeforeAsrReady" or "_pendingSpeechDuration")
        {
            turns.SetRecognitionTestField(fieldName, value);
            return;
        }

        throw new InvalidOperationException(
            $"'{fieldName}' is not a turn-resource field; the owner that holds it must expose it explicitly.");
    }

}
