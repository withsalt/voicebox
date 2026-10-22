using System.Reflection;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Gates;
using Xunit;
using RuntimeTurnContextStore = VoiceBox.Assistant.Runtime.TurnContext.TurnContextStore;

namespace VoiceBox.Assistant.Tests;

public sealed class TurnKernelTests
{
    [Fact]
    public void Snapshot_IsImmutableAndContainsOnlyCanonicalTurnSemantics()
    {
        Type snapshot = typeof(TurnSnapshot);

        Assert.True(snapshot.IsSealed);
        Assert.Equal(
            [
                nameof(TurnSnapshot.ConversationState),
                nameof(TurnSnapshot.WakeMode),
                nameof(TurnSnapshot.TurnId),
                nameof(TurnSnapshot.NextTurnOrdinal),
                nameof(TurnSnapshot.InterruptedTurnId),
                nameof(TurnSnapshot.ValidationTurnId),
                nameof(TurnSnapshot.InputGateOpen),
                nameof(TurnSnapshot.ResponseLifecycle),
                nameof(TurnSnapshot.Authorization),
                nameof(TurnSnapshot.CandidateId),
                nameof(TurnSnapshot.PhysicalStopIssued),
                nameof(TurnSnapshot.PhysicalStopTrigger),
            ],
            snapshot.GetProperties(BindingFlags.Instance | BindingFlags.Public)
                .Select(static property => property.Name)
                .ToArray());
        // Identity facts in the snapshot are plain values: no speaker result, score or embedding.
        Assert.All(
            snapshot.GetProperties(BindingFlags.Instance | BindingFlags.Public),
            static property => Assert.True(property.PropertyType.IsValueType || property.PropertyType == typeof(string)));

        Assert.DoesNotContain(
            snapshot.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            static field => field.FieldType == typeof(CancellationTokenSource) ||
                typeof(Task).IsAssignableFrom(field.FieldType) ||
                field.FieldType.Name.Contains("Provider", StringComparison.Ordinal) ||
                field.FieldType.Name.Contains("Queue", StringComparison.Ordinal) ||
                field.FieldType.Name.Contains("Buffer", StringComparison.Ordinal));
    }

    [Fact]
    public void NormalTurn_ReducesToTheCanonicalStateAndEffectOrder()
    {
        TurnSnapshot state = TurnSnapshot.Initial with { InputGateOpen = true };
        TurnDecision[] decisions =
        [
            TurnKernel.Reduce(state, TurnSignal.SpeechStarted(1, 11)),
            TurnKernel.Reduce(
                TurnSnapshot.Initial with
                {
                    InputGateOpen = true,
                    ConversationState = AssistantState.Listening,
                    TurnId = 1,
                },
                TurnSignal.RecognitionSessionReady(1, 12)),
            TurnKernel.Reduce(
                TurnSnapshot.Initial with
                {
                    InputGateOpen = true,
                    ConversationState = AssistantState.Recognizing,
                    TurnId = 1,
                },
                TurnSignal.RecognitionCompleted(1, 13, "你好")),
            TurnKernel.Reduce(
                TurnSnapshot.Initial with
                {
                    InputGateOpen = true,
                    ConversationState = AssistantState.Thinking,
                    TurnId = 1,
                },
                TurnSignal.PlaybackStarted(1, 14)),
            TurnKernel.Reduce(
                TurnSnapshot.Initial with
                {
                    InputGateOpen = true,
                    ConversationState = AssistantState.Speaking,
                    TurnId = 1,
                },
                TurnSignal.ResponseGenerated(1, 15, "好的。")),
            TurnKernel.Reduce(
                TurnSnapshot.Initial with
                {
                    InputGateOpen = true,
                    ConversationState = AssistantState.Speaking,
                    TurnId = 1,
                    ResponseLifecycle = ResponseLifecycle.Generated,
                },
                TurnSignal.PlaybackCompleted(1, 16)),
        ];

        Assert.Equal(AssistantState.Listening, decisions[0].Next.ConversationState);
        Assert.Equal(AssistantState.Recognizing, decisions[1].Next.ConversationState);
        Assert.Equal(AssistantState.Thinking, decisions[2].Next.ConversationState);
        Assert.Equal(AssistantState.Speaking, decisions[3].Next.ConversationState);
        Assert.Equal(ResponseLifecycle.Generated, decisions[4].Next.ResponseLifecycle);
        Assert.Equal(AssistantState.Idle, decisions[5].Next.ConversationState);
        Assert.Collection(
            decisions[0].Effects,
            effect => Assert.IsType<PublishStateChangedEffect>(effect),
            effect => Assert.IsType<PrepareRecognitionEffect>(effect));
        Assert.Collection(
            decisions[2].Effects,
            effect => Assert.IsType<PublishUserSpokeEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect),
            effect => Assert.IsType<GenerateResponseEffect>(effect));
        Assert.Collection(
            decisions[5].Effects,
            effect => Assert.IsType<PublishAssistantRepliedEffect>(effect),
            effect => Assert.IsType<CompleteTurnEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect));
    }

    [Fact]
    public void Reducer_IsDeterministicAndDoesNotMutateInput()
    {
        TurnSnapshot input = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Recognizing,
            TurnId = 7,
            InputGateOpen = false,
        };
        TurnSignal signal = TurnSignal.RecognitionCompleted(7, 22, "text");

        TurnDecision first = TurnKernel.Reduce(input, signal);
        TurnDecision second = TurnKernel.Reduce(input, signal);

        Assert.Equal(input, TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Recognizing,
            TurnId = 7,
            InputGateOpen = false,
        });
        Assert.Equal(first.Next, second.Next);
        Assert.Equal(first.Accepted, second.Accepted);
        Assert.Equal(
            first.Effects.Select(static effect => (effect.Kind, effect.TurnId, effect.OperationId)),
            second.Effects.Select(static effect => (effect.Kind, effect.TurnId, effect.OperationId)));
        Assert.Equal(AssistantState.Recognizing, input.ConversationState);
    }

    [Fact]
    public void Reducer_RejectsWrongTurnAndOutOfStateCompletionWithoutEffects()
    {
        TurnSnapshot input = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Recognizing,
            TurnId = 7,
        };

        TurnDecision wrongTurn = TurnKernel.Reduce(
            input,
            TurnSignal.RecognitionCompleted(8, 22, "late"));
        TurnDecision outOfState = TurnKernel.Reduce(
            input with { ConversationState = AssistantState.Idle },
            TurnSignal.PlaybackCompleted(7, 23));

        Assert.Equal(input, wrongTurn.Next);
        Assert.Empty(wrongTurn.Effects);
        Assert.Equal(AssistantState.Idle, outOfState.Next.ConversationState);
        Assert.Empty(outOfState.Effects);
    }

    [Fact]
    public void SpeechEnded_RequiresCurrentRecognizingTurnPositiveOperationAndSessionId()
    {
        TurnSnapshot recognizing = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Recognizing,
            TurnId = 7,
        };

        AssertRejected(recognizing, TurnSignal.SpeechEnded(8, 17, TimeSpan.Zero, "asr-8"));
        AssertRejected(
            recognizing with { ConversationState = AssistantState.Listening },
            TurnSignal.SpeechEnded(7, 17, TimeSpan.Zero, "asr-7"));
        AssertRejected(recognizing, TurnSignal.SpeechEnded(7, 0, TimeSpan.Zero, "asr-7"));
        AssertRejected(recognizing, TurnSignal.SpeechEnded(7, 17, TimeSpan.Zero, string.Empty));

        TurnDecision accepted = TurnKernel.Reduce(
            recognizing,
            TurnSignal.SpeechEnded(7, 17, TimeSpan.Zero, "asr-7"));

        Assert.True(accepted.Accepted);
        Assert.Equal(
            "asr-7",
            Assert.IsType<FinishRecognitionEffect>(Assert.Single(accepted.Effects)).SessionId);
    }

    [Fact]
    public void Reducer_RejectsZeroOperationIdForEveryOperationCorrelatedSignal()
    {
        TurnSnapshot recognizing = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Recognizing,
            TurnId = 7,
        };
        TurnSnapshot thinking = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Thinking,
            TurnId = 7,
        };
        TurnSnapshot speaking = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Speaking,
            TurnId = 7,
        };

        AssertRejected(
            TurnSnapshot.Initial,
            TurnSignal.WakeupObserved(
                0,
                AssistantWakeMode.WaitingForWakeWord,
                AssistantWakeMode.ContinuousConversation,
                hasWakeup: true,
                keyword: "wake",
                score: 1,
                threshold: 1,
                durationMs: 1,
                startTimeMs: 1,
                KernelWakeupTriggerKind.InitialWakeup));
        AssertRejected(recognizing, TurnSignal.RecognitionCompleted(7, 0, "text"));
        AssertRejected(thinking, TurnSignal.ResponseGenerated(7, 0, "text"));
        AssertRejected(thinking, TurnSignal.PlaybackStarted(7, 0));
        AssertRejected(speaking, TurnSignal.PlaybackCompleted(7, 0));
        AssertRejected(recognizing, TurnSignal.RecognitionFailed(7, 0, Failure()));
        AssertRejected(thinking, TurnSignal.OperationFailed(7, 0, Failure()));
        AssertRejected(thinking, TurnSignal.CancelRequested(7, 0));
    }

    [Fact]
    public void Reducer_EnforcesEveryStateAndCorrelationGuardBeforeIssuingEffects()
    {
        TurnSnapshot idle = TurnSnapshot.Initial;
        TurnSnapshot listening = idle with { ConversationState = AssistantState.Listening, TurnId = 7 };
        TurnSnapshot recognizing = idle with { ConversationState = AssistantState.Recognizing, TurnId = 7 };
        TurnSnapshot thinking = idle with
        {
            ConversationState = AssistantState.Thinking,
            TurnId = 7,
            NextTurnOrdinal = 20,
        };
        TurnSnapshot speaking = idle with
        {
            ConversationState = AssistantState.Speaking,
            TurnId = 7,
            NextTurnOrdinal = 20,
        };

        AssertRejected(idle, TurnSignal.InputGateChanged(1, 1, true));
        TurnDecision gateOpened = TurnKernel.Reduce(idle with { InputGateOpen = false }, TurnSignal.InputGateChanged(2, 2, true));
        Assert.True(gateOpened.Accepted);
        Assert.True(gateOpened.Next.InputGateOpen);
        Assert.IsType<SetInputGateEffect>(Assert.Single(gateOpened.Effects));

        AssertRejected(idle, TurnSignal.SpeechStarted(0, 10));
        AssertRejected(idle, TurnSignal.SpeechStarted(-1, 10));
        AssertRejected(idle, TurnSignal.SpeechStarted(2, 10));
        AssertRejected(
            idle with { NextTurnOrdinal = 0 },
            TurnSignal.SpeechStarted(0, 10));
        AssertRejected(
            listening with { NextTurnOrdinal = 8 },
            TurnSignal.SpeechStarted(8, 10));
        TurnDecision started = TurnKernel.Reduce(idle, TurnSignal.SpeechStarted(1, 10));
        Assert.True(started.Accepted);
        Assert.Equal(2, started.Next.NextTurnOrdinal);
        AssertRejected(listening, TurnSignal.SpeechStarted(8, 10));
        AssertRejected(listening, TurnSignal.RecognitionSessionReady(8, 11));
        AssertRejected(listening, TurnSignal.RecognitionSessionReady(7, 0));
        AssertRejected(recognizing, TurnSignal.RecognitionCompleted(0, 12, "late"));

        TurnDecision recognized = TurnKernel.Reduce(recognizing, TurnSignal.RecognitionCompleted(7, 13, null!));
        Assert.True(recognized.Accepted);
        GenerateResponseEffect generate = Assert.IsType<GenerateResponseEffect>(recognized.Effects.Last());
        Assert.Equal(string.Empty, generate.Text);
        TurnDecision recognizedText = TurnKernel.Reduce(recognizing, TurnSignal.RecognitionCompleted(7, 13, "recognized"));
        Assert.Equal("recognized", Assert.IsType<GenerateResponseEffect>(recognizedText.Effects.Last()).Text);

        AssertRejected(thinking, TurnSignal.ResponseGenerated(8, 14, "late"));
        AssertRejected(idle, TurnSignal.ResponseGenerated(0, 14, "invalid"));
        TurnDecision generated = TurnKernel.Reduce(thinking, TurnSignal.ResponseGenerated(7, 15, null!));
        Assert.True(generated.Accepted);
        Assert.Equal(string.Empty, Assert.IsType<CommitHistoryEffect>(generated.Effects[0]).Text);
        Assert.Equal(string.Empty, Assert.IsType<PublishAssistantResponseGeneratedEffect>(generated.Effects[1]).Text);
        TurnDecision generatedText = TurnKernel.Reduce(thinking, TurnSignal.ResponseGenerated(7, 15, "generated"));
        Assert.Equal("generated", Assert.IsType<CommitHistoryEffect>(generatedText.Effects[0]).Text);
        Assert.Equal("generated", Assert.IsType<PublishAssistantResponseGeneratedEffect>(generatedText.Effects[1]).Text);

        AssertRejected(thinking, TurnSignal.PlaybackStarted(8, 16));
        AssertRejected(idle, TurnSignal.PlaybackStarted(0, 16));
        TurnDecision playbackStarted = TurnKernel.Reduce(thinking, TurnSignal.PlaybackStarted(7, 17));
        Assert.True(playbackStarted.Accepted);
        Assert.Equal(AssistantState.Speaking, playbackStarted.Next.ConversationState);
        PublishStateChangedEffect speakingEffect =
            Assert.IsType<PublishStateChangedEffect>(Assert.Single(playbackStarted.Effects));
        Assert.Equal((AssistantState.Thinking, AssistantState.Speaking), (speakingEffect.From, speakingEffect.To));
        TurnDecision playbackContinues = TurnKernel.Reduce(speaking, TurnSignal.PlaybackStarted(7, 18));
        Assert.True(playbackContinues.Accepted);
        Assert.Empty(playbackContinues.Effects);

        AssertRejected(speaking, TurnSignal.InterruptionRequested(8, 19));
        AssertRejected(listening, TurnSignal.InterruptionRequested(7, 19));
        AssertRejected(speaking, TurnSignal.InterruptionRequested(7, 0));
        AssertRejected(speaking, TurnSignal.InterruptionRequested(7, 21));
        AssertRejected(speaking, TurnSignal.InterruptionRequested(8, 20));
        AssertRejected(
            speaking with { NextTurnOrdinal = 0 },
            TurnSignal.InterruptionRequested(7, 0));
        // KWS disabled: the acoustic interrupt enters as NotRequired and stops playback at once,
        // exactly once, with no duck.
        TurnDecision interruptionRequested = TurnKernel.Reduce(speaking, TurnSignal.InterruptionRequested(7, 20));
        Assert.True(interruptionRequested.Accepted);
        Assert.Equal(20, interruptionRequested.Next.ValidationTurnId);
        Assert.Equal(21, interruptionRequested.Next.NextTurnOrdinal);
        Assert.Equal(TurnAuthorizationStatus.NotRequired, interruptionRequested.Next.Authorization);
        Assert.True(interruptionRequested.Next.PhysicalStopIssued);
        Assert.Equal(PhysicalStopTrigger.AcousticInterrupt, interruptionRequested.Next.PhysicalStopTrigger);
        // Device audio is stopped before the response source is cancelled, so the owner-scoped stop
        // always finds the reply still active.
        Assert.Collection(
            interruptionRequested.Effects,
            effect => Assert.Equal(AssistantEffectKind.StopPlayback, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.ClearPlaybackQueue, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.CancelResponse, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.PrepareRecognition, effect.Kind));
        Assert.All(interruptionRequested.Effects.Take(3), effect => Assert.Equal(7, effect.TurnId));
        TurnDecision interruptionHeard = TurnKernel.Reduce(
            speaking,
            TurnSignal.InterruptionRequested(7, 20, publishHeard: true));
        Assert.True(interruptionHeard.Accepted);
        Assert.Collection(
            interruptionHeard.Effects,
            effect => Assert.Equal(AssistantEffectKind.StopPlayback, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.ClearPlaybackQueue, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.CancelResponse, effect.Kind),
            effect => Assert.Equal(AssistantEffectKind.PrepareRecognition, effect.Kind),
            effect =>
            {
                PublishBargeInValidationEffect heard = Assert.IsType<PublishBargeInValidationEffect>(effect);
                Assert.Equal(BargeInValidationKind.Heard, heard.ValidationKind);
                Assert.Equal(20, heard.TurnId);
            });
        // A second request while one candidate is outstanding is rejected: one stop per owner.
        AssertRejected(interruptionRequested.Next, TurnSignal.InterruptionRequested(7, 21));

        // KWS enabled: the candidate enters as PendingSpeaker with no playback side effect at all.
        TurnDecision pendingSpeaker = TurnKernel.Reduce(
            speaking,
            TurnSignal.InterruptionRequested(
                7,
                20,
                publishHeard: true,
                authorization: TurnAuthorizationStatus.PendingSpeaker,
                candidateId: 41));
        Assert.True(pendingSpeaker.Accepted);
        Assert.Equal(TurnAuthorizationStatus.PendingSpeaker, pendingSpeaker.Next.Authorization);
        Assert.Equal(41, pendingSpeaker.Next.CandidateId);
        Assert.False(pendingSpeaker.Next.PhysicalStopIssued);
        Assert.Equal(AssistantState.Speaking, pendingSpeaker.Next.ConversationState);
        Assert.DoesNotContain(pendingSpeaker.Effects, static effect => effect.Kind is
            AssistantEffectKind.CancelResponse or AssistantEffectKind.StopPlayback or AssistantEffectKind.ClearPlaybackQueue);
        Assert.Collection(
            pendingSpeaker.Effects,
            effect => Assert.Equal(AssistantEffectKind.PrepareRecognition, effect.Kind),
            effect => Assert.IsType<PublishBargeInValidationEffect>(effect));
        AssertRejected(speaking, TurnSignal.InterruptionRequested(7, 20, authorization: TurnAuthorizationStatus.Authorized));

        TurnSnapshot validating = speaking with { InterruptedTurnId = 7, ValidationTurnId = 20 };
        AssertRejected(validating, TurnSignal.InterruptionResolved(19, 21, true));
        AssertRejected(speaking with { ValidationTurnId = 0 }, TurnSignal.InterruptionResolved(0, 21, true));
        AssertRejected(validating, TurnSignal.InterruptionResolved(20, 0, true));
        TurnDecision rejectedInterruption = TurnKernel.Reduce(validating, TurnSignal.InterruptionResolved(20, 22, false));
        Assert.True(rejectedInterruption.Accepted);
        Assert.Equal(0, rejectedInterruption.Next.ValidationTurnId);
        Assert.Equal(0, rejectedInterruption.Next.InterruptedTurnId);
        // Nothing is restored: a denied candidate never touched the interrupted reply.
        Assert.Collection(
            rejectedInterruption.Effects,
            effect =>
            {
                PublishBargeInValidationEffect rejected = Assert.IsType<PublishBargeInValidationEffect>(effect);
                Assert.Equal(BargeInValidationKind.Rejected, rejected.ValidationKind);
                Assert.Equal(20, rejected.TurnId);
            });

        TurnDecision restoredInterruption = TurnKernel.Reduce(
            validating,
            TurnSignal.InterruptionResolved(
                20,
                22,
                accepted: false,
                restoreState: AssistantState.Thinking));
        Assert.True(restoredInterruption.Accepted);
        Assert.Collection(
            restoredInterruption.Effects,
            effect => Assert.IsType<PublishBargeInValidationEffect>(effect),
            effect =>
            {
                PublishStateChangedEffect state = Assert.IsType<PublishStateChangedEffect>(effect);
                Assert.Equal((AssistantState.Speaking, AssistantState.Thinking), (state.From, state.To));
            });

        TurnDecision turnCompleted = TurnKernel.Reduce(
            thinking,
            TurnSignal.CancelRequested(7, 25, "turn-completed"));
        Assert.True(turnCompleted.Accepted);
        Assert.Equal(AssistantState.Idle, turnCompleted.Next.ConversationState);
        Assert.Equal(ResponseLifecycle.Completed, turnCompleted.Next.ResponseLifecycle);
        Assert.Collection(
            turnCompleted.Effects,
            effect => Assert.IsType<CompleteTurnEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect));
        Assert.DoesNotContain(turnCompleted.Effects, static effect => effect is RollbackHistoryEffect);

        // AuthorizationResolved is only meaningful for a PendingSpeaker candidate of the same
        // validation turn and candidate; anywhere else it is deterministically rejected.
        AssertRejected(idle, TurnSignal.AuthorizationResolved(0, 30, TurnAuthorizationStatus.Authorized, 0));
        AssertRejected(listening, TurnSignal.AuthorizationResolved(7, 30, TurnAuthorizationStatus.Authorized, 0));
        AssertRejected(speaking, TurnSignal.AuthorizationResolved(7, 30, TurnAuthorizationStatus.DeniedUnavailable, 0, "timeout"));
        AssertRejected(validating, TurnSignal.AuthorizationResolved(20, 30, TurnAuthorizationStatus.Authorized, 0));
        TurnSnapshot pendingCandidate = pendingSpeaker.Next;
        AssertRejected(pendingCandidate, TurnSignal.AuthorizationResolved(19, 30, TurnAuthorizationStatus.Authorized, 41));
        AssertRejected(pendingCandidate, TurnSignal.AuthorizationResolved(20, 30, TurnAuthorizationStatus.Authorized, 42));
        AssertRejected(pendingCandidate, TurnSignal.AuthorizationResolved(20, 30, TurnAuthorizationStatus.PendingSpeaker, 41));
        // Accepting a still-pending candidate is impossible: the identity gate must resolve first.
        AssertRejected(pendingCandidate, TurnSignal.InterruptionResolved(20, 30, accepted: true));

        // Same-candidate Match: Authorized, and the single owner-scoped stop is issued now — no DTD
        // Interrupt is required.
        TurnDecision authorized = TurnKernel.Reduce(
            pendingCandidate,
            TurnSignal.AuthorizationResolved(20, 31, TurnAuthorizationStatus.Authorized, 41, "speaker-match"));
        Assert.True(authorized.Accepted);
        Assert.Equal(TurnAuthorizationStatus.Authorized, authorized.Next.Authorization);
        Assert.True(authorized.Next.PhysicalStopIssued);
        Assert.Equal(PhysicalStopTrigger.SpeakerMatch, authorized.Next.PhysicalStopTrigger);
        Assert.Equal(AssistantState.Speaking, authorized.Next.ConversationState);
        Assert.Collection(
            authorized.Effects,
            effect => Assert.Equal((AssistantEffectKind.StopPlayback, 7L), (effect.Kind, effect.TurnId)),
            effect => Assert.Equal((AssistantEffectKind.ClearPlaybackQueue, 7L), (effect.Kind, effect.TurnId)),
            effect => Assert.Equal((AssistantEffectKind.CancelResponse, 7L), (effect.Kind, effect.TurnId)));
        // A duplicate Match for the same candidate is rejected and issues nothing more.
        AssertRejected(authorized.Next, TurnSignal.AuthorizationResolved(20, 32, TurnAuthorizationStatus.Authorized, 41));
        TurnDecision authorizedAccepted = TurnKernel.Reduce(authorized.Next, TurnSignal.InterruptionResolved(20, 33, accepted: true, reason: "speaker-match"));
        Assert.True(authorizedAccepted.Accepted);
        Assert.DoesNotContain(authorizedAccepted.Effects, static effect => effect.Kind is
            AssistantEffectKind.CancelResponse or AssistantEffectKind.StopPlayback or AssistantEffectKind.ClearPlaybackQueue);
        Assert.Equal(AssistantState.Listening, authorizedAccepted.Next.ConversationState);
        Assert.Equal(TurnAuthorizationStatus.NotRequired, authorizedAccepted.Next.Authorization);
        Assert.False(authorizedAccepted.Next.PhysicalStopIssued);

        // Denied verdicts record the denial and leave the reply alone; the rejection then closes
        // the candidate without any playback effect.
        foreach (TurnAuthorizationStatus denied in new[]
        {
            TurnAuthorizationStatus.DeniedMismatch,
            TurnAuthorizationStatus.DeniedInsufficientEvidence,
            TurnAuthorizationStatus.DeniedUnavailable,
        })
        {
            TurnDecision deniedDecision = TurnKernel.Reduce(
                pendingCandidate,
                TurnSignal.AuthorizationResolved(20, 34, denied, 41));
            Assert.True(deniedDecision.Accepted);
            Assert.Equal(denied, deniedDecision.Next.Authorization);
            Assert.Empty(deniedDecision.Effects);
            Assert.False(deniedDecision.Next.PhysicalStopIssued);
            AssertRejected(deniedDecision.Next, TurnSignal.InterruptionResolved(20, 35, accepted: true));
            TurnDecision closed = TurnKernel.Reduce(deniedDecision.Next, TurnSignal.InterruptionResolved(20, 35, accepted: false, reason: "command-denied"));
            Assert.True(closed.Accepted);
            Assert.Equal(AssistantState.Speaking, closed.Next.ConversationState);
            Assert.Single(closed.Effects);
            Assert.IsType<PublishBargeInValidationEffect>(closed.Effects[0]);
        }

        // Playback that finished on its own while the candidate was pending leaves nothing to stop.
        TurnDecision idleWhilePending = TurnKernel.Reduce(pendingCandidate, TurnSignal.PlaybackCompleted(7, 36));
        Assert.True(idleWhilePending.Accepted);
        TurnDecision authorizedAfterIdle = TurnKernel.Reduce(
            idleWhilePending.Next,
            TurnSignal.AuthorizationResolved(20, 37, TurnAuthorizationStatus.Authorized, 41));
        Assert.True(authorizedAfterIdle.Accepted);
        Assert.Empty(authorizedAfterIdle.Effects);
        Assert.False(authorizedAfterIdle.Next.PhysicalStopIssued);

        foreach (TurnSnapshot state in new[] { listening, recognizing })
        {
            TurnDecision recognitionFailed = TurnKernel.Reduce(
                state,
                TurnSignal.RecognitionFailed(7, 23, Failure()));
            Assert.True(recognitionFailed.Accepted);
            Assert.Equal(AssistantState.Idle, recognitionFailed.Next.ConversationState);
            Assert.Equal(ResponseLifecycle.Failed, recognitionFailed.Next.ResponseLifecycle);
        }

        AssertRejected(thinking, TurnSignal.RecognitionFailed(7, 23, Failure()));

        TurnDecision operationFailed = TurnKernel.Reduce(thinking, TurnSignal.OperationFailed(7, 24, Failure()));
        Assert.True(operationFailed.Accepted);
        Assert.Collection(
            operationFailed.Effects,
            effect => Assert.IsType<RollbackHistoryEffect>(effect),
            effect => Assert.IsType<CancelOperationEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect));
        AssertRejected(thinking, TurnSignal.OperationFailed(8, 24, Failure()));

        TurnDecision cancelled = TurnKernel.Reduce(thinking, TurnSignal.CancelRequested(7, 25));
        Assert.True(cancelled.Accepted);
        Assert.Collection(
            cancelled.Effects,
            effect => Assert.IsType<CancelOperationEffect>(effect),
            effect => Assert.IsType<RollbackHistoryEffect>(effect),
            effect => Assert.IsType<CompleteTurnEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect));
        AssertRejected(thinking, TurnSignal.CancelRequested(8, 25));
        AssertRejected(
            idle with { ConversationState = AssistantState.Recognizing, TurnId = 0 },
            TurnSignal.RecognitionCompleted(0, 26, "unowned"));
    }

    [Fact]
    public void AcceptedBargeInRecognition_CompletesThroughTheCanonicalResponseEffect()
    {
        TurnSnapshot validating = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Speaking,
            TurnId = 7,
            InterruptedTurnId = 7,
            ValidationTurnId = 20,
            ResponseLifecycle = ResponseLifecycle.Playing,
        };

        TurnDecision interruption = TurnKernel.Reduce(
            validating,
            TurnSignal.InterruptionResolved(20, 21, accepted: true, reason: "recognized", text: "停一下"));

        Assert.True(interruption.Accepted);
        Assert.Equal(AssistantState.Listening, interruption.Next.ConversationState);
        Assert.Equal(20, interruption.Next.TurnId);
        // A validation snapshot whose request predates the identity gate (no stop recorded yet)
        // still gets exactly one owner-scoped stop, addressed to the interrupted turn.
        Assert.Collection(
            interruption.Effects,
            effect => Assert.Equal(7, Assert.IsType<StopPlaybackEffect>(effect).TurnId),
            effect => Assert.Equal(7, Assert.IsType<ClearPlaybackQueueEffect>(effect).TurnId),
            effect => Assert.Equal(7, Assert.IsType<CancelResponseEffect>(effect).TurnId),
            effect =>
            {
                PublishStateChangedEffect state = Assert.IsType<PublishStateChangedEffect>(effect);
                Assert.Equal((AssistantState.Speaking, AssistantState.Listening), (state.From, state.To));
            },
            effect =>
            {
                // The Accepted publication follows the StateChanged effect, matching the required
                // public-event order EO-13 (StateChanged(→Listening) precedes Accepted).
                PublishBargeInValidationEffect accepted = Assert.IsType<PublishBargeInValidationEffect>(effect);
                Assert.Equal(BargeInValidationKind.Accepted, accepted.ValidationKind);
                Assert.Equal("recognized", accepted.Reason);
                Assert.Equal("停一下", accepted.Text);
            });

        TurnDecision recognition = TurnKernel.Reduce(
            interruption.Next,
            TurnSignal.RecognitionCompleted(20, 22, "停一下"));

        Assert.True(recognition.Accepted);
        Assert.Equal(AssistantState.Thinking, recognition.Next.ConversationState);
        Assert.Collection(
            recognition.Effects,
            effect => Assert.IsType<PublishUserSpokeEffect>(effect),
            effect => Assert.IsType<PublishStateChangedEffect>(effect),
            effect => Assert.IsType<GenerateResponseEffect>(effect));
        PublishStateChangedEffect recognitionState =
            Assert.IsType<PublishStateChangedEffect>(recognition.Effects[1]);
        Assert.Equal(
            (AssistantState.Listening, AssistantState.Thinking),
            (recognitionState.From, recognitionState.To));
    }

    [Fact]
    public void AcceptedBargeIn_AfterInterruptedPlaybackCompleted_PublishesTheStateItActuallyLeaves()
    {
        // Playback completion for the interrupted turn clears Speaking but deliberately keeps the
        // outstanding validation, so an acceptance that arrives afterwards must not report a
        // Speaking origin it never observed (EO-18 forbids inventing a transition).
        TurnSnapshot validating = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Speaking,
            TurnId = 7,
            NextTurnOrdinal = 21,
            InterruptedTurnId = 7,
            ValidationTurnId = 20,
            ResponseLifecycle = ResponseLifecycle.Playing,
        };

        TurnDecision playbackCompleted = TurnKernel.Reduce(
            validating,
            TurnSignal.PlaybackCompleted(7, 30, "好的。", TimeSpan.FromMilliseconds(40)));

        Assert.True(playbackCompleted.Accepted);
        Assert.Equal(AssistantState.Idle, playbackCompleted.Next.ConversationState);
        // The validation this acceptance belongs to survives the completed playback.
        Assert.Equal(20, playbackCompleted.Next.ValidationTurnId);

        TurnDecision accepted = TurnKernel.Reduce(
            playbackCompleted.Next,
            TurnSignal.InterruptionResolved(20, 31, accepted: true, reason: "recognized", text: "停一下"));

        Assert.True(accepted.Accepted);
        Assert.Equal(AssistantState.Listening, accepted.Next.ConversationState);
        Assert.Equal(20, accepted.Next.TurnId);
        Assert.Equal(0, accepted.Next.ValidationTurnId);
        PublishStateChangedEffect state = Assert.IsType<PublishStateChangedEffect>(
            accepted.Effects.Single(static effect => effect is PublishStateChangedEffect));
        Assert.Equal((AssistantState.Idle, AssistantState.Listening), (state.From, state.To));
        Assert.NotEqual(state.From, state.To);
    }

    [Fact]
    public void ThinkingBargeIn_CancelsThePriorResponseAndStartsTheReplacementRecognitionTurn()
    {
        TurnSnapshot thinking = TurnSnapshot.Initial with
        {
            ConversationState = AssistantState.Thinking,
            TurnId = 7,
            NextTurnOrdinal = 20,
            ResponseLifecycle = ResponseLifecycle.Generating,
        };

        TurnDecision decision = TurnKernel.Reduce(
            thinking,
            TurnSignal.InterruptionRequested(7, 20));

        Assert.True(decision.Accepted);
        Assert.Equal(AssistantState.Listening, decision.Next.ConversationState);
        Assert.Equal(20, decision.Next.TurnId);
        Assert.Equal(21, decision.Next.NextTurnOrdinal);
        Assert.Collection(
            decision.Effects,
            effect => Assert.IsType<CancelResponseEffect>(effect),
            effect => Assert.IsType<RollbackHistoryEffect>(effect),
            effect =>
            {
                PublishStateChangedEffect state = Assert.IsType<PublishStateChangedEffect>(effect);
                Assert.Equal((AssistantState.Thinking, AssistantState.Listening), (state.From, state.To));
            },
            effect => Assert.IsType<PrepareRecognitionEffect>(effect));
    }

    [Fact]
    public void SignalAndEffects_AreClosedTypedOwnedValues()
    {
        Assert.Equal(17, Enum.GetValues<TurnSignalKind>().Length);
        // Duck/Restore are gone: playback has exactly one physical side effect family (stop/clear).
        Assert.Equal(12, Enum.GetValues<AssistantEffectKind>().Length);
        Assert.DoesNotContain(Enum.GetNames<AssistantEffectKind>(), static name => name.Contains("Duck", StringComparison.Ordinal));
        Assert.DoesNotContain(Enum.GetNames<AssistantEffectKind>(), static name => name.Contains("Restore", StringComparison.Ordinal));
        Assert.DoesNotContain(
            typeof(TurnSignal).GetProperties(BindingFlags.Instance | BindingFlags.Public),
            static property => property.PropertyType == typeof(Exception) ||
                property.PropertyType == typeof(CancellationTokenSource) ||
                typeof(Task).IsAssignableFrom(property.PropertyType));
        Assert.DoesNotContain(
            typeof(AssistantEffect).Assembly.GetTypes().Where(static type =>
                type.Namespace == "VoiceBox.Assistant.Kernel" &&
                type.Name.EndsWith("Effect", StringComparison.Ordinal)),
            static type => type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Any(field => field.FieldType == typeof(CancellationTokenSource) ||
                    typeof(Task).IsAssignableFrom(field.FieldType)));
    }

    [Fact]
    public void TurnContextStore_OwnsCorrelationResourcesExactlyOnce()
    {
        using var store = new RuntimeTurnContextStore(new OperationCorrelationGate());
        using var response = new CancellationTokenSource();

        Assert.True(store.TryCreate(7));
        Assert.False(store.TryCreate(7));
        Assert.True(store.TryAttachAsrSession(7, 11, "asr-7"));
        Assert.False(store.TryAttachAsrSession(7, 12, "duplicate"));
        Assert.True(store.TryCompleteAsrSession(7, 11, out string? session));
        Assert.Equal("asr-7", session);
        Assert.False(store.TryCompleteAsrSession(7, 11, out _));

        Assert.True(store.TrySetResponseCts(7, response));
        using var duplicate = new CancellationTokenSource();
        Assert.False(store.TrySetResponseCts(7, duplicate));
        Assert.True(store.TryTakeResponseCts(7, out CancellationTokenSource? taken));
        Assert.Same(response, taken);
        taken!.Dispose();
        Assert.True(store.TryRemove(7));
        Assert.False(store.TryRemove(7));

        var ownedByStore = new CancellationTokenSource();
        Assert.True(store.TryCreate(8));
        Assert.True(store.TrySetResponseCts(8, ownedByStore));
        Assert.True(store.TryRemove(8));
        Assert.Throws<ObjectDisposedException>(() => _ = ownedByStore.Token);
    }

    [Fact]
    public void SingleTalkTurn_PendingSpeakerGatesRecognitionCompletionUntilTheOwnerMatch()
    {
        TurnSnapshot idle = TurnSnapshot.Initial with { InputGateOpen = true, NextTurnOrdinal = 7 };

        // A plain Listening turn keeps the NotRequired path untouched.
        TurnDecision plain = TurnKernel.Reduce(idle, TurnSignal.SpeechStarted(7, 10));
        Assert.True(plain.Accepted);
        Assert.Equal(TurnAuthorizationStatus.NotRequired, plain.Next.Authorization);
        Assert.True(TurnKernel.Reduce(
            plain.Next with { ConversationState = AssistantState.Recognizing },
            TurnSignal.RecognitionCompleted(7, 11, "你好")).Accepted);

        // Only NotRequired and PendingSpeaker may open a turn.
        AssertRejected(idle, TurnSignal.SpeechStarted(7, 10, authorization: TurnAuthorizationStatus.Authorized));
        AssertRejected(idle, TurnSignal.SpeechStarted(7, 10, authorization: TurnAuthorizationStatus.DeniedMismatch));

        // SingleTalk effective: the turn opens as PendingSpeaker with the same effects and no
        // playback side effect of any kind.
        TurnDecision pending = TurnKernel.Reduce(
            idle,
            TurnSignal.SpeechStarted(7, 10, authorization: TurnAuthorizationStatus.PendingSpeaker));
        Assert.True(pending.Accepted);
        Assert.Equal(AssistantState.Listening, pending.Next.ConversationState);
        Assert.Equal(TurnAuthorizationStatus.PendingSpeaker, pending.Next.Authorization);
        Assert.Equal(0, pending.Next.ValidationTurnId);
        Assert.Collection(
            pending.Effects,
            effect => Assert.IsType<PublishStateChangedEffect>(effect),
            effect => Assert.IsType<PrepareRecognitionEffect>(effect));

        TurnSnapshot listening = pending.Next;
        TurnSnapshot recognizing = TurnKernel.Reduce(listening, TurnSignal.RecognitionSessionReady(7, 12)).Next;
        Assert.Equal(AssistantState.Recognizing, recognizing.ConversationState);
        Assert.Equal(TurnAuthorizationStatus.PendingSpeaker, recognizing.Authorization);

        // ASR text never substitutes for the verdict: a pending turn cannot complete recognition.
        AssertRejected(listening, TurnSignal.RecognitionCompleted(7, 13, "你好"));
        AssertRejected(recognizing, TurnSignal.RecognitionCompleted(7, 13, "你好"));

        // The verdict targets the current turn only, and PendingSpeaker is not a resolution.
        AssertRejected(recognizing, TurnSignal.AuthorizationResolved(8, 14, TurnAuthorizationStatus.Authorized, 0));
        AssertRejected(recognizing, TurnSignal.AuthorizationResolved(7, 14, TurnAuthorizationStatus.PendingSpeaker, 0));
        AssertRejected(recognizing, TurnSignal.AuthorizationResolved(7, 14, TurnAuthorizationStatus.NotRequired, 0));

        // Match in either state records Authorized without any effect; recognition then proceeds
        // through the canonical UserSpoke → StateChanged → GenerateResponse order.
        foreach (TurnSnapshot state in new[] { listening, recognizing })
        {
            TurnDecision authorized = TurnKernel.Reduce(
                state,
                TurnSignal.AuthorizationResolved(7, 14, TurnAuthorizationStatus.Authorized, 0, "speaker-match"));
            Assert.True(authorized.Accepted);
            Assert.Empty(authorized.Effects);
            Assert.Equal(TurnAuthorizationStatus.Authorized, authorized.Next.Authorization);
            Assert.Equal(state.ConversationState, authorized.Next.ConversationState);
            Assert.False(authorized.Next.PhysicalStopIssued);
            AssertRejected(authorized.Next, TurnSignal.AuthorizationResolved(7, 15, TurnAuthorizationStatus.Authorized, 0));

            TurnDecision completed = TurnKernel.Reduce(authorized.Next, TurnSignal.RecognitionCompleted(7, 16, "你好"));
            Assert.True(completed.Accepted);
            Assert.Equal(AssistantState.Thinking, completed.Next.ConversationState);
            Assert.Collection(
                completed.Effects,
                effect => Assert.IsType<PublishUserSpokeEffect>(effect),
                effect => Assert.IsType<PublishStateChangedEffect>(effect),
                effect => Assert.IsType<GenerateResponseEffect>(effect));
        }

        // Denied verdicts only record the denial: the utterance never becomes a semantic turn and
        // the turn completes without a response, resetting the identity facts for the next turn.
        foreach (TurnAuthorizationStatus denied in new[]
        {
            TurnAuthorizationStatus.DeniedMismatch,
            TurnAuthorizationStatus.DeniedInsufficientEvidence,
            TurnAuthorizationStatus.DeniedUnavailable,
        })
        {
            TurnDecision deniedDecision = TurnKernel.Reduce(
                recognizing,
                TurnSignal.AuthorizationResolved(7, 17, denied, 0, "command-denied"));
            Assert.True(deniedDecision.Accepted);
            Assert.Empty(deniedDecision.Effects);
            Assert.Equal(denied, deniedDecision.Next.Authorization);
            Assert.Equal(AssistantState.Recognizing, deniedDecision.Next.ConversationState);
            AssertRejected(deniedDecision.Next, TurnSignal.RecognitionCompleted(7, 18, "你好"));

            TurnDecision completedWithoutResponse = TurnKernel.Reduce(
                deniedDecision.Next,
                TurnSignal.CancelRequested(7, 19, "turn-completed"));
            Assert.True(completedWithoutResponse.Accepted);
            Assert.Equal(AssistantState.Idle, completedWithoutResponse.Next.ConversationState);
            Assert.Equal(TurnAuthorizationStatus.NotRequired, completedWithoutResponse.Next.Authorization);
            Assert.Collection(
                completedWithoutResponse.Effects,
                effect => Assert.IsType<CompleteTurnEffect>(effect),
                effect => Assert.IsType<PublishStateChangedEffect>(effect));
            Assert.DoesNotContain(completedWithoutResponse.Effects, static effect => effect is PublishUserSpokeEffect or GenerateResponseEffect);
        }

        // A recognition failure of the pending turn returns to Idle with no lingering identity facts.
        TurnDecision failed = TurnKernel.Reduce(recognizing, TurnSignal.RecognitionFailed(7, 20, Failure()));
        Assert.True(failed.Accepted);
        Assert.Equal(AssistantState.Idle, failed.Next.ConversationState);
        Assert.Equal(TurnAuthorizationStatus.NotRequired, failed.Next.Authorization);

        // A pending single-talk turn is not an interruption candidate: nothing here touches playback.
        TurnSnapshot afterOwnerTurn = TurnKernel.Reduce(
            recognizing,
            TurnSignal.AuthorizationResolved(7, 21, TurnAuthorizationStatus.Authorized, 0)).Next;
        Assert.Equal(PhysicalStopTrigger.None, afterOwnerTurn.PhysicalStopTrigger);
        Assert.False(afterOwnerTurn.PhysicalStopIssued);

        // A Thinking-state interruption opens the next Listening turn with a clean identity slate.
        TurnSnapshot thinking = TurnKernel.Reduce(afterOwnerTurn, TurnSignal.RecognitionCompleted(7, 22, "你好")).Next;
        TurnDecision interrupted = TurnKernel.Reduce(thinking, TurnSignal.InterruptionRequested(7, 8));
        Assert.True(interrupted.Accepted);
        Assert.Equal(AssistantState.Listening, interrupted.Next.ConversationState);
        Assert.Equal(TurnAuthorizationStatus.NotRequired, interrupted.Next.Authorization);
    }

    private static KernelFailure Failure()
        => new("VBX-TEST", "test", "kernel", "failure");

    private static void AssertRejected(TurnSnapshot current, TurnSignal signal)
    {
        TurnDecision decision = TurnKernel.Reduce(current, signal);
        Assert.False(decision.Accepted);
        Assert.Same(current, decision.Next);
        Assert.Empty(decision.Effects);
    }
}
