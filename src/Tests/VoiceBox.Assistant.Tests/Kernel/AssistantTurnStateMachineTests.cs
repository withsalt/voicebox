using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AssistantTurnObservationStoreTests
{
    [Fact]
    public void RuntimeProjection_ChangesOnlyAfterKernelSynchronization()
    {
        var turns = new AssistantTurnObservationStoreHarness(
            new BargeInDetector(0, 0, 16000, 1));

        SpeechStartedDecision decision = turns.ObserveSpeechStartedWithoutCommit(
            enableBargeIn: false,
            proposedTurnId: 1);

        Assert.Equal(SpeechStartedDecisionKind.StartListening, decision.Kind);
        Assert.Equal(1, decision.TurnId);
        Assert.Equal(AssistantState.Idle, turns.State);
        Assert.False(turns.IsCurrentTurn(1));

        turns.SynchronizeKernelState(AssistantState.Listening, decision.TurnId);

        Assert.Equal(AssistantState.Listening, turns.State);
        Assert.True(turns.IsCurrentTurn(1));
    }

    [Fact]
    public void OnSpeechStarted_FromIdle_StartsListeningTurn()
    {
        var turns = CreateTurns();

        SpeechStartedDecision decision = turns.OnSpeechStarted(enableBargeIn: false);

        Assert.Equal(SpeechStartedDecisionKind.StartListening, decision.Kind);
        Assert.Equal(AssistantState.Idle, decision.OldState);
        Assert.Equal(AssistantState.Listening, turns.State);
        Assert.Equal(1, decision.TurnId);
    }

    [Fact]
    public void SpeechEndBeforeAsrReady_IsRememberedAndReturnedOnAttach()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        turns.OnSpeaking([0.1f, 0.2f]);
        SpeechEndedDecision ended = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(120));
        var fed = new List<float[]>();

        bool attached = turns.TryAttachStartedAsrSession(
            turnId,
            "session-1",
            audio => fed.Add(audio),
            out bool speechAlreadyEnded,
            out TimeSpan duration);

        Assert.Equal(SpeechEndedDecisionKind.WaitingForAsrStart, ended.Kind);
        Assert.True(attached);
        Assert.True(speechAlreadyEnded);
        Assert.Equal(TimeSpan.FromMilliseconds(120), duration);
        Assert.Single(fed);
        Assert.Equal([0.1f, 0.2f], fed[0]);
    }

    [Fact]
    public void TryAttachStartedAsrSession_DrainsAudioArrivingDuringPendingFlushBeforePublishingSession()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        turns.OnSpeaking([0.1f]);
        var fed = new List<float[]>();
        bool injected = false;

        bool attached = turns.TryAttachStartedAsrSession(
            turnId,
            "session-1",
            audio =>
            {
                fed.Add(audio);
                if (!injected)
                {
                    injected = true;
                    SpeakingDecision decision = turns.OnSpeaking([0.2f]);
                    Assert.Equal(SpeakingDecisionKind.None, decision.Kind);
                }
            },
            out _,
            out _);

        Assert.True(attached);
        Assert.Equal(2, fed.Count);
        Assert.Equal([0.1f], fed[0]);
        Assert.Equal([0.2f], fed[1]);

        SpeakingDecision afterAttach = turns.OnSpeaking([0.3f]);
        Assert.Equal(SpeakingDecisionKind.FeedAsr, afterAttach.Kind);
        Assert.Equal("session-1", afterAttach.AsrSessionId);
    }

    [Fact]
    public void NearEndWindowConfirmation_FromSpeaking_PreservesAudioForAsrAfterSessionAttach()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision speaking = ConfirmSpeakingBargeIn(turns, [0.4f, 0.5f]);
        var fed = new List<float[]>();

        bool attached = turns.TryAttachStartedAsrSession(
            speaking.TurnId,
            "session-2",
            audio => fed.Add(audio),
            out bool speechAlreadyEnded,
            out _);

        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, speaking.Kind);
        Assert.Equal(firstTurnId, speaking.InterruptedTurnId);
        Assert.Equal(AssistantState.Speaking, turns.State);
        Assert.True(turns.IsCurrentTurn(firstTurnId));
        Assert.True(turns.IsValidatingBargeInTurn(speaking.TurnId));
        Assert.Equal(2, speaking.ConfirmedCandidateSamples);
        Assert.Equal(Math.Sqrt((0.4 * 0.4 + 0.5 * 0.5) / 2), speaking.ConfirmedCandidateRms, 3);
        Assert.Equal(2, speaking.ConfirmedConfirmationSamples);
        Assert.Equal(speaking.ConfirmedCandidateRms, speaking.ConfirmedConfirmationRms);
        Assert.Equal(0, speaking.ConfirmationMinDurationMs);
        Assert.Equal(0, speaking.ConfirmationMinRms);
        Assert.Equal(0, speaking.ConfirmationInitialSuppressionMs);
        Assert.True(attached);
        Assert.False(speechAlreadyEnded);
        Assert.Single(fed);
        Assert.Equal([0.4f, 0.5f], fed[0]);
    }

    [Fact]
    public void BargeInCandidate_SpeakerPendingDoesNotCommitAndMismatchKeepsCurrentResponse()
    {
        var turns = CreateTurns();
        long activeTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(activeTurnId, AssistantState.Thinking, out _, out _));
        SpeechStartedDecision started = turns.OnSpeechStarted(enableBargeIn: true);

        SpeakingDecision pending = turns.OnSpeaking(
            [0.4f],
            BargeInSpeakerGateStatus.Pending);
        SpeakingDecision rejected = turns.OnSpeaking(
            [0.5f],
            BargeInSpeakerGateStatus.Rejected);

        Assert.Equal(SpeechStartedDecisionKind.BargeInCandidate, started.Kind);
        Assert.Equal(SpeakingDecisionKind.None, pending.Kind);
        Assert.Equal(SpeakingDecisionKind.RejectedBySpeaker, rejected.Kind);
        Assert.Equal(activeTurnId, rejected.TurnId);
        Assert.Equal(AssistantState.Thinking, turns.State);
        Assert.True(turns.IsCurrentTurn(activeTurnId));
        Assert.False(turns.HasPendingBargeInCandidateOrValidation);
    }

    [Fact]
    public void BargeInCandidate_FromThinking_SpeakerPendingRetainsAllAudioUntilAsyncMatch()
    {
        var turns = CreateTurns();
        long activeTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(activeTurnId, AssistantState.Thinking, out _, out _));
        Assert.Equal(
            SpeechStartedDecisionKind.BargeInCandidate,
            turns.OnSpeechStarted(enableBargeIn: true).Kind);

        Assert.Equal(
            SpeakingDecisionKind.None,
            turns.OnSpeaking([0.4f], BargeInSpeakerGateStatus.Pending).Kind);
        Assert.Equal(
            SpeakingDecisionKind.None,
            turns.OnSpeaking([0.5f], BargeInSpeakerGateStatus.Pending).Kind);
        Assert.True(turns.IsBargeInAwaitingSpeaker);

        SpeakingDecision matched = turns.ResolveBargeInSpeakerGate(BargeInSpeakerGateStatus.Match);
        var fed = new List<float[]>();
        bool attached = turns.TryAttachStartedAsrSession(
            matched.TurnId,
            "session-2",
            audio => fed.Add(audio),
            out bool speechAlreadyEnded,
            out _);

        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, matched.Kind);
        Assert.Equal(2, matched.ConfirmedCandidateSamples);
        Assert.True(attached);
        Assert.False(speechAlreadyEnded);
        Assert.Equal([0.4f, 0.5f], fed.SelectMany(static chunk => chunk));
    }

    [Fact]
    public void WakeupInterrupt_FromSpeakingWithValidatedBargeIn_ReturnsToIdle()
    {
        var turns = CreateTurns();
        using var responseCts = new CancellationTokenSource();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TrySetResponseCtsForTurn(firstTurnId, responseCts));
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f]);
        bool interrupted = turns.TryInterruptForWakeup(out WakeupInterruptDecision decision);

        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, confirmed.Kind);
        Assert.True(interrupted);
        Assert.Equal(AssistantState.Speaking, decision.OldState);
        Assert.Equal(firstTurnId, decision.InterruptedTurnId);
        Assert.Same(responseCts, decision.ResponseToCancel);
        Assert.Null(decision.AsrSessionIdToCleanup);
        Assert.False(turns.IsCurrentTurn(decision.TurnId));
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void WakeupInterrupt_FromValidatingBargeIn_ClearsValidationAndReturnsAsrSessionToCleanup()
    {
        var turns = CreateTurns();
        using var responseCts = new CancellationTokenSource();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TrySetResponseCtsForTurn(firstTurnId, responseCts));
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f, 0.5f]);
        Assert.True(turns.TryAttachStartedAsrSession(confirmed.TurnId, "validation-session", _ => { }, out _, out _));

        bool interrupted = turns.TryInterruptForWakeup(out WakeupInterruptDecision decision);

        Assert.True(interrupted);
        Assert.Equal(AssistantState.Speaking, decision.OldState);
        Assert.Equal(firstTurnId, decision.InterruptedTurnId);
        Assert.Same(responseCts, decision.ResponseToCancel);
        Assert.Equal("validation-session", decision.AsrSessionIdToCleanup);
        Assert.False(turns.IsValidatingBargeInTurn(confirmed.TurnId));
        Assert.False(turns.IsCurrentTurn(decision.TurnId));
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void ValidatedBargeIn_FromSpeaking_AcceptSwitchesToListeningTurn()
    {
        var turns = CreateTurns();
        using var responseCts = new CancellationTokenSource();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TrySetResponseCtsForTurn(firstTurnId, responseCts));
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f]);
        using var acceptedResponseCts = new CancellationTokenSource();

        bool accepted = turns.TryAcceptValidatedBargeIn(
            confirmed.TurnId,
            acceptedResponseCts,
            clearValidationContext: true,
            out AssistantState oldState,
            out CancellationTokenSource? responseToCancel,
            out long interruptedTurnId);

        Assert.True(accepted);
        Assert.Equal(AssistantState.Speaking, oldState);
        Assert.Same(responseCts, responseToCancel);
        Assert.Equal(firstTurnId, interruptedTurnId);
        Assert.True(turns.IsCurrentTurn(confirmed.TurnId));
        Assert.Same(acceptedResponseCts, turns.GetResponseCtsForTurn(confirmed.TurnId));
        Assert.Equal(AssistantState.Listening, turns.State);
    }

    [Fact]
    public void ValidatedBargeIn_FromSpeaking_RejectRestoresPreviousState()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f]);

        bool rejected = turns.TryRejectValidatedBargeIn(
            confirmed.TurnId,
            out AssistantState oldState,
            out AssistantState restoredState);

        Assert.True(rejected);
        Assert.Equal(AssistantState.Speaking, oldState);
        Assert.Equal(AssistantState.Speaking, restoredState);
        Assert.True(turns.IsCurrentTurn(firstTurnId));
        Assert.False(turns.IsValidatingBargeInTurn(confirmed.TurnId));
        Assert.Equal(AssistantState.Speaking, turns.State);
    }

    [Fact]
    public void ValidatedBargeIn_WhenInterruptedTurnCompletesBeforeReject_RestoresIdle()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f]);
        Assert.True(turns.TryCompleteTurn(firstTurnId, out _, out _, out bool completedInterruptedTurn));
        Assert.True(completedInterruptedTurn);

        bool rejected = turns.TryRejectValidatedBargeIn(
            confirmed.TurnId,
            out _,
            out AssistantState restoredState);

        Assert.True(rejected);
        Assert.Equal(AssistantState.Idle, restoredState);
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void ValidatingBargeIn_WhenInterruptedTurnCompletesBeforeAsrAttach_PreservesCandidateAudio()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f, 0.5f]);
        Assert.True(turns.TryCompleteTurn(firstTurnId, out _, out _, out _));

        var fed = new List<float[]>();
        bool attached = turns.TryAttachStartedAsrSession(
            confirmed.TurnId,
            "session-2",
            audio => fed.Add(audio),
            out _,
            out _);

        Assert.True(attached);
        Assert.Single(fed);
        Assert.Equal([0.4f, 0.5f], fed[0]);
        Assert.True(turns.IsValidatingBargeInTurn(confirmed.TurnId));
    }

    [Fact]
    public void SpeakingVadEvents_AreIgnoredUntilDtdAuthorizesBargeIn()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeechStartedDecision started = turns.OnSpeechStarted(enableBargeIn: true);
        SpeakingDecision vadAudio = turns.OnSpeaking([0.05f, 0.05f]);
        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(turns, [0.4f, 0.5f]);

        Assert.Equal(SpeechStartedDecisionKind.Ignored, started.Kind);
        Assert.Equal(SpeakingDecisionKind.None, vadAudio.Kind);
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, confirmed.Kind);
        Assert.Equal(firstTurnId, confirmed.InterruptedTurnId);
        Assert.Equal(AssistantState.Speaking, turns.State);
    }

    [Fact]
    public void BargeInCandidate_FromThinking_UsesOriginalConfirmationPolicy()
    {
        var turns = CreateTurns(
            thinking: new BargeInConfirmationSettings(minDurationMs: 0, minRms: 0),
            speaking: new BargeInConfirmationSettings(minDurationMs: 100, minRms: 0.08f, initialSuppressionMs: 50),
            sampleRate: 1000);
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Thinking, out _, out _));

        SpeechStartedDecision started = turns.OnSpeechStarted(enableBargeIn: true);
        SpeakingDecision speaking = turns.OnSpeaking([0.01f]);

        Assert.Equal(SpeechStartedDecisionKind.BargeInCandidate, started.Kind);
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, speaking.Kind);
        Assert.Equal(firstTurnId, speaking.InterruptedTurnId);
        Assert.Equal(AssistantState.Listening, turns.State);
    }

    [Fact]
    public void ListeningGate_FromIdle_ReturnsCandidateWithoutStateChange()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 200, minRms: 0.02f, sampleRate: 1000);

        SpeechStartedDecision decision = turns.OnSpeechStarted(enableBargeIn: false);

        Assert.Equal(SpeechStartedDecisionKind.ListeningCandidate, decision.Kind);
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void ListeningGate_LowRmsCandidate_DiscardedOnSpeechEndedAndStaysIdle()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 200, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        // 500ms @1kHz, rms=0.014 — 模拟 AEC 回声残差: 时长够但能量低于门槛。
        SpeakingDecision speaking = turns.OnSpeaking(ConstantSamples(0.014f, 500));
        SpeechEndedDecision ended = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(500));

        Assert.Equal(SpeakingDecisionKind.None, speaking.Kind);
        Assert.Equal(SpeechEndedDecisionKind.DiscardedListeningCandidate, ended.Kind);
        Assert.Equal(AssistantState.Idle, turns.State);
        Assert.Equal(500, ended.DiscardedCandidateSamples);
        Assert.Equal(0.014, ended.DiscardedCandidateRms, 3);

        // 候选被丢弃后不留残余: 下一次语音重新从候选开始。
        SpeechStartedDecision next = turns.OnSpeechStarted(enableBargeIn: false);
        Assert.Equal(SpeechStartedDecisionKind.ListeningCandidate, next.Kind);
    }

    [Fact]
    public void ListeningGate_ShortCandidate_DiscardedEvenWhenLoud()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 200, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        // 100ms @1kHz, 响但太短 (如键盘敲击) — 不应确认。
        SpeakingDecision speaking = turns.OnSpeaking(ConstantSamples(0.3f, 100));
        SpeechEndedDecision ended = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(100));

        Assert.Equal(SpeakingDecisionKind.None, speaking.Kind);
        Assert.Equal(SpeechEndedDecisionKind.DiscardedListeningCandidate, ended.Kind);
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void ListeningGate_ConfirmedCandidate_StartsTurnAndPreservesAudioForAsr()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 200, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        SpeakingDecision first = turns.OnSpeaking(ConstantSamples(0.1f, 150));   // 未达 200ms
        SpeakingDecision second = turns.OnSpeaking(ConstantSamples(0.1f, 150));  // 累计 300ms → 确认

        Assert.Equal(SpeakingDecisionKind.None, first.Kind);
        Assert.Equal(SpeakingDecisionKind.ConfirmedListening, second.Kind);
        Assert.Equal(AssistantState.Idle, second.OldState);
        Assert.Equal(AssistantState.Listening, turns.State);

        // 候选期间的全部音频保留, ASR session 就绪后补喂。
        var fed = new List<float[]>();
        bool attached = turns.TryAttachStartedAsrSession(
            second.TurnId,
            "session-1",
            audio => fed.Add(audio),
            out bool speechAlreadyEnded,
            out _);

        Assert.True(attached);
        Assert.False(speechAlreadyEnded);
        Assert.Equal(2, fed.Count);
        Assert.Equal(300, fed.Sum(c => c.Length));
    }

    [Fact]
    public void ListeningGate_AudioArrivingAfterConfirmation_IsFedToAsrInChronologicalOrder()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 100, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        SpeakingDecision confirmed = turns.OnSpeaking(ConstantSamples(0.1f, 200));  // 候选期音频 (较早)
        Assert.Equal(SpeakingDecisionKind.ConfirmedListening, confirmed.Kind);

        // 确认后、ASR session 就绪前到达的音频走 _pendingAsrAudio (较晚)。
        SpeakingDecision later = turns.OnSpeaking(ConstantSamples(0.2f, 50));
        Assert.Equal(SpeakingDecisionKind.None, later.Kind);

        var fed = new List<float[]>();
        bool attached = turns.TryAttachStartedAsrSession(
            confirmed.TurnId,
            "session-1",
            audio => fed.Add(audio),
            out _,
            out _);

        Assert.True(attached);
        Assert.Equal(2, fed.Count);
        Assert.Equal(0.1f, fed[0][0]); // 候选期音频必须先喂
        Assert.Equal(0.2f, fed[1][0]);
    }

    [Fact]
    public void ListeningGate_ConfirmedThenSpeechEnded_FollowsNormalTurnFlow()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 100, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        SpeakingDecision confirmed = turns.OnSpeaking(ConstantSamples(0.1f, 200));
        SpeechEndedDecision ended = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(200));

        Assert.Equal(SpeakingDecisionKind.ConfirmedListening, confirmed.Kind);
        // ASR session 未就绪时语音先结束 → 记下待办, attach 时直接进入识别。
        Assert.Equal(SpeechEndedDecisionKind.WaitingForAsrStart, ended.Kind);

        bool attached = turns.TryAttachStartedAsrSession(
            confirmed.TurnId,
            "session-1",
            _ => { },
            out bool speechAlreadyEnded,
            out TimeSpan duration);

        Assert.True(attached);
        Assert.True(speechAlreadyEnded);
        Assert.Equal(TimeSpan.FromMilliseconds(200), duration);
    }

    [Fact]
    public void ListeningGate_Disabled_KeepsImmediateStartBehavior()
    {
        var turns = CreateTurns(); // 无门槛

        SpeechStartedDecision decision = turns.OnSpeechStarted(enableBargeIn: false);

        Assert.Equal(SpeechStartedDecisionKind.StartListening, decision.Kind);
        Assert.Equal(AssistantState.Listening, turns.State);
    }

    [Fact]
    public void ListeningGate_CandidateWhenStateLeavesIdle_IsDroppedOnNextSpeaking()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 100, minRms: 0.02f, sampleRate: 1000);

        turns.OnSpeechStarted(enableBargeIn: false);
        Assert.False(turns.IsIdleWithoutCandidate);

        // 候选累积期间状态被外部转走 (如 Stop) → 候选作废, 不会误确认。
        Assert.True(turns.TransitionTo(AssistantState.Stopped, out _));
        SpeakingDecision speaking = turns.OnSpeaking(ConstantSamples(0.3f, 500));

        Assert.Equal(SpeakingDecisionKind.None, speaking.Kind);
        Assert.Equal(AssistantState.Stopped, turns.State);

        // 候选已被清掉: SpeechEnded 不再报 Discarded。
        SpeechEndedDecision ended = turns.OnSpeechEnded(TimeSpan.FromMilliseconds(500));
        Assert.Equal(SpeechEndedDecisionKind.None, ended.Kind);
    }

    [Fact]
    public void IsIdleWithoutCandidate_ReflectsCandidateAndTurnLifecycle()
    {
        var turns = CreateTurnsWithListeningGate(minDurationMs: 100, minRms: 0.02f, sampleRate: 1000);
        Assert.True(turns.IsIdleWithoutCandidate);

        turns.OnSpeechStarted(enableBargeIn: false);
        Assert.False(turns.IsIdleWithoutCandidate); // 候选累积中

        turns.OnSpeechEnded(TimeSpan.FromMilliseconds(50)); // 未确认 → 丢弃
        Assert.True(turns.IsIdleWithoutCandidate);

        turns.OnSpeechStarted(enableBargeIn: false);
        SpeakingDecision confirmed = turns.OnSpeaking(ConstantSamples(0.1f, 200));
        Assert.Equal(SpeakingDecisionKind.ConfirmedListening, confirmed.Kind);
        Assert.False(turns.IsIdleWithoutCandidate); // Listening 回合进行中

        Assert.True(turns.TryCompleteTurn(confirmed.TurnId, out _, out _, out _));
        Assert.True(turns.IsIdleWithoutCandidate);
    }

    [Fact]
    public void NearEndWindowConfirmation_AudioArrivingAfterAuthorization_IsFedToAsrInChronologicalOrder()
    {
        var turns = CreateTurns();
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        SpeakingDecision confirmed = ConfirmSpeakingBargeIn(
            turns,
            ConstantSamples(0.1f, 200));
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, confirmed.Kind);

        SpeakingDecision later = turns.OnSpeaking(ConstantSamples(0.2f, 50));
        Assert.Equal(SpeakingDecisionKind.None, later.Kind);

        var fed = new List<float[]>();
        Assert.True(turns.TryAttachStartedAsrSession(confirmed.TurnId, "session-2", audio => fed.Add(audio), out _, out _));

        Assert.Equal(2, fed.Count);
        Assert.Equal(0.1f, fed[0][0]); // 候选期音频必须先喂
        Assert.Equal(0.2f, fed[1][0]);
    }

    [Fact]
    public void NearEndWindowConfirmation_FromSpeaking_StartsValidationTurnAndPreservesPreRollAudio()
    {
        var turns = CreateTurns(
            thinking: new BargeInConfirmationSettings(minDurationMs: 0, minRms: 0),
            speaking: new BargeInConfirmationSettings(minDurationMs: 300, minRms: 0.03f),
            sampleRate: 1000);
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        bool confirmed = turns.TryConfirmBargeInFromNearEndWindow(
            [ConstantSamples(0.1f, 120), ConstantSamples(0.2f, 80)],
            ConfirmedNearEndWindow(),
            out SpeakingDecision decision);

        Assert.True(confirmed);
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, decision.Kind);
        Assert.Equal(firstTurnId, decision.InterruptedTurnId);
        Assert.Equal(AssistantState.Speaking, decision.OldState);
        Assert.Equal(AssistantState.Speaking, turns.State);
        Assert.True(turns.IsCurrentTurn(firstTurnId));
        Assert.True(turns.IsValidatingBargeInTurn(decision.TurnId));
        Assert.Equal(200, decision.ConfirmedCandidateSamples);
        Assert.Equal(300, decision.ConfirmationMinDurationMs);
        Assert.Equal(0.03f, decision.ConfirmationMinRms);
        Assert.Equal(15, decision.ConfirmedBargeInDiagnostics.FrameCount);

        var fed = new List<float[]>();
        Assert.True(turns.TryAttachStartedAsrSession(decision.TurnId, "session-2", audio => fed.Add(audio), out _, out _));
        Assert.Equal(2, fed.Count);
        Assert.Equal(0.1f, fed[0][0]);
        Assert.Equal(0.2f, fed[1][0]);
    }

    [Fact]
    public void NearEndWindowConfirmation_RejectsContinueDecision()
    {
        var turns = CreateTurns(
            thinking: new BargeInConfirmationSettings(minDurationMs: 0, minRms: 0),
            speaking: new BargeInConfirmationSettings(minDurationMs: 300, minRms: 0.03f),
            sampleRate: 1000);
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));

        bool confirmed = turns.TryConfirmBargeInFromNearEndWindow(
            [ConstantSamples(0.1f, 200)],
            ConfirmedNearEndWindow() with { Action = BargeInAction.Continue },
            out SpeakingDecision decision);

        Assert.False(confirmed);
        Assert.Equal(SpeakingDecisionKind.None, decision.Kind);
        Assert.Equal(AssistantState.Speaking, turns.State);
    }

    [Fact]
    public void NearEndWindowConfirmation_RemainsAvailableAfterSpeakingVadEvent()
    {
        var turns = CreateTurns(
            thinking: new BargeInConfirmationSettings(minDurationMs: 0, minRms: 0),
            speaking: new BargeInConfirmationSettings(minDurationMs: 300, minRms: 0.03f),
            sampleRate: 1000);
        long firstTurnId = turns.OnSpeechStarted(enableBargeIn: true).TurnId;
        Assert.True(turns.TryTransitionForTurn(firstTurnId, AssistantState.Speaking, out _, out _));
        SpeechStartedDecision vadStarted = turns.OnSpeechStarted(enableBargeIn: true);

        bool confirmed = turns.TryConfirmBargeInFromNearEndWindow(
            [ConstantSamples(0.1f, 200)],
            ConfirmedNearEndWindow(),
            out SpeakingDecision decision);

        Assert.Equal(SpeechStartedDecisionKind.Ignored, vadStarted.Kind);
        Assert.True(confirmed);
        Assert.Equal(SpeakingDecisionKind.ConfirmedBargeIn, decision.Kind);
    }

    [Fact]
    public void TryEnterRecognizing_RequiresCurrentListeningTurnAndSession()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryAttachStartedAsrSession(turnId, "session-1", _ => { }, out _, out _));

        Assert.False(turns.TryEnterRecognizing(turnId + 1, "session-1", out _, out _));
        Assert.False(turns.TryEnterRecognizing(turnId, "wrong-session", out _, out _));
        Assert.True(turns.TryEnterRecognizing(turnId, "session-1", out AssistantState oldState, out bool isValidationTurn));

        Assert.Equal(AssistantState.Listening, oldState);
        Assert.False(isValidationTurn);
        Assert.Equal(AssistantState.Recognizing, turns.State);
    }

    [Fact]
    public void TryCompleteTurn_ClearsCurrentTurnBackToIdle()
    {
        var turns = CreateTurns();
        long turnId = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        turns.TryTransitionForTurn(turnId, AssistantState.Thinking, out _, out _);

        bool completed = turns.TryCompleteTurn(
            turnId,
            out AssistantState oldState,
            out bool changed,
            out bool completedInterruptedTurn);

        Assert.True(completed);
        Assert.True(changed);
        Assert.False(completedInterruptedTurn);
        Assert.Equal(AssistantState.Thinking, oldState);
        Assert.Equal(AssistantState.Idle, turns.State);
    }

    [Fact]
    public void DuplicateAndLateTurnCompletion_CannotCompleteTheCurrentSuccessor()
    {
        var turns = CreateTurns();
        long completedTurn = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.True(turns.TryTransitionForTurn(completedTurn, AssistantState.Thinking, out _, out _));
        Assert.True(turns.TryCompleteTurn(completedTurn, out _, out _, out _));

        Assert.True(turns.TryCompleteTurn(completedTurn, out AssistantState duplicateOldState, out bool duplicateChanged, out _));
        Assert.Equal(AssistantState.Idle, duplicateOldState);
        Assert.False(duplicateChanged);
        long successor = turns.OnSpeechStarted(enableBargeIn: false).TurnId;
        Assert.False(turns.TryCompleteTurn(completedTurn, out _, out _, out _));

        Assert.Equal(completedTurn + 1, successor);
        Assert.Equal(AssistantState.Listening, turns.State);
    }

    private static AssistantTurnObservationStoreHarness CreateTurns()
    {
        var bargeIn = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 16000, channels: 1);
        return new AssistantTurnObservationStoreHarness(bargeIn);
    }

    private static AssistantTurnObservationStoreHarness CreateTurns(
        BargeInConfirmationSettings thinking,
        BargeInConfirmationSettings speaking,
        int sampleRate)
    {
        var bargeIn = new BargeInDetector(
            minDurationMs: thinking.MinDurationMs,
            minRms: thinking.MinRms,
            sampleRate: sampleRate,
            channels: 1);
        var policy = new BargeInConfirmationPolicy(thinking, speaking);
        return new AssistantTurnObservationStoreHarness(bargeIn, policy);
    }

    private static AssistantTurnObservationStoreHarness CreateTurnsWithListeningGate(
        int minDurationMs,
        float minRms,
        int sampleRate)
    {
        var bargeIn = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: sampleRate, channels: 1);
        return new AssistantTurnObservationStoreHarness(
            bargeIn,
            listeningGate: new BargeInConfirmationSettings(minDurationMs, minRms));
    }

    private static float[] ConstantSamples(float value, int count)
    {
        var samples = new float[count];
        Array.Fill(samples, value);
        return samples;
    }

    private static SpeakingDecision ConfirmSpeakingBargeIn(
        AssistantTurnObservationStoreHarness turns,
        params float[][] initialAudio)
    {
        Assert.True(turns.TryConfirmBargeInFromNearEndWindow(
            initialAudio,
            ConfirmedNearEndWindow(),
            out SpeakingDecision decision));
        return decision;
    }

    private static float[] Flatten(IReadOnlyList<float[]> chunks)
    {
        int length = 0;
        foreach (float[] chunk in chunks)
            length += chunk.Length;

        var result = new float[length];
        int offset = 0;
        foreach (float[] chunk in chunks)
        {
            Array.Copy(chunk, 0, result, offset, chunk.Length);
            offset += chunk.Length;
        }

        return result;
    }

    private static BargeInDecision ConfirmedNearEndWindow()
        => new(
            Action: BargeInAction.Interrupt,
            FrameCount: 15,
            AcceptedFrameCount: 15,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: 300,
            AcceptedDurationMs: 300,
            RequiredDurationMs: 300,
            ScoreDurationMs: 300,
            PeakScoreDurationMs: 300,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.055,
            MinRequiredRms: 0.055,
            MaxRequiredRms: 0.055,
            Reason: "confirmed-nearend-window");

    private static DuplexAudioMetrics Metrics(
        double renderRms,
        double captureOutputRms,
        bool captureProcessed,
        double? correlation = null,
        double? inputCorrelation = null,
        double? outputCorrelation = null)
    {
        double corr = correlation ?? (renderRms > 0 ? 0.8 : 0);
        double inputCorr = inputCorrelation ?? corr;
        double outputCorr = outputCorrelation ?? corr;
        return new DuplexAudioMetrics(
            TimestampTicks: 0,
            SampleRate: 1000,
            Channels: 1,
            FrameCount: 20,
            RenderRms: renderRms,
            RenderPeak: renderRms * 2,
            CaptureInputRms: captureOutputRms,
            CaptureInputPeak: captureOutputRms * 2,
            CaptureOutputRms: captureOutputRms,
            CaptureOutputPeak: captureOutputRms * 2,
            RenderCaptureCorrelation: outputCorr,
            CaptureProcessed: captureProcessed,
            RenderCaptureInputCorrelation: inputCorr,
            RenderCaptureOutputCorrelation: outputCorr);
    }

    private static AcousticProcessingMetrics Metrics(double erle)
        => new(
            HasEchoReturnLoss: false,
            EchoReturnLoss: 0,
            HasEchoReturnLossEnhancement: true,
            EchoReturnLossEnhancement: erle,
            HasDelayMs: false,
            DelayMs: 0,
            HasDelayMedianMs: false,
            DelayMedianMs: 0,
            HasResidualEchoLikelihood: false,
            ResidualEchoLikelihood: 0);

    private static VadProbabilityEventArgs VadProbability(
        float probability = 0.80f,
        float smoothedProbability = 0.75f,
        bool isSpeech = true)
        => new(
            probability,
            smoothedProbability,
            isSpeech,
            frameIndex: 1,
            timestamp: TimeSpan.Zero,
            frameDuration: TimeSpan.FromMilliseconds(20));
}
