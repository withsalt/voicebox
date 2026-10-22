using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class BackgroundResponseKernelTests
{
    [Theory]
    [InlineData(AssistantState.Listening, 0, 8)]
    [InlineData(AssistantState.Recognizing, 0, 8)]
    [InlineData(AssistantState.Thinking, 0, 8)]
    [InlineData(AssistantState.Speaking, 0, 8)]
    [InlineData(AssistantState.Idle, 9, 8)]
    [InlineData(AssistantState.Idle, 0, 7)]
    public void BackgroundResponse_CannotReplaceUserTurnOrPendingValidation(
        AssistantState state, long validationTurnId, long proposedTurnId)
    {
        var current = new TurnSnapshot
        {
            ConversationState = state, TurnId = 7, NextTurnOrdinal = 8, ValidationTurnId = validationTurnId,
        };
        TurnDecision decision = TurnKernel.Reduce(current, TurnSignal.BackgroundResponseStarted(proposedTurnId, 10));

        Assert.False(decision.Accepted);
        Assert.Same(current, decision.Next);
        Assert.Empty(decision.Effects);
    }

    [Fact]
    public void BackgroundResponse_UsesNewOwnerAndOldCompletionCannotFinishNewTurn()
    {
        var idle = new TurnSnapshot { TurnId = 7, NextTurnOrdinal = 8 };
        TurnDecision started = TurnKernel.Reduce(idle, TurnSignal.BackgroundResponseStarted(8, 10));
        Assert.True(started.Accepted);
        Assert.Equal(8, started.Next.TurnId);
        Assert.Equal(9, started.Next.NextTurnOrdinal);
        Assert.DoesNotContain(started.Effects, effect => effect is PrepareRecognitionEffect or GenerateResponseEffect);

        TurnDecision speaking = TurnKernel.Reduce(started.Next, TurnSignal.PlaybackStarted(8, 10));
        Assert.True(speaking.Accepted);
        Assert.Equal(AssistantState.Speaking, speaking.Next.ConversationState);

        TurnDecision interrupted = TurnKernel.Reduce(speaking.Next, TurnSignal.InterruptionRequested(8, 9));
        Assert.True(interrupted.Accepted);
        Assert.Contains(interrupted.Effects, effect => effect is StopPlaybackEffect { TurnId: 8 });
        Assert.Contains(interrupted.Effects, effect => effect is CancelResponseEffect { TurnId: 8 });

        TurnDecision completed = TurnKernel.Reduce(speaking.Next, TurnSignal.PlaybackCompleted(8, 10));
        Assert.True(completed.Accepted);
        Assert.Equal(AssistantState.Idle, completed.Next.ConversationState);
        TurnDecision next = TurnKernel.Reduce(completed.Next, TurnSignal.BackgroundResponseStarted(9, 11));
        Assert.True(next.Accepted);
        Assert.False(TurnKernel.Reduce(next.Next, TurnSignal.PlaybackCompleted(8, 10)).Accepted);
    }
}
