using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Runtime.Contracts;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class BargeInConfirmationPolicyMutationTests
{
    [Fact]
    public void UniformPolicy_AcceptsOnlyThinkingAndSpeakingWithDefaultSettings()
    {
        var settings = new BargeInConfirmationSettings(120, 0.04f, 10, 20, 300, 400);
        BargeInConfirmationPolicy policy = new(settings, settings);

        Assert.True(policy.TryGetSettings(AssistantState.Thinking, out BargeInConfirmationSettings thinking));
        Assert.True(policy.TryGetSettings(AssistantState.Speaking, out BargeInConfirmationSettings speaking));
        Assert.Equal(settings, thinking);
        Assert.Equal(settings, speaking);
        Assert.False(policy.TryGetSettings(AssistantState.Idle, out BargeInConfirmationSettings idle));
        Assert.Equal(default, idle);
        Assert.False(policy.TryGetSettings(AssistantState.Listening, out _));
        Assert.False(policy.TryGetSettings(AssistantState.Recognizing, out _));
        Assert.False(policy.TryGetSettings(AssistantState.Stopped, out _));
    }

    [Fact]
    public void SpeakingPolicy_UsesTheMaximumOfEveryAdaptiveStartupSuppression()
    {
        var thinking = new BargeInConfirmationSettings(100, 0.02f, 5, 7, 200, 300);
        var speaking = new BargeInConfirmationSettings(150, 0.05f, 20, 60, 500, 700);
        var policy = new BargeInConfirmationPolicy(thinking, speaking, new FixedStartupGuard(80));

        Assert.True(policy.TryGetSettings(AssistantState.Thinking, out BargeInConfirmationSettings actualThinking));
        Assert.Equal(thinking, actualThinking);
        Assert.True(policy.TryGetSettings(AssistantState.Speaking, out BargeInConfirmationSettings actualSpeaking));
        Assert.Equal(150, actualSpeaking.MinDurationMs);
        Assert.Equal(0.05f, actualSpeaking.MinRms);
        Assert.Equal(80, actualSpeaking.InitialSuppressionMs);
        Assert.Equal(80, actualSpeaking.RecognitionSuppressionMs);
        Assert.Equal(500, actualSpeaking.MaxCandidateMs);
        Assert.Equal(700, actualSpeaking.RecognitionAudioLimitMs);
    }

    [Fact]
    public void SpeakingPolicy_PreservesEachIndependentSuppressionWhenOnlyOneNeedsAdaptation()
    {
        var speaking = new BargeInConfirmationSettings(100, 0.02f, 90, 30, 400, 500);
        var policy = new BargeInConfirmationPolicy(speaking, speaking, new FixedStartupGuard(60));

        Assert.True(policy.TryGetSettings(AssistantState.Speaking, out BargeInConfirmationSettings actual));
        Assert.Equal(90, actual.InitialSuppressionMs);
        Assert.Equal(60, actual.RecognitionSuppressionMs);
        Assert.Equal(speaking.MinDurationMs, actual.MinDurationMs);
        Assert.Equal(speaking.MinRms, actual.MinRms);
        Assert.Equal(speaking.MaxCandidateMs, actual.MaxCandidateMs);
        Assert.Equal(speaking.RecognitionAudioLimitMs, actual.RecognitionAudioLimitMs);
    }

    [Fact]
    public void SpeakingPolicy_AdaptsInitialSuppressionWithoutChangingRecognitionSuppression()
    {
        var speaking = new BargeInConfirmationSettings(100, 0.02f, 30, 90, 400, 500);
        var policy = new BargeInConfirmationPolicy(speaking, speaking, new FixedStartupGuard(60));

        Assert.True(policy.TryGetSettings(AssistantState.Speaking, out BargeInConfirmationSettings actual));
        Assert.Equal(60, actual.InitialSuppressionMs);
        Assert.Equal(90, actual.RecognitionSuppressionMs);
        Assert.Equal(speaking.MinDurationMs, actual.MinDurationMs);
        Assert.Equal(speaking.MinRms, actual.MinRms);
        Assert.Equal(speaking.MaxCandidateMs, actual.MaxCandidateMs);
        Assert.Equal(speaking.RecognitionAudioLimitMs, actual.RecognitionAudioLimitMs);
    }

    private sealed class FixedStartupGuard : IPlaybackStartupGuardView
    {
        public FixedStartupGuard(int remainingMilliseconds)
        {
            RemainingMilliseconds = remainingMilliseconds;
        }

        public bool IsActive => RemainingMilliseconds > 0;
        public int RemainingMilliseconds { get; }
        public bool IsActiveAt(long timestampTicks) => IsActive;
    }
}
