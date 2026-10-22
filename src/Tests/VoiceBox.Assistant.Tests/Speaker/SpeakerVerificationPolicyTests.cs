using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Extensions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerVerificationPolicyTests
{
    [Fact]
    public void Create_RejectsMismatchedFormatAndInvalidThresholds()
    {
        var speaker = new ScriptedSpeakerService();
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        var all = new SpeakerPurposeSet(true, true, false);

        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, audioSampleRate: 8000, audioChannels: 1, all));
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, audioSampleRate: 16000, audioChannels: 2, all));

        options.Confirmation.Threshold = 1.5f;
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, all));

        options.Confirmation.Threshold = 0.5f;
        options.Confirmation.HighConfidenceThreshold = 0.4f;
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, all));

        options.Confirmation.HighConfidenceThreshold = null;
        options.Confirmation.IndeterminateScoreMargin = 3f;
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, all));
    }

    [Fact]
    public void Create_RejectsRealtimePurposeOnProviderWindowLongerThanAllowed()
    {
        // A 2s window provider (WeSpeaker-like) cannot serve Wakeup/BargeInCommand.
        var slow = new ScriptedSpeakerService(windowSamples: 32000, hopSamples: 1600);
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.AudioRetentionMs = 60000;

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(slow, options, 16000, 1, new SpeakerPurposeSet(true, false, false)));
        Assert.Contains("MaxRealtimeWindowMs", error.Message, StringComparison.Ordinal);

        // SingleTalk alone is not realtime and may use it.
        options.Confirmation.Enable = true;
        SpeakerVerificationPolicy policy = SpeakerVerificationPolicy.Create(
            slow, options, 16000, 1, new SpeakerPurposeSet(false, false, true));
        Assert.True(policy.Get(SpeakerVerificationPurpose.SingleTalk).Enabled);
    }

    [Fact]
    public void Create_RejectsInvalidServiceGeometry()
    {
        var invalid = new ScriptedSpeakerService(windowSamples: 0, hopSamples: 1600);
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(invalid, SpeakerTestProfiles.Options("unused"), 16000, 1, new SpeakerPurposeSet(true, true, false)));
    }

    [Fact]
    public void Create_AppliesPurposeThresholdsBanksAndEffectiveEnablement()
    {
        var speaker = new ScriptedSpeakerService();
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.Confirmation.Threshold = 0.6f;

        SpeakerVerificationPolicy policy = SpeakerVerificationPolicy.Create(
            speaker, options, 16000, 1, new SpeakerPurposeSet(Wakeup: true, BargeInCommand: false, SingleTalk: false));
        Assert.Equal(8000, policy.WindowSamples);
        Assert.Equal(1600, policy.HopSamples);
        Assert.Equal(0.6f, policy.GetThreshold(SpeakerVerificationPurpose.BargeInCommand));
        Assert.Equal(0.6f, policy.GetThreshold(SpeakerVerificationPurpose.Wakeup));
        Assert.Equal(0.75f, policy.GetHighConfidenceThreshold(SpeakerVerificationPurpose.Wakeup), 3);
        Assert.True(policy.Get(SpeakerVerificationPurpose.Wakeup).Enabled);
        Assert.False(policy.Get(SpeakerVerificationPurpose.BargeInCommand).Enabled);
        Assert.Equal([SpeakerPurposeBank.CleanNearEnd], policy.RequiredBanks);
        Assert.Equal(SpeakerPurposeBank.DoubleTalkNearEnd, SpeakerVerificationPurpose.BargeInCommand.Bank());
        Assert.Equal(SpeakerPurposeBank.CleanNearEnd, SpeakerVerificationPurpose.SingleTalk.Bank());
    }

    [Fact]
    public void Create_PrefixPathMustBeShorterThanProviderWindow()
    {
        var speaker = new ScriptedSpeakerService();
        VoiceAssistantSpeakerOptions options = SpeakerTestProfiles.Options("unused");
        options.PrefixHighConfidenceEnabled = true;
        options.PrefixHighConfidenceMs = 500;
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, new SpeakerPurposeSet(true, true, false)));

        options.PrefixHighConfidenceMs = 300;
        SpeakerVerificationPolicy policy = SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, new SpeakerPurposeSet(true, true, false));
        Assert.Equal(4800, policy.PrefixHighConfidenceSamples);
        Assert.True(policy.PrefixHighConfidenceEnabled);
    }
}
