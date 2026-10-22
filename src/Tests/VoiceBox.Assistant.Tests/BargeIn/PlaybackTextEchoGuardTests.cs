using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class PlaybackTextEchoGuardTests
{
    [Fact]
    public void ExactTtsSubstring_WithDifferentPunctuation_IsEcho()
    {
        var guard = new PlaybackTextEchoGuard();
        guard.BeginResponse(7);
        guard.AppendScheduledSpeech(
            7,
            "没问题，我给你讲个长一点的。在一座被山围着的小镇上，有个叫阿明的少年。");

        PlaybackTextEchoMatch match = guard.Match("在一座被山围着的小镇上。");

        Assert.True(match.IsMatch);
        Assert.Equal("exact-substring", match.Reason);
    }

    [Fact]
    public void OneCharacterAsrDifference_InLongTtsExcerpt_IsEcho()
    {
        var guard = new PlaybackTextEchoGuard();
        guard.BeginResponse(7);
        guard.AppendScheduledSpeech(7, "在一座被山围着的小镇上，有个叫阿明的少年。");

        PlaybackTextEchoMatch match = guard.Match("在一座被山围这的小镇上");

        Assert.True(match.IsMatch);
        Assert.Equal("bigram-coverage", match.Reason);
    }

    [Theory]
    [InlineData("不好听，换一个")]
    [InlineData("停一下")]
    [InlineData("阿明后来怎么样了")]
    public void DifferentUserCommand_IsNotEcho(string text)
    {
        var guard = new PlaybackTextEchoGuard();
        guard.BeginResponse(7);
        guard.AppendScheduledSpeech(
            7,
            "没问题，我给你讲个长一点的。在一座被山围着的小镇上，有个叫阿明的少年。");

        Assert.False(guard.Match(text).IsMatch);
    }

    [Fact]
    public void NewResponse_DropsPreviousResponseText()
    {
        var guard = new PlaybackTextEchoGuard();
        guard.BeginResponse(7);
        guard.AppendScheduledSpeech(7, "在一座被山围着的小镇上，有个叫阿明的少年。");
        guard.BeginResponse(8);
        guard.AppendScheduledSpeech(8, "现在开始播放天气预报。");

        Assert.False(guard.Match("在一座被山围着的小镇上").IsMatch);
    }
}
