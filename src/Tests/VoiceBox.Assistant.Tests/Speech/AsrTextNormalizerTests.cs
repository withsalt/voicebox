using VoiceBox.Assistant.Modules.Recognition.Formatting;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AsrTextNormalizerTests
{
    [Theory]
    [InlineData(null, "")]
    [InlineData("", "")]
    [InlineData("  <sil> 你好 <noise> ", "你好")]
    [InlineData("请打开<laughter>灯", "请打开灯")]
    public void Sanitize_RemovesShortAngleBracketTokens(string? input, string expected)
    {
        Assert.Equal(expected, AsrTextNormalizer.Sanitize(input));
    }

    [Fact]
    public void Sanitize_KeepsLongAngleBracketContent()
    {
        string longToken = "<" + new string('x', 33) + ">";

        Assert.Equal(longToken, AsrTextNormalizer.Sanitize(longToken));
    }

    [Theory]
    [InlineData("你好")]
    [InlineData("ok")]
    [InlineData("123")]
    [InlineData("嗯")]
    public void HasMeaningfulCharacters_ReturnsTrueForLettersOrDigits(string text)
    {
        Assert.True(AsrTextNormalizer.HasMeaningfulCharacters(text));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("？！。。。")]
    public void HasMeaningfulCharacters_ReturnsFalseForWhitespaceOrPunctuation(string text)
    {
        Assert.False(AsrTextNormalizer.HasMeaningfulCharacters(text));
    }

    [Theory]
    [InlineData("好的，再 见！", true)]
    [InlineData("我们下次聊。", true)]
    [InlineData("结束一下", false)]
    [InlineData("只是聊天", false)]
    public void ContainsAnyPhrase_IgnoresPunctuationAndWhitespace(string text, bool expected)
    {
        Assert.Equal(expected, AsrTextNormalizer.ContainsAnyPhrase(text, ["再见", "下次聊"]));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("？！。。。")]
    [InlineData("嗯")]
    [InlineData("啊")]
    [InlineData("嗯嗯")]
    [InlineData("uh")]
    [InlineData("um")]
    [InlineData("hmm")]
    public void IsMeaningfulPlaybackBargeInText_ReturnsFalseForEmptyOrShortFillers(string text)
    {
        Assert.False(AsrTextNormalizer.IsMeaningfulPlaybackBargeInText(text));
    }

    [Theory]
    [InlineData("你好")]
    [InlineData("停一下")]
    [InlineData("ok")]
    [InlineData("别说了")]
    public void IsMeaningfulPlaybackBargeInText_ReturnsTrueForActionableText(string text)
    {
        Assert.True(AsrTextNormalizer.IsMeaningfulPlaybackBargeInText(text));
    }
}
