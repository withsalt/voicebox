using VoiceBox.Assistant.Modules.Response.Formatting;
using Xunit;
using Xunit.Abstractions;

namespace VoiceBox.Assistant.Tests;

public sealed class TtsTextSanitizerTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("**你好，VoiceBox 👋!**", "你好，VoiceBox !")]
    [InlineData("☀⌚⬆\ue000\u200b\u200c\u200d\ufe0e\ufe0f`~*", "")]
    [InlineData("*a\r\n\t 1，。！*", "a\r\n\t 1，。！")]
    public void Clean_PreservesExistingUnicodeAndFilteringBehavior(string input, string expected)
        => Assert.Equal(expected, TtsTextSanitizer.Clean(input));

    [Fact]
    public void Clean_PreservesSupplementaryRunesAndReplacesUnpairedSurrogates()
    {
        string input = "*𠀀" + (char)0xd800 + "A" + (char)0xdc00 + "*";
        Assert.Equal("𠀀\ufffdA\ufffd", TtsTextSanitizer.Clean(input));
    }

    [Fact]
    public void Clean_UnchangedInputUsesSameString()
    {
        string input = new("你好𠀀".ToCharArray());
        Assert.Same(input, TtsTextSanitizer.Clean(input));
    }

    [Fact]
    public void Clean_FilteredTextDoesNotAllocatePerRetainedRune()
    {
        const string input = "**你好，VoiceBox 👋!**";
        for (int i = 0; i < 1000; i++)
            _ = TtsTextSanitizer.Clean(input);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++)
            _ = TtsTextSanitizer.Clean(input);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        output.WriteLine($"10000 filtered fragments allocated {allocated} bytes.");
        Assert.InRange(allocated, 1, 2_000_000);
    }
}
