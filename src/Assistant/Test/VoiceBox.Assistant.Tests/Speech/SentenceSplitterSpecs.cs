using VoiceBox.Assistant.Modules.Response.Formatting;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SentenceSplitterSpecs
{
    [Fact]
    public void PushUpdatesPendingTextBeforeEnumeration()
    {
        var splitter = new SentenceSplitter(1);

        IEnumerable<string> sentences = splitter.Push("pending text");

        Assert.Equal("pending text", splitter.Flush());
        Assert.Empty(sentences);
        Assert.Null(splitter.Flush());
    }

    [Fact]
    public void RepeatedEnumerationDoesNotRepeatBufferedText()
    {
        var splitter = new SentenceSplitter(1);

        IEnumerable<string> sentences = splitter.Push("First. Tail");

        Assert.Equal(["First."], sentences);
        Assert.Equal(["First."], sentences);
        Assert.Equal(" Tail", splitter.Flush());
    }

    [Fact]
    public void FragmentsShareTheMinimumLengthAndFlushTheFinalTail()
    {
        var splitter = new SentenceSplitter(5);

        Assert.Empty(splitter.Push("Hi!"));
        Assert.Equal(["Hi! OK。"], splitter.Push(" OK。tail"));
        Assert.Equal("tail", splitter.Flush());
        Assert.Null(splitter.Flush());
    }
}
