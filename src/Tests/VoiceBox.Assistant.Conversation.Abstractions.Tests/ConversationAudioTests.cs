using Xunit;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tests;

public sealed class ConversationAudioTests
{
    [Fact]
    public void AudioOwnsItsSamplesAndDescribesInterleavedFrames()
    {
        float[] samples = [0.1f, -0.1f, 0.2f, -0.2f];
        var audio = new ConversationAudio(samples, sampleRate: 8_000, channels: 2);
        var request = new ConversationRequest(Guid.NewGuid(), "你好", audio);
        Array.Clear(samples);

        Assert.Same(audio, request.Audio);
        Assert.Equal([0.1f, -0.1f, 0.2f, -0.2f], audio.Samples.ToArray());
        Assert.Equal(8_000, audio.SampleRate);
        Assert.Equal(2, audio.Channels);
        Assert.Equal(TimeSpan.FromMilliseconds(0.25), audio.Duration);
    }

    [Fact]
    public void TextRequestsHaveNoAudio()
        => Assert.Null(new ConversationRequest(Guid.NewGuid(), "你好").Audio);

    [Theory]
    [InlineData(0, 1)]
    [InlineData(-1, 1)]
    [InlineData(16_000, 0)]
    [InlineData(16_000, -1)]
    public void AudioRequiresAPositiveSampleRateAndChannelCount(int sampleRate, int channels)
        => Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationAudio([0f], sampleRate, channels));

    [Fact]
    public void AudioRequiresNonemptyCompleteFrames()
    {
        Assert.Throws<ArgumentException>(() => new ConversationAudio([], 16_000));
        Assert.Throws<ArgumentException>(() => new ConversationAudio([0f, 1f, 0f], 16_000, 2));
    }
}
