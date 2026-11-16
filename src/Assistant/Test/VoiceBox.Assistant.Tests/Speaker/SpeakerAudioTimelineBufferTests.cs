using VoiceBox.Assistant.Modules.Speaker;
using VoiceBox.Assistant.Modules.Speaker.Buffering;
using VoiceBox.Assistant.Modules.Speaker.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerAudioTimelineBufferTests
{
    [Fact]
    public void ContinuousAppend_SnapshotsExactSourceRange()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 4000, sampleRate: 16000);
        ring.Append(SpeakerTestProfiles.Frame(1, 0, Ramp(0, 320)));
        ring.Append(SpeakerTestProfiles.Frame(1, 320, Ramp(320, 320), sequence: 1));

        var destination = new float[100];
        Assert.True(ring.TrySnapshot(1, 300, 400, destination, out SpeakerTimelineFailure failure));
        Assert.Equal(SpeakerTimelineFailure.None, failure);
        Assert.Equal(300f, destination[0]);
        Assert.Equal(399f, destination[99]);
    }

    [Fact]
    public void WrapAround_KeepsNewestSuffixAndExpiresOlderAudio()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 1000, sampleRate: 16000);
        for (int i = 0; i < 5; i++)
            ring.Append(SpeakerTestProfiles.Frame(1, i * 400, Ramp(i * 400, 400), sequence: i));

        Assert.True(ring.TryGetBounds(out long epoch, out long segmentStart, out long retainedStart, out long end));
        Assert.Equal(1, epoch);
        Assert.Equal(2000, end);
        Assert.Equal(1000, retainedStart);
        Assert.Equal(0, segmentStart); // Retention does not move the hop-grid origin.

        var destination = new float[200];
        Assert.False(ring.TrySnapshot(1, 900, 1100, destination, out SpeakerTimelineFailure expired));
        Assert.Equal(SpeakerTimelineFailure.AudioExpired, expired);
        Assert.True(ring.TrySnapshot(1, 1800, 2000, destination, out _));
        Assert.Equal(1800f, destination[0]);
        Assert.Equal(1999f, destination[199]);
    }

    [Fact]
    public void Gap_FailsQueriesThatCrossItButServesTheNewSuffix()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8000, sampleRate: 16000);
        ring.Append(SpeakerTestProfiles.Frame(1, 0, Ramp(0, 800)));
        int generationBefore = ring.GapGeneration;
        // Missing [800, 1200): the producer dropped a frame.
        ring.Append(SpeakerTestProfiles.Frame(1, 1200, Ramp(1200, 800), sequence: 3));
        ring.Append(SpeakerTestProfiles.Frame(1, 2000, Ramp(2000, 800), sequence: 4));

        Assert.NotEqual(generationBefore, ring.GapGeneration);
        var destination = new float[1200];
        Assert.False(ring.TrySnapshot(1, 600, 1600, destination, out SpeakerTimelineFailure crossing));
        Assert.Equal(SpeakerTimelineFailure.TimelineGap, crossing);
        Assert.True(ring.TrySnapshot(1, 1200, 2400, destination, out _));
        Assert.Equal(1200f, destination[0]);
        Assert.Equal(2399f, destination[1199]);
    }

    [Fact]
    public void EpochChange_ResetsCoordinatesAndRejectsOldEpoch()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8000, sampleRate: 16000);
        ring.Append(SpeakerTestProfiles.Frame(1, 0, Ramp(0, 800)));
        ring.Append(SpeakerTestProfiles.Frame(2, 0, Ramp(5000, 800)));

        var destination = new float[800];
        Assert.False(ring.TrySnapshot(1, 0, 800, destination, out SpeakerTimelineFailure failure));
        Assert.Equal(SpeakerTimelineFailure.StreamEpochMismatch, failure);
        Assert.True(ring.TrySnapshot(2, 0, 800, destination, out _));
        Assert.Equal(5000f, destination[0]);
    }

    [Fact]
    public void NotYetAvailable_AndFormatMismatch_AreReported()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8000, sampleRate: 16000);
        ring.Append(SpeakerTestProfiles.Frame(1, 0, Ramp(0, 800)));
        var destination = new float[400];
        Assert.False(ring.TrySnapshot(1, 600, 1000, destination, out SpeakerTimelineFailure failure));
        Assert.Equal(SpeakerTimelineFailure.NotYetAvailable, failure);

        var stereo = SpeakerTestProfiles.Frame(1, 800, Ramp(800, 400), sequence: 1) with { Channels = 2 };
        Assert.False(ring.Append(stereo));
        Assert.True(ring.TryGetBounds(out _, out long segmentStart, out _, out long end));
        Assert.Equal(800, end);
        Assert.Equal(800, segmentStart);
    }

    [Fact]
    public void SpeechCount_UsesEnergyMaskOverTenMillisecondFrames()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8000, sampleRate: 16000);
        float[] audio = new float[1600];
        for (int i = 800; i < 1600; i++)
            audio[i] = (i & 1) == 0 ? 0.2f : -0.2f;
        ring.Append(SpeakerTestProfiles.Frame(1, 0, audio));

        Assert.True(ring.TryCountSpeechSamples(1, 0, 1600, 0.01f, out int speech, out _));
        Assert.Equal(800, speech);
    }

    [Fact]
    public void SpeechCount_HandlesRingWrapAndPartialFrame()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 13, sampleRate: 1000);
        float[] audio = new float[30];
        Array.Fill(audio, 0.1f, 18, 10);
        Array.Fill(audio, 0.5f, 28, 2);
        ring.Append(SpeakerTestProfiles.Frame(1, 0, audio) with { SampleRate = 1000 });

        Assert.True(ring.TryCountSpeechSamples(1, 18, 30, 0.2f, out int speech, out _));
        Assert.Equal(2, speech);
    }

    [Fact]
    public void SpeechCount_IgnoresNonFiniteSamplesAcrossRingWrap()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8, sampleRate: 1000);
        float[] audio = [0.5f, float.NaN, float.PositiveInfinity, 0.5f, float.NegativeInfinity];
        ring.Append(SpeakerTestProfiles.Frame(1, 6, audio) with { SampleRate = 1000 });

        Assert.True(ring.TryCountSpeechSamples(1, 6, 11, 0.2f, out int speech, out _));
        Assert.Equal(2, speech);
    }

    [Fact]
    public void SpeechCount_HandlesTheEndOfTheSampleTimeline()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 8, sampleRate: 1000);
        float[] audio = [0.5f, 0.5f];
        ring.Append(SpeakerTestProfiles.Frame(1, long.MaxValue - 2, audio) with { SampleRate = 1000 });

        Assert.True(ring.TryCountSpeechSamples(1, long.MaxValue - 2, long.MaxValue, 0.2f, out int speech, out _));
        Assert.Equal(2, speech);
    }

    [Fact]
    public async Task ConcurrentAppendAndSnapshot_NeverThrowOrReadOutOfBounds()
    {
        var ring = new SpeakerAudioTimelineBuffer(capacitySamples: 4000, sampleRate: 16000);
        using var stop = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
        Task producer = Task.Run(() =>
        {
            long position = 0;
            long sequence = 0;
            while (!stop.IsCancellationRequested)
            {
                ring.Append(SpeakerTestProfiles.Frame(1, position, Ramp(position, 160), sequence++));
                position += 160;
            }
        });
        Task consumer = Task.Run(() =>
        {
            var destination = new float[1000];
            while (!stop.IsCancellationRequested)
            {
                if (ring.TryGetBounds(out long epoch, out _, out long retained, out long end) && end - retained >= 1000)
                {
                    if (ring.TrySnapshot(epoch, end - 1000, end, destination, out _))
                        Assert.Equal(RampValue(end - 1000), destination[0]);
                }
            }
        });
        await Task.WhenAll(producer, consumer);
    }

    /// <summary>Sample value encoding its position; wrapped so it stays exactly representable as a float.</summary>
    private static float RampValue(long position) => position & 0xFFFFF;

    private static float[] Ramp(long start, int length)
    {
        var audio = new float[length];
        for (int i = 0; i < length; i++)
            audio[i] = RampValue(start + i);
        return audio;
    }
}
