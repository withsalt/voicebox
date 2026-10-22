using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class NearEndAudioReservoirTests
{
    [Fact]
    public void SnapshotLast_ReturnsBoundedTailInTimelineOrder()
    {
        var reservoir = new NearEndAudioReservoir(6);
        long position = 100;
        Add(reservoir, [1, 2, 3], 1, ref position);
        Add(reservoir, [4, 5, 6], 1, ref position);
        Add(reservoir, [7, 8], 1, ref position);

        Assert.Equal([4, 5, 6, 7, 8], Flatten(reservoir.SnapshotLast(5)));
        Assert.True(reservoir.TryGetBounds(out long epoch, out long start, out long end));
        Assert.Equal(1, epoch);
        Assert.Equal(102, start);
        Assert.Equal(108, end);
    }

    [Fact]
    public void ExactSnapshot_ReturnsCandidateIntervalWithoutShifting()
    {
        var reservoir = new NearEndAudioReservoir(12);
        long position = 50;
        Add(reservoir, [1, 2, 3, 4], 3, ref position);
        long candidateStart = 51;
        Add(reservoir, [5, 6, 7, 8], 3, ref position);

        Assert.True(reservoir.TrySnapshotFrom(3, candidateStart, 12, out IReadOnlyList<float[]> snapshot));
        Assert.Equal([2, 3, 4, 5, 6, 7, 8], Flatten(snapshot));
    }

    [Fact]
    public void ExactSnapshot_FailsWhenCandidateStartWasEvicted()
    {
        var reservoir = new NearEndAudioReservoir(4);
        long position = 0;
        Add(reservoir, [1, 2, 3, 4], 1, ref position);
        Add(reservoir, [5, 6], 1, ref position);

        Assert.False(reservoir.TrySnapshotFrom(1, 1, 10, out IReadOnlyList<float[]> snapshot));
        Assert.Empty(snapshot);
        Assert.Equal([3, 4, 5, 6], Flatten(reservoir.SnapshotLast(10)));
    }

    [Fact]
    public void ExactSnapshot_FailsInsteadOfTruncatingAtSafetyLimit()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3, 4, 5, 6], 1, ref position);

        Assert.False(reservoir.TrySnapshotFrom(1, 0, 4, out _));
    }

    [Fact]
    public void ExactRangeSnapshot_ReturnsOneRetainedIntervalAfterLaterFramesArrive()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3, 4], 1, ref position);
        Add(reservoir, [5, 6, 7, 8], 1, ref position);
        Add(reservoir, [9, 10], 1, ref position);

        Assert.True(reservoir.TrySnapshotRange(1, 4, 8, out float[] snapshot));
        Assert.Equal([5, 6, 7, 8], snapshot);
        Assert.False(reservoir.TrySnapshotRange(1, 1, 4, out _));
        Assert.False(reservoir.TrySnapshotRange(2, 4, 8, out _));
    }

    [Fact]
    public void ExactRangeSnapshot_AcceptsRetainedAndCurrentBoundsIncludingWrap()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3, 4, 5, 6, 7, 8], 1, ref position);
        Add(reservoir, [9, 10, 11, 12], 1, ref position);

        Assert.True(reservoir.TrySnapshotRange(1, 4, 12, out float[] snapshot));
        Assert.Equal([5, 6, 7, 8, 9, 10, 11, 12], snapshot);
    }

    [Fact]
    public void ExactRangeSnapshot_AcceptsZeroStartAndPartialRingWithoutWrap()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3], 1, ref position);

        Assert.True(reservoir.TrySnapshotRange(1, 0, 3, out float[] snapshot));
        Assert.Equal([1, 2, 3], snapshot);
    }

    [Fact]
    public void ExactRangeSnapshot_CopiesAnIntervalAcrossRingBoundary()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3, 4, 5, 6, 7, 8], 1, ref position);
        Add(reservoir, [9, 10, 11, 12], 1, ref position);

        Assert.True(reservoir.TrySnapshotRange(1, 6, 10, out float[] snapshot));
        Assert.Equal([7, 8, 9, 10], snapshot);
    }

    [Fact]
    public void ExactRangeSnapshot_RejectsNegativeReversedAndEmptyIntervals()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3], 1, ref position);

        Assert.False(reservoir.TrySnapshotRange(1, -1, 1, out _));
        Assert.False(reservoir.TrySnapshotRange(1, 2, 1, out _));
        Assert.False(reservoir.TrySnapshotRange(1, 2, 2, out _));
    }

    [Fact]
    public void GapResetsTimelineAndReportsDiscontinuity()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long position = 0;
        Add(reservoir, [1, 2, 3], 1, ref position);
        var gap = new CaptureRange(1, 2, 2, 10, 12, 2);

        bool reset = reservoir.Add([9, 10], in gap);

        Assert.True(reset);
        Assert.Equal([9, 10], Flatten(reservoir.SnapshotLast(8)));
        Assert.True(reservoir.TryGetBounds(out _, out long start, out long end));
        Assert.Equal(10, start);
        Assert.Equal(12, end);
    }

    [Fact]
    public void EpochChangeInvalidatesOldSnapshotDomain()
    {
        var reservoir = new NearEndAudioReservoir(8);
        long firstPosition = 0;
        Add(reservoir, [1, 2], 1, ref firstPosition);
        long secondPosition = 0;
        Add(reservoir, [3, 4], 2, ref secondPosition);

        Assert.False(reservoir.TrySnapshotFrom(1, 0, 8, out _));
        Assert.True(reservoir.TrySnapshotFrom(2, 0, 8, out IReadOnlyList<float[]> snapshot));
        Assert.Equal([3, 4], Flatten(snapshot));
    }

    [Fact]
    public void Locate_FindsVadChunkBeforeCurrentPredictTail()
    {
        var reservoir = new NearEndAudioReservoir(16);
        long position = 100;
        Add(reservoir, [1, 2, 3, 4, 5, 6, 7, 8], 7, ref position);

        Assert.True(reservoir.TryLocate([3, 4, 5], out long epoch, out long start, out long end));
        Assert.Equal(7, epoch);
        Assert.Equal(102, start);
        Assert.Equal(105, end);
    }

    private static void Add(
        NearEndAudioReservoir reservoir,
        float[] audio,
        long epoch,
        ref long position)
    {
        long start = position;
        position += audio.Length;
        var range = new CaptureRange(epoch, start + 1, start + 1, start, position, start + 1);
        reservoir.Add(audio, in range);
    }

    private static float[] Flatten(IReadOnlyList<float[]> chunks)
        => chunks.SelectMany(static chunk => chunk).ToArray();
}
