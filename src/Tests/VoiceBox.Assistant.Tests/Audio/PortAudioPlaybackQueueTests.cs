using VoiceBox.Audio.PortAudio.Backend;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class PortAudioPlaybackQueueTests
{
    [Fact]
    public void ClearDiscardsActiveAndQueuedAudioWithoutCompletingTheNextPlayback()
    {
        using var queue = new PortAudioPlaybackQueue(2);
        Assert.True(queue.TryEnqueue([1f, 2f, 3f, 4f], out _));
        Assert.True(queue.TryEnqueue([5f, 6f], out _));
        float[] output = new float[2];
        Assert.False(queue.Fill(output));
        Assert.Equal([1f, 2f], output);
        Assert.True(queue.IsPlaying);

        queue.Clear();
        Assert.False(queue.IsPlaying);
        Assert.Equal(0, queue.Count);
        Assert.False(queue.Fill(output));
        Assert.Equal([0f, 0f], output);

        Assert.True(queue.TryEnqueue([7f, 8f], out _));
        Assert.False(queue.Fill(output));
        Assert.Equal([7f, 8f], output);
        Assert.True(queue.Fill(output));
        Assert.False(queue.Fill(output));
    }

    [Fact]
    public void FillPreservesChunkOrderAndPadsTheEndWithSilence()
    {
        using var queue = new PortAudioPlaybackQueue(2);
        queue.TryEnqueue([1f, 2f, 3f], out _);
        queue.TryEnqueue([4f, 5f, 6f], out _);
        float[] output = new float[4];

        Assert.False(queue.Fill(output));
        Assert.Equal([1f, 2f, 3f, 4f], output);
        Assert.True(queue.IsPlaying);
        Assert.True(queue.Fill(output));
        Assert.Equal([5f, 6f, 0f, 0f], output);
        Assert.False(queue.IsPlaying);
        Assert.False(queue.Fill(output));
        Assert.Equal([0f, 0f, 0f, 0f], output);
    }

    [Fact]
    public void EnqueueCopiesInputAndRejectsOverflowWithoutReplacingQueuedAudio()
    {
        using var queue = new PortAudioPlaybackQueue(1);
        float[] input = [1f, 2f];
        Assert.True(queue.TryEnqueue(input, out int dropped));
        Assert.Equal(0, dropped);
        input.AsSpan().Fill(9f);
        Assert.False(queue.TryEnqueue([3f, 4f, 5f], out dropped));
        Assert.Equal(3, dropped);

        float[] output = new float[2];
        queue.Fill(output);
        Assert.Equal([1f, 2f], output);
        queue.Complete();
        Assert.False(queue.TryEnqueue([6f], out dropped));
        Assert.Equal(1, dropped);
    }

    [Fact]
    public async Task ConcurrentClearsKeepCallbacksAndLaterPlaybackUsable()
    {
        using var queue = new PortAudioPlaybackQueue(4);
        using var start = new Barrier(2);
        Task callbacks = Task.Factory.StartNew(() =>
        {
            Assert.True(start.SignalAndWait(TimeSpan.FromSeconds(5)));
            float[] output = new float[513];
            for (int i = 0; i < 5000; i++)
            {
                output.AsSpan().Fill(-1f);
                queue.Fill(output);
                Assert.All(output, sample => Assert.True(sample is 0f or 1f));
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        Task interruptions = Task.Factory.StartNew(() =>
        {
            Assert.True(start.SignalAndWait(TimeSpan.FromSeconds(5)));
            float[] input = Enumerable.Repeat(1f, 1024).ToArray();
            for (int i = 0; i < 5000; i++)
            {
                queue.TryEnqueue(input, out _);
                queue.Clear();
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        await Task.WhenAll(callbacks, interruptions).WaitAsync(TimeSpan.FromSeconds(10));
        queue.Clear();
        Assert.False(queue.IsPlaying);
        Assert.Equal(0, queue.Count);
        Assert.True(queue.TryEnqueue([2f, 3f], out _));
        float[] final = new float[3];
        Assert.True(queue.Fill(final));
        Assert.Equal([2f, 3f, 0f], final);
    }
}
