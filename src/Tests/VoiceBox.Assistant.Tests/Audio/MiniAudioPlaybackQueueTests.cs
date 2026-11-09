using VoiceBox.Audio.MiniAudio.Backend;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class MiniAudioPlaybackQueueTests
{
    [Fact]
    public void ClearingPlaybackDiscardsTheActiveTailAndQueuedAudio()
    {
        using var queue = new MiniAudioPlaybackQueue(4);
        Assert.True(queue.TryEnqueue([1f, 2f, 3f, 4f], out _));
        Assert.True(queue.TryEnqueue([5f, 6f], out _));
        float[] output = new float[2];
        queue.Fill(output);
        Assert.Equal([1f, 2f], output);

        queue.Clear();
        Assert.False(queue.IsPlaying);
        Assert.Equal(0, queue.Count);
        queue.Fill(output);
        Assert.Equal([0f, 0f], output);
        Assert.True(queue.TryEnqueue([7f, 8f], out _));
        queue.Fill(output);
        Assert.Equal([7f, 8f], output);
    }

    [Fact]
    public async Task ClearingPlaybackWhileCallbacksRunKeepsBuffersValid()
    {
        using var queue = new MiniAudioPlaybackQueue(4);
        var start = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task callbacks = Task.Run(async () =>
        {
            await start.Task;
            float[] output = new float[513];
            for (int i = 0; i < 5000; i++)
            {
                queue.Fill(output);
                Assert.All(output, sample => Assert.True(sample is 0f or 1f));
            }
        });
        Task interruptions = Task.Run(async () =>
        {
            await start.Task;
            float[] input = Enumerable.Repeat(1f, 1024).ToArray();
            for (int i = 0; i < 5000; i++)
            {
                queue.TryEnqueue(input, out _);
                queue.Clear();
            }
        });
        start.SetResult();
        await Task.WhenAll(callbacks, interruptions).WaitAsync(TimeSpan.FromSeconds(10));
        queue.Clear();
        float[] final = [9f, 9f];
        queue.Fill(final);
        Assert.Equal([0f, 0f], final);
    }
}
