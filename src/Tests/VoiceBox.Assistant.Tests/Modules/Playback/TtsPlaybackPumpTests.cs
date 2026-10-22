using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Audio.Common.Resampling.Service;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class TtsPlaybackPumpTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SynthesisFailure_CancelsAndReleasesWaitingTextSource(bool throwingCancellationCallback)
    {
        var failure = new InvalidOperationException("synthesis failed");
        var tts = new RecordingSynthesisPort { Failure = failure };
        using var resampler = new PassthroughResampler();
        using var stop = new CancellationTokenSource();
        var released = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var pump = CreatePump(tts, resampler);
        Task<string> run = pump.RunAsync(
            WaitingSource(released, throwingCancellationCallback, stop.Token), stop.Token);
        try
        {
            Exception actual = await Assert.ThrowsAsync<InvalidOperationException>(
                () => run.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Same(failure, actual);
            Assert.True(released.Task.IsCompletedSuccessfully);
        }
        finally
        {
            stop.Cancel();
            try { await run; } catch { }
        }
    }

    [Fact]
    public async Task TextSourceFailure_CancelsAndJoinsWaitingSynthesis()
    {
        var failure = new InvalidOperationException("text failed");
        var tts = new RecordingSynthesisPort { WaitUntilCancelled = true };
        using var resampler = new PassthroughResampler();
        using var stop = new CancellationTokenSource();
        var pump = CreatePump(tts, resampler);
        Task<string> run = pump.RunAsync(FailingSource(tts.FirstSynthesis.Task, failure), stop.Token);
        try
        {
            Exception actual = await Assert.ThrowsAsync<InvalidOperationException>(
                () => run.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.Same(failure, actual);
            Assert.True(tts.Released.Task.IsCompletedSuccessfully);
        }
        finally
        {
            stop.Cancel();
            try { await run; } catch { }
        }
    }

    private static TtsPlaybackPump CreatePump(ISpeechSynthesisStreamPort tts, IAudioResampleService resampler)
        => new(tts, new AcceptingPlaybackPort(), NullLogger.Instance,
            minSentenceChars: 1, tailBufferMs: 0, prerollMs: 0, resampler,
            ownerTurnId: null, new NullPlaybackStartedSink(), new NullScheduledSpeechSink());

    private static async IAsyncEnumerable<SpeakableTextDelta> WaitingSource(
        TaskCompletionSource released,
        bool throwingCancellationCallback,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var registration = cancellationToken.Register(() =>
        {
            if (throwingCancellationCallback)
                throw new InvalidOperationException("cancellation callback failed");
        });
        try
        {
            yield return new SpeakableTextDelta("第一句。");
            await Task.Delay(Timeout.Infinite, cancellationToken);
        }
        finally
        {
            released.TrySetResult();
        }
    }

    private static async IAsyncEnumerable<SpeakableTextDelta> FailingSource(Task synthesisStarted, Exception failure)
    {
        yield return new SpeakableTextDelta("第一句。");
        await synthesisStarted;
        throw failure;
    }

    [Fact]
    public async Task ForcedSpeechBoundarySubmitsShortPromptBeforeSourceContinues()
    {
        var tts = new RecordingSynthesisPort();
        var playback = new AcceptingPlaybackPort();
        var releaseSource = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        using var resampler = new PassthroughResampler();
        var pump = new TtsPlaybackPump(
            tts,
            playback,
            NullLogger.Instance,
            minSentenceChars: 100,
            tailBufferMs: 0,
            prerollMs: 250,
            resampler,
            ownerTurnId: null,
            new NullPlaybackStartedSink(),
            new NullScheduledSpeechSink());

        Task<string> run = pump.RunAsync(
            PromptThenWaitAsync(releaseSource.Task),
            CancellationToken.None);

        string firstSynthesis = await tts.FirstSynthesis.Task.WaitAsync(TimeSpan.FromSeconds(1));
        await playback.FirstEnqueue.Task.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal("短提示", firstSynthesis);
        Assert.False(run.IsCompleted);

        releaseSource.TrySetResult();
        Assert.Equal("短提示最终回答。", await run.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(["短提示", "最终回答。"], tts.Texts);
    }

    [Fact]
    public async Task ForcedSpeechBoundaryFailsWhenSynthesisProducesNoAudio()
    {
        using var resampler = new PassthroughResampler();
        var pump = new TtsPlaybackPump(
            new EmptySynthesisPort(),
            new AcceptingPlaybackPort(),
            NullLogger.Instance,
            minSentenceChars: 100,
            tailBufferMs: 0,
            prerollMs: 250,
            resampler,
            ownerTurnId: null,
            new NullPlaybackStartedSink(),
            new NullScheduledSpeechSink());

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => pump.RunAsync(
                OneDeltaAsync(new SpeakableTextDelta(
                    "正在查询",
                    ForceSpeechBoundary: true)),
                CancellationToken.None));

        Assert.Contains("did not produce any audio", exception.Message, StringComparison.Ordinal);
    }

    private static async IAsyncEnumerable<SpeakableTextDelta> PromptThenWaitAsync(
        Task release,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return new SpeakableTextDelta("短提示", ForceSpeechBoundary: true);
        await release.WaitAsync(cancellationToken);
        yield return new SpeakableTextDelta("最终回答。");
    }

    private static async IAsyncEnumerable<SpeakableTextDelta> OneDeltaAsync(
        SpeakableTextDelta delta)
    {
        yield return delta;
        await Task.CompletedTask;
    }

    private sealed class RecordingSynthesisPort : ISpeechSynthesisStreamPort
    {
        public Exception? Failure { get; set; }

        public bool WaitUntilCancelled { get; set; }

        public TaskCompletionSource Released { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public int OutputSampleRate => 16000;

        public int OutputChannels => 1;

        public List<string> Texts { get; } = [];

        public TaskCompletionSource<string> FirstSynthesis { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public async IAsyncEnumerable<SynthesizedAudioChunk> SynthesizeAsync(
            string text,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            Texts.Add(text);
            FirstSynthesis.TrySetResult(text);
            try
            {
                await Task.Yield();
                if (Failure is not null)
                    throw Failure;
                if (WaitUntilCancelled)
                    await Task.Delay(Timeout.Infinite, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                yield return new SynthesizedAudioChunk([0.1f], OutputSampleRate, OutputChannels);
            }
            finally
            {
                Released.TrySetResult();
            }
        }
    }

    private sealed class EmptySynthesisPort : ISpeechSynthesisStreamPort
    {
        public int OutputSampleRate => 16000;

        public int OutputChannels => 1;

        public async IAsyncEnumerable<SynthesizedAudioChunk> SynthesizeAsync(
            string text,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            yield break;
        }
    }

    private sealed class AcceptingPlaybackPort : IBoundedPlaybackPort
    {
        public TaskCompletionSource FirstEnqueue { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public bool IsActive => true;

        public string StateName => "Active";

        public int CurrentQueueSize => 0;

        public bool IsPlaying => false;

        public bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples)
        {
            droppedSamples = 0;
            FirstEnqueue.TrySetResult();
            return true;
        }
    }

    private sealed class PassthroughResampler : IAudioResampleService
    {
        public int TargetSampleRate => 16000;

        public int TargetChannels => 1;

        public int Convert(
            ReadOnlySpan<float> samples,
            int sourceSampleRate,
            int sourceChannels,
            ref float[] output)
        {
            if (output.Length < samples.Length)
                output = new float[samples.Length];
            samples.CopyTo(output);
            return samples.Length;
        }

        public int Flush(ref float[] output) => 0;

        public float[] Convert(
            ReadOnlySpan<float> samples,
            int sourceSampleRate,
            int sourceChannels) => samples.ToArray();

        public float[] Convert(ReadOnlySpan<float> samples) => samples.ToArray();

        public float[] Flush() => [];

        public void Reset() { }

        public void Dispose() { }
    }

    private sealed class NullPlaybackStartedSink : IPlaybackStartedSink
    {
        public void OnPlaybackStarted(long turnId) { }
    }

    private sealed class NullScheduledSpeechSink : IScheduledSpeechSink
    {
        public void OnSpeechScheduled(long turnId, string text) { }
    }
}
