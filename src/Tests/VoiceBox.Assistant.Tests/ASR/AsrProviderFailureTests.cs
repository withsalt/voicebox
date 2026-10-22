using Microsoft.Extensions.Logging;
using VoiceBox.ASR.Confucius4;
using VoiceBox.ASR.Confucius4.Internal;
using VoiceBox.ASR.Confucius4.Streaming;
using VoiceBox.ASR.DoubaoOnline;
using VoiceBox.ASR.DoubaoOnline.Session;
using VoiceBox.ASR.DoubaoOnline.Transport;
using VoiceBox.ASR.FireRedASR2.Internal;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.ASR.FireRedASR2.Session;
using VoiceBox.ASR.Qwen3.Internal;
using VoiceBox.ASR.Qwen3.Session;
using Xunit;
using QwenMelState = VoiceBox.ASR.Qwen3.Internal.IncrementalMelState;
using ConfuciusMelState = VoiceBox.ASR.Confucius4.Internal.IncrementalMelState;

namespace VoiceBox.Assistant.Tests.ASR;

public sealed class AsrProviderFailureTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    [Fact]
    public async Task ConfuciusDecodeFailure_AutomaticallyReleasesAndFaultsResultStream()
    {
        var failure = new IOException("decode failed");
        using var gate = new SemaphoreSlim(1);
        using var runtime = new FailingConfuciusRuntime(failure);
        await using var session = new Confucius4Session("confucius-failure", new(), runtime, gate);
        await session.StartAsync(default);
        var results = session.Read(default);

        session.FeedAudio(new float[5120]);
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(async () =>
        {
            await foreach (var result in results) { }
        }).WaitAsync(Timeout));
        await session.ReleaseTask.WaitAsync(Timeout);

        Assert.Same(failure, session.FailureException);
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.FinishAsync(default)));
    }

    [Fact]
    public async Task QwenFinalDecodeFailure_ReleasesBuffersAndPreservesCause()
    {
        var failure = new IOException("final decode failed");
        using var gate = new SemaphoreSlim(1);
        using var runtime = new TestQwenRuntime(failure, failCalls: int.MaxValue);
        await using var session = CreateQwen(runtime, gate, chunkSize: 32000);
        await session.StartAsync(default);
        session.FeedAudio(new float[160]);

        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.FinishAsync(default)).WaitAsync(Timeout));

        Assert.Same(failure, session.FailureException);
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.ResultReader.Completion));
        Assert.Equal(Qwen3SessionState.Finished, session.State);
        Assert.Equal(0, session.AccumulatedLength);
    }

    [Fact]
    public async Task QwenPartialDecodeFailure_RetainsAudioForSuccessfulFinalRetry()
    {
        using var gate = new SemaphoreSlim(1);
        using var runtime = new TestQwenRuntime(new IOException("recoverable partial failure"), failCalls: 1);
        await using var session = CreateQwen(runtime, gate);
        await session.StartAsync(default);
        session.FeedAudio(new float[160]);
        await runtime.FirstCall.Task.WaitAsync(Timeout);

        var result = await session.FinishAsync(default).WaitAsync(Timeout);

        Assert.Equal("complete", result!.Text);
        Assert.Null(session.FailureException);
        Assert.Equal(new[] { 160, 160 }, runtime.AudioLengths);
        Assert.Equal(Qwen3SessionState.Finished, session.State);
    }

    [Fact]
    public async Task QwenPumpFailure_IsNotReplacedBySecondaryLoggingFailure()
    {
        var pumpFailure = new IOException("partial warning logger failed");
        using var gate = new SemaphoreSlim(1);
        using var runtime = new TestQwenRuntime(new Exception("partial decode failed"), failCalls: 1);
        await using var session = CreateQwen(runtime, gate, logger: new ThrowingLogger(pumpFailure));
        await session.StartAsync(default);

        session.FeedAudio(new float[160]);
        await session.ReleaseTask.WaitAsync(Timeout);

        Assert.Same(pumpFailure, session.FailureException);
        Assert.Same(pumpFailure, await Assert.ThrowsAsync<IOException>(() => session.ResultReader.Completion));
        Assert.Equal(Qwen3SessionState.Finished, session.State);
        Assert.Equal(0, session.AccumulatedLength);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    [InlineData(true, true)]
    [InlineData(false, true)]
    public async Task DoubaoTransportFailure_AutomaticallyDisposesTransportAndPreservesCause(
        bool failSend, bool unexpectedCancellation = false)
    {
        Exception failure = unexpectedCancellation
            ? new OperationCanceledException("transport cancelled without requesting session cancellation")
            : new IOException(failSend ? "send failed" : "receive failed");
        var channel = new TestWebSocket(failure, failSend);
        await using var session = new SeedAsrSession("doubao-failure", "request", "connection", 16000,
            channel, new DoubaoOnlineASROptions { CompressAudio = false, ThrottleSendToRealtime = false },
            "{}", new ThrowingLogger(new Exception("logger failed")));
        await session.StartAsync(default);

        if (failSend) session.FeedAudio(new float[3200]);
        else channel.ReceiveFailure.SetException(failure);
        await session.ReleaseTask.WaitAsync(Timeout);

        var error = Assert.IsType<DoubaoOnlineASRException>(session.FailureException);
        Assert.Same(failure, error.InnerException);
        Assert.Same(error, await Assert.ThrowsAsync<DoubaoOnlineASRException>(() => session.ResultReader.Completion));
        Assert.Same(error, await Assert.ThrowsAsync<DoubaoOnlineASRException>(() => session.FinalResult));
        Assert.Same(error, await Assert.ThrowsAsync<DoubaoOnlineASRException>(() => session.FinishAsync(default)));
        Assert.Equal(SeedAsrSessionState.Finished, session.State);
        Assert.Equal(0, session.BufferedSampleCount);
        Assert.Equal(1, channel.DisposeCount);
        Assert.True(channel.AbortCount > 0);
    }

    [Fact]
    public async Task DoubaoIngressOverflow_AutomaticallyReleasesAndRetainsOverflowDetails()
    {
        await using var session = new SeedAsrSession("overflow", "request", "connection", maxBufferedSamples: 1);
        await session.StartAsync(default);

        var error = Assert.Throws<SeedAsrAudioBufferOverflowException>(() => session.FeedAudio(new float[2]));
        await session.ReleaseTask.WaitAsync(Timeout);

        Assert.Same(error, session.FailureException);
        Assert.Same(error, await Assert.ThrowsAsync<SeedAsrAudioBufferOverflowException>(() => session.FinalResult));
        Assert.Equal(0, session.BufferedSampleCount);
        Assert.Equal(SeedAsrSessionState.Finished, session.State);
    }

    [Fact]
    public async Task FireRedFeatureAppendFailure_AutomaticallyReleasesInsteadOfDroppingChunk()
    {
        string cmvn = Path.GetTempFileName();
        try
        {
            File.WriteAllText(cmvn, $"<AddShift>\n<LearnRateCoef> [ {string.Join(' ', Enumerable.Repeat(0, 80))} 1 ]\n" +
                $"<Rescale>\n<LearnRateCoef> [ {string.Join(' ', Enumerable.Repeat(1, 80))} 1 ]\n");
            using var gate = new SemaphoreSlim(1);
            using var engine = new UnusedFireRedEngine();
            using var recognizer = new OfflineRecognizer(engine, new FireRedTokenAssembler(["<blank>"]), cmvn);
            var stream = recognizer.CreateOfflineStream();
            await using var session = new FireRedSession("firered-failure", stream, 8, recognizer, gate, 5,
                new ThrowingLogger(new Exception("logger failed")));
            await session.StartAsync(default);
            // Simulate a feature extractor that is no longer usable while the pump is idle.
            stream.Dispose();

            session.FeedAudio(new float[160]);
            await session.ReleaseTask.WaitAsync(Timeout);

            var error = Assert.IsType<ObjectDisposedException>(session.FailureException);
            Assert.Same(error, await Assert.ThrowsAsync<ObjectDisposedException>(() => session.FinishAsync(default)));
            Assert.Equal(FireRedSessionState.Finished, session.State);
            Assert.False(session.TryReadAudio(out _));
        }
        finally { File.Delete(cmvn); }
    }

    private static Qwen3AsrSession CreateQwen(TestQwenRuntime runtime, SemaphoreSlim gate,
        int chunkSize = 160, ILogger? logger = null)
        => new("qwen-failure", "", null, chunkSize, 2, 5, 16, 8, runtime, gate, 5, logger);

    private sealed class FailingConfuciusRuntime(Exception error) : IConfucius4Runtime
    {
        public string Rollback(string text, int count) => text;
        public string Decode(ReadOnlySpan<float> audio, string prefix, int maxTokens,
            ConfuciusMelState melState, CancellationToken cancellationToken, bool final = false) => throw error;
        public void Dispose() { }
    }

    private sealed class TestQwenRuntime(Exception error, int failCalls) : IQwen3AsrRuntime
    {
        public List<int> AudioLengths { get; } = [];
        public TaskCompletionSource FirstCall { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public void Warmup(double audioSeconds = 1.0, int maxNewTokens = 4, CancellationToken ct = default) { }
        public string? BuildStreamingPrefix(int chunkId, string accumulatedRaw, int unfixedChunkNum, int unfixedTokenNum) => null;
        public Qwen3AsrTranscription Transcribe(ReadOnlySpan<float> waveform, string context, string? language,
            string? prefixText, CancellationToken ct, QwenMelState? melState = null)
        {
            AudioLengths.Add(waveform.Length);
            FirstCall.TrySetResult();
            if (AudioLengths.Count <= failCalls) throw error;
            return new("complete", "Chinese", 0, 0, 0, 0, 0, 0, 0, 1);
        }
        public void Dispose() { }
    }

    private sealed class TestWebSocket(Exception error, bool failSend) : IWebSocketChannel
    {
        private int _sends;
        public TaskCompletionSource<byte[]?> ReceiveFailure { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public int DisposeCount;
        public int AbortCount;
        public string? LogId => "test-log-id";
        public Task ConnectAsync(CancellationToken ct) => Task.CompletedTask;
        public Task SendAsync(ReadOnlyMemory<byte> frame, CancellationToken ct)
        {
            if (Interlocked.Increment(ref _sends) > 1 && failSend) throw error;
            return Task.CompletedTask;
        }
        public Task<byte[]?> ReceiveAsync(CancellationToken ct) => ReceiveFailure.Task.WaitAsync(ct);
        public Task CloseAsync(CancellationToken ct) => Task.CompletedTask;
        public void Abort() => Interlocked.Increment(ref AbortCount);
        public void Dispose() => Interlocked.Increment(ref DisposeCount);
    }

    private sealed class ThrowingLogger(Exception warningError) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
            => throw (logLevel == LogLevel.Warning ? warningError : new InvalidOperationException("secondary logging failure"));
    }

    private sealed class UnusedFireRedEngine : IAsrInferenceEngine
    {
        public AsrModelMetadata Metadata { get; } = new()
        {
            BlankId = 0, UnkId = 1, PadId = 2, SosId = 3, EosId = 4,
            FeatureDim = 80, SampleRate = 16000, ChunkLength = 16, ShiftLength = 16,
            RequiredCacheSize = 0, EncoderHiddenDim = 1, DecoderCacheLayers = 1,
            EncoderFrameShiftSeconds = 0.04,
        };
        public bool HasDecoder => false;
        public bool HasCtc => false;
        public EncoderOutputEntity RunEncoder(IReadOnlyList<AsrInputEntity> inputs) => throw new NotSupportedException();
        public DecoderOutputEntity RunDecoderStep(long[] tokens, int tokensLen, float[] encoderOutputs,
            bool[] srcMask, IReadOnlyList<float[]> cacheList) => throw new NotSupportedException();
        public CtcOutputEntity RunCtc(float[] encoderOutputs, int batchSize = 1) => throw new NotSupportedException();
        public void Dispose() { }
    }
}
