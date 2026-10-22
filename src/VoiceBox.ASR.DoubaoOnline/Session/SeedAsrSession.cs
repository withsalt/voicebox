using System.Buffers;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.DoubaoOnline.Transport;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.DoubaoOnline.Session;

internal sealed class SeedAsrSession : AsrSession
{
    private readonly Channel<PooledAudio> _audioChannel;
    private readonly Channel<AsrResult> _resultChannel;
    private readonly TaskCompletionSource<AsrResult> _finalResult =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly long _maxBufferedSamples;
    private readonly DoubaoOnlineASROptions? _options;
    private readonly ILogger _logger;
    private readonly SeedAsrConnectionWorker? _worker;
    private long _bufferedSamples;
    private int _state = (int)SeedAsrSessionState.Active;
    private AsrResult? _latestResult;

    public string RequestId { get; }
    public string ConnectId { get; }
    public ChannelReader<AsrResult> ResultReader => _resultChannel.Reader;
    public SeedAsrSessionState State => (SeedAsrSessionState)Volatile.Read(ref _state);
    public long BufferedSampleCount => Volatile.Read(ref _bufferedSamples);
    public long MaxBufferedSamples => _maxBufferedSamples;
    public Task<AsrResult> FinalResult => _finalResult.Task;
    public string? LogId { get; set; }
    public AsrResult? LatestResult => Volatile.Read(ref _latestResult);

    internal SeedAsrSession(
        string sessionId,
        string requestId,
        string connectId,
        long maxBufferedSamples,
        IWebSocketChannel? channel = null,
        DoubaoOnlineASROptions? options = null,
        string? requestJson = null,
        ILogger? logger = null)
        : base(sessionId)
    {
        if (maxBufferedSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxBufferedSamples));
        RequestId = requestId;
        ConnectId = connectId;
        _maxBufferedSamples = maxBufferedSamples;
        _options = options;
        _logger = logger ?? NullLogger.Instance;

        _audioChannel = Channel.CreateUnbounded<PooledAudio>(
            new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false,
                AllowSynchronousContinuations = false,
            });
        _resultChannel = Channel.CreateBounded<AsrResult>(
            new BoundedChannelOptions(64)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true,
            });

        if (channel is not null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentException.ThrowIfNullOrWhiteSpace(requestJson);
            _worker = new SeedAsrConnectionWorker(this, channel, options, requestJson, _logger);
        }

        _ = _finalResult.Task.ContinueWith(
            static t => _ = t.Exception,
            CancellationToken.None,
            TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }

    protected override async Task StartSessionAsync(CancellationToken ct)
    {
        if (_worker is not null)
            await _worker.StartAsync(ct).ConfigureAwait(false);
    }

    public override void FeedAudio(ReadOnlySpan<float> audio)
        => TryFeedAudio(audio);

    protected override async Task<AsrResult?> FinishSessionAsync()
    {
        if (_worker is null)
            throw new InvalidOperationException("The test-only session has no transport worker.");

        MarkFinishing();
        CancellationToken ct = CancellationToken;
        AsrResult? result;
        try
        {
            result = _options!.FinishTimeoutSec <= 0
                ? await FinalResult.WaitAsync(ct).ConfigureAwait(false)
                : await FinalResult.WaitAsync(TimeSpan.FromSeconds(_options.FinishTimeoutSec), ct).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            _logger.LogWarning("SeedASR session {SessionId} timed out waiting for final result.", SessionId);
            result = LatestResult;
            _worker.SignalAbort();
            return result;
        }

        using var stopCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        int budget = _options.FinishTimeoutSec > 0 ? _options.FinishTimeoutSec : 10;
        stopCts.CancelAfter(TimeSpan.FromSeconds(budget));
        await _worker.StopAsync(stopCts.Token).ConfigureAwait(false);
        ct.ThrowIfCancellationRequested();
        return result;
    }

    protected override Task CancelSessionAsync()
    {
        AbortWork();
        return Task.CompletedTask;
    }

    protected override async Task ReleaseSessionAsync()
    {
        try
        {
            if (_worker is not null)
                await _worker.AbortAsync().ConfigureAwait(false);
        }
        finally
        {
            DrainAudio();
            MarkFinished();
        }
    }

    /// <summary>Queues audio and returns false after the session has started closing.</summary>
    internal bool TryFeedAudio(ReadOnlySpan<float> audio)
    {
        if (State != SeedAsrSessionState.Active || FailureException is not null || audio.IsEmpty) return false;
        if (!TryReserveSamples(audio.Length, out long bufferedBefore))
        {
            var overflow = new SeedAsrAudioBufferOverflowException(
                SessionId, _maxBufferedSamples, bufferedBefore, audio.Length);
            if (FailAudioIngress(overflow)) throw overflow;
            return false;
        }

        float[]? rented = null;
        bool ownsReservation = true;
        try
        {
            rented = ArrayPool<float>.Shared.Rent(audio.Length);
            audio.CopyTo(rented);
            var pooled = new PooledAudio(rented, audio.Length);
            if (State == SeedAsrSessionState.Active && _audioChannel.Writer.TryWrite(pooled))
            {
                rented = null;
                ownsReservation = false;
                return true;
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<float>.Shared.Return(rented);
            if (ownsReservation) ReleaseSamples(audio.Length);
        }
        return false;
    }

    internal bool TryReadAudio(out PooledAudio audio)
    {
        if (!_audioChannel.Reader.TryRead(out audio)) return false;
        ReleaseSamples(audio.Length);
        return true;
    }

    internal ValueTask<bool> WaitToReadAudioAsync(CancellationToken ct)
        => _audioChannel.Reader.WaitToReadAsync(ct);

    internal void PublishResult(AsrResult result)
    {
        Volatile.Write(ref _latestResult, result);
        _resultChannel.Writer.TryWrite(result);
    }

    internal void MarkFinishing()
    {
        Interlocked.CompareExchange(ref _state, (int)SeedAsrSessionState.Finishing,
            (int)SeedAsrSessionState.Active);
        _audioChannel.Writer.TryComplete();
    }

    internal void TrySetFinalResult(AsrResult result)
    {
        MarkFinishing();
        _finalResult.TrySetResult(result);
        _resultChannel.Writer.TryComplete();
    }

    protected override void OnSessionFaulted(Exception ex)
    {
        Interlocked.CompareExchange(ref _state, (int)SeedAsrSessionState.Finishing,
            (int)SeedAsrSessionState.Active);
        _finalResult.TrySetException(ex);
        _resultChannel.Writer.TryComplete(ex);
        // Keep the original failure available to Finish while stopping any
        // blocked peer loop. Abort before completing ingress so the sender cannot
        // mistake a fault for a graceful finish and emit a final audio packet.
        try { _worker?.AbortTransport(); }
        finally { _audioChannel.Writer.TryComplete(); }
    }

    internal void AbortWork()
    {
        Interlocked.CompareExchange(ref _state, (int)SeedAsrSessionState.Finishing,
            (int)SeedAsrSessionState.Active);
        _resultChannel.Writer.TryComplete();
        _finalResult.TrySetCanceled();
        _audioChannel.Writer.TryComplete();
    }

    internal void MarkFinished() => Volatile.Write(ref _state, (int)SeedAsrSessionState.Finished);

    private bool TryReserveSamples(int sampleCount, out long bufferedBefore)
    {
        while (true)
        {
            bufferedBefore = Volatile.Read(ref _bufferedSamples);
            if (sampleCount > _maxBufferedSamples - bufferedBefore) return false;
            if (Interlocked.CompareExchange(ref _bufferedSamples, bufferedBefore + sampleCount, bufferedBefore) == bufferedBefore)
                return true;
        }
    }

    private void ReleaseSamples(int sampleCount)
    {
        long remaining = Interlocked.Add(ref _bufferedSamples, -sampleCount);
        if (remaining < 0)
        {
            Interlocked.Exchange(ref _bufferedSamples, 0);
            throw new InvalidOperationException("SeedASR audio buffer accounting underflow.");
        }
    }

    private bool FailAudioIngress(Exception error)
    {
        if (Interlocked.CompareExchange(ref _state, (int)SeedAsrSessionState.Finishing,
                (int)SeedAsrSessionState.Active) != (int)SeedAsrSessionState.Active)
            return false;
        _ = FailAsync(error);
        return true;
    }

    internal void DrainAudio()
    {
        while (TryReadAudio(out var item)) item.Dispose();
    }
}
