using System.Buffers;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.Qwen3.Internal;
using VoiceBox.ASR.Qwen3.Pump;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Qwen3.Session;

internal sealed class Qwen3AsrSession : AsrSession
{
    private readonly Channel<PooledAudio> _audioChannel;
    private readonly Channel<AsrResult> _resultChannel;
    private readonly CancellationTokenSource _workStop = new();
    private readonly ILogger _logger;
    private readonly Qwen3AsrAudioPump _pump;
    private readonly double _finishDrainTimeoutSec;
    private readonly object _bufferLock = new();
    private float[] _accumulated;
    private int _accumulatedLength;
    private long _droppedFrames;
    private long _lastDropLogTicks;
    private int _disposed;
    private int _state = (int)Qwen3SessionState.Active;

    public string Context { get; }
    public string? Language { get; }
    public int ChunkSizeSamples { get; }
    public int UnfixedChunkNum { get; }
    public int UnfixedTokenNum { get; }
    public int MaxNewTokens { get; }
    public ChannelReader<AsrResult> ResultReader => _resultChannel.Reader;
    public Qwen3SessionState State => (Qwen3SessionState)Volatile.Read(ref _state);
    public long DroppedFrameCount => Volatile.Read(ref _droppedFrames);
    public string LatestText { get; set; } = string.Empty;
    public string LatestLanguage { get; set; } = string.Empty;
    public int ChunkId { get; set; }

    public Qwen3AsrSession(
        string sessionId,
        string context,
        string? language,
        int chunkSizeSamples,
        int unfixedChunkNum,
        int unfixedTokenNum,
        int maxNewTokens,
        int maxBufferChunks,
        IQwen3AsrRuntime runtime,
        SemaphoreSlim recognitionGate,
        double finishDrainTimeoutSec,
        ILogger? logger = null)
        : base(sessionId)
    {
        if (chunkSizeSamples <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSizeSamples));
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(recognitionGate);

        Context = context ?? string.Empty;
        Language = language;
        ChunkSizeSamples = chunkSizeSamples;
        UnfixedChunkNum = unfixedChunkNum;
        UnfixedTokenNum = unfixedTokenNum;
        MaxNewTokens = maxNewTokens;
        _logger = logger ?? NullLogger.Instance;
        _finishDrainTimeoutSec = finishDrainTimeoutSec;

        _audioChannel = Channel.CreateBounded<PooledAudio>(
            new BoundedChannelOptions(maxBufferChunks)
            {
                FullMode = BoundedChannelFullMode.DropWrite,
                SingleReader = true,
                SingleWriter = false,
            },
            itemDropped: OnAudioChunkDropped);
        _resultChannel = Channel.CreateBounded<AsrResult>(
            new BoundedChannelOptions(64)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true,
            });
        _accumulated = ArrayPool<float>.Shared.Rent(chunkSizeSamples * 4);
        _pump = new Qwen3AsrAudioPump(this, runtime, recognitionGate, _logger);
    }

    internal CancellationToken WorkCancellationToken => _workStop.Token;

    protected override Task StartSessionAsync(CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _pump.Start();
        return Task.CompletedTask;
    }

    protected override async Task<AsrResult?> FinishSessionAsync()
    {
        using var drainCts = CancellationTokenSource.CreateLinkedTokenSource(CancellationToken);
        if (_finishDrainTimeoutSec > 0)
            drainCts.CancelAfter(TimeSpan.FromSeconds(_finishDrainTimeoutSec));
        await _pump.StopAsync(drainCts.Token).ConfigureAwait(false);
        CancellationToken.ThrowIfCancellationRequested();

        AsrResult result = _pump.RunFinalTranscription(CancellationToken);
        CancellationToken.ThrowIfCancellationRequested();
        PublishResult(result);
        LogSummary(_pump.GetMetricsSnapshot());
        return result;
    }

    public override void FeedAudio(ReadOnlySpan<float> audio)
    {
        if (State != Qwen3SessionState.Active || FailureException is not null || audio.IsEmpty) return;
        var rented = ArrayPool<float>.Shared.Rent(audio.Length);
        audio.CopyTo(rented);
        var pooled = new PooledAudio(rented, audio.Length);
        if (_audioChannel.Writer.TryWrite(pooled)) return;
        pooled.Dispose();
        OnFrameDropped();
    }

    protected override Task CancelSessionAsync()
    {
        AbortWork();
        return Task.CompletedTask;
    }

    protected override async Task ReleaseSessionAsync()
    {
        AbortWork();
        try
        {
            // A failed pump must not prevent the rest of the provider-owned
            // buffers from being released.
            try { await _pump.RunTask.ConfigureAwait(false); }
            catch { }
            DrainAudio();
            _pump.ResetState();
            Interlocked.Exchange(ref _disposed, 1);
            MarkFinished();

            float[] toReturn;
            lock (_bufferLock)
            {
                toReturn = _accumulated;
                _accumulated = Array.Empty<float>();
                _accumulatedLength = 0;
            }
            if (toReturn.Length > 0)
                ArrayPool<float>.Shared.Return(toReturn);
        }
        finally
        {
            try { _workStop.Dispose(); }
            catch (ObjectDisposedException) { }
        }
    }

    internal bool TryReadAudio(out PooledAudio audio) => _audioChannel.Reader.TryRead(out audio);

    internal ValueTask<bool> WaitToReadAsync(CancellationToken ct = default)
        => _audioChannel.Reader.WaitToReadAsync(ct);

    public bool AppendSamples(ReadOnlySpan<float> audio)
    {
        if (audio.IsEmpty) return false;
        lock (_bufferLock)
        {
            if (Volatile.Read(ref _disposed) != 0) return false;
            EnsureCapacity(_accumulatedLength + audio.Length);
            audio.CopyTo(_accumulated.AsSpan(_accumulatedLength));
            _accumulatedLength += audio.Length;
            return true;
        }
    }

    public float[] SnapshotAccumulated()
    {
        lock (_bufferLock)
        {
            if (Volatile.Read(ref _disposed) != 0) return Array.Empty<float>();
            var copy = new float[_accumulatedLength];
            _accumulated.AsSpan(0, _accumulatedLength).CopyTo(copy);
            return copy;
        }
    }

    public int AccumulatedLength
    {
        get { lock (_bufferLock) return Volatile.Read(ref _disposed) != 0 ? 0 : _accumulatedLength; }
    }

    public bool PublishResult(AsrResult result) => _resultChannel.Writer.TryWrite(result);

    internal void MarkFinishing()
    {
        Interlocked.CompareExchange(ref _state, (int)Qwen3SessionState.Finishing, (int)Qwen3SessionState.Active);
        _audioChannel.Writer.TryComplete();
    }

    internal void AbortWork()
    {
        MarkFinishing();
        _resultChannel.Writer.TryComplete();
        try { _workStop.Cancel(); } catch (ObjectDisposedException) { }
    }

    protected override void OnSessionFaulted(Exception error)
    {
        MarkFinishing();
        _resultChannel.Writer.TryComplete(error);
    }

    internal void MarkFinished()
    {
        Volatile.Write(ref _state, (int)Qwen3SessionState.Finished);
        _resultChannel.Writer.TryComplete();
    }

    internal void DrainAudio()
    {
        while (_audioChannel.Reader.TryRead(out var item)) item.Dispose();
    }

    private void OnAudioChunkDropped(PooledAudio dropped)
    {
        try { dropped.Dispose(); } catch { }
        try { OnFrameDropped(); } catch { }
    }

    private void OnFrameDropped()
    {
        long total = Interlocked.Increment(ref _droppedFrames);
        long now = Environment.TickCount64;
        long last = Volatile.Read(ref _lastDropLogTicks);
        if (now - last < 1000) return;
        if (Interlocked.CompareExchange(ref _lastDropLogTicks, now, last) != last) return;
        _logger.LogWarning("ASR session {SessionId} audio buffer full, frame dropped (total dropped: {Total}).", SessionId, total);
    }

    private void EnsureCapacity(int required)
    {
        if (_accumulated.Length >= required) return;
        int newSize = Math.Max(_accumulated.Length * 2, required);
        var bigger = ArrayPool<float>.Shared.Rent(newSize);
        _accumulated.AsSpan(0, _accumulatedLength).CopyTo(bigger);
        ArrayPool<float>.Shared.Return(_accumulated);
        _accumulated = bigger;
    }

    private void LogSummary(Qwen3AsrSessionMetrics m)
    {
        double avgRtf = m.TotalAudioSeconds > 0 ? m.TotalInferenceSeconds / m.TotalAudioSeconds : 0.0;
        double firstMs = m.FirstResultPublished ? m.FirstResultElapsedSeconds * 1000.0 : -1.0;
        _logger.LogInformation(
            "Qwen3-ASR 会话结束: session={SessionId} 总耗时={ElapsedMs:F0}ms 识别次数={Count} 累计音频={AudioSec:F2}s 累计推理={InfMs:F0}ms (encode={EncodeMs:F0}ms, prefill={PrefillMs:F0}ms, decode={DecodeMs:F0}ms) 平均RTF={Rtf:F2} 首次流式输出={FirstMs:F0}ms",
            SessionId, m.ElapsedSeconds * 1000.0, m.TranscribeCount, m.TotalAudioSeconds,
            m.TotalInferenceSeconds * 1000.0, m.TotalEncodeSeconds * 1000.0,
            m.TotalPrefillSeconds * 1000.0, m.TotalDecodeStepSeconds * 1000.0,
            avgRtf, firstMs);
    }
}
