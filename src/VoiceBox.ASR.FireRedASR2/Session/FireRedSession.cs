using System.Buffers;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.FireRedASR2.Internal;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.ASR.FireRedASR2.Pump;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.FireRedASR2.Session;

internal sealed class FireRedSession : AsrSession
{
    private readonly Channel<PooledAudio> _audioChannel;
    private readonly CancellationTokenSource _workStop = new();
    private readonly ILogger _logger;
    private readonly OfflineStream _stream;
    private readonly OfflineRecognizer _recognizer;
    private readonly SemaphoreSlim _recognitionGate;
    private readonly FireRedFeaturePump _pump;
    private readonly double _finishDrainTimeoutSec;
    private long _droppedFrames;
    private long _lastDropLogTicks;
    private int _state = (int)FireRedSessionState.Active;

    public FireRedSessionState State => (FireRedSessionState)Volatile.Read(ref _state);
    public long DroppedFrameCount => Volatile.Read(ref _droppedFrames);
    internal OfflineStream Stream => _stream;
    internal CancellationToken WorkCancellationToken => _workStop.Token;

    public FireRedSession(
        string sessionId,
        OfflineStream stream,
        int maxBufferChunks,
        OfflineRecognizer recognizer,
        SemaphoreSlim recognitionGate,
        double finishDrainTimeoutSec,
        ILogger? logger = null)
        : base(sessionId)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(recognizer);
        ArgumentNullException.ThrowIfNull(recognitionGate);
        _stream = stream;
        _recognizer = recognizer;
        _recognitionGate = recognitionGate;
        _finishDrainTimeoutSec = finishDrainTimeoutSec;
        _logger = logger ?? NullLogger.Instance;

        _audioChannel = Channel.CreateBounded<PooledAudio>(
            new BoundedChannelOptions(maxBufferChunks)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = false,
            },
            itemDropped: OnAudioChunkDropped);
        _pump = new FireRedFeaturePump(this, _logger);
    }

    protected override Task StartSessionAsync(CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _pump.Start();
        return Task.CompletedTask;
    }

    public override void FeedAudio(ReadOnlySpan<float> audioData)
    {
        if (State != FireRedSessionState.Active || FailureException is not null || audioData.IsEmpty) return;
        var rented = ArrayPool<float>.Shared.Rent(audioData.Length);
        audioData.CopyTo(rented);
        var pooled = new PooledAudio(rented, audioData.Length);
        if (_audioChannel.Writer.TryWrite(pooled)) return;
        pooled.Dispose();
    }

    protected override async Task<AsrResult?> FinishSessionAsync()
    {
        using var drainCts = CancellationTokenSource.CreateLinkedTokenSource(CancellationToken);
        if (_finishDrainTimeoutSec > 0)
            drainCts.CancelAfter(TimeSpan.FromSeconds(_finishDrainTimeoutSec));
        await _pump.StopAsync(drainCts.Token).ConfigureAwait(false);
        CancellationToken.ThrowIfCancellationRequested();

        _recognitionGate.Wait(CancellationToken);
        try
        {
            OfflineRecognizerResultEntity result = _recognizer.GetResult(_stream, CancellationToken);
            CancellationToken.ThrowIfCancellationRequested();
            return new AsrResult { Text = result.Text ?? string.Empty };
        }
        finally
        {
            _recognitionGate.Release();
        }
    }

    protected override Task CancelSessionAsync()
    {
        AbortWork();
        return Task.CompletedTask;
    }

    protected override async Task ReleaseSessionAsync()
    {
        AbortWork();
        try { await _pump.RunTask.ConfigureAwait(false); }
        catch { /* A faulted pump must not prevent provider resource cleanup. */ }
        try { DrainAndDispose(); }
        finally
        {
            try { _stream.Dispose(); }
            finally
            {
                _workStop.Dispose();
                Volatile.Write(ref _state, (int)FireRedSessionState.Finished);
            }
        }
    }

    protected override void OnSessionFaulted(Exception error) => CompleteWriter();

    internal bool TryReadAudio(out PooledAudio audio) => _audioChannel.Reader.TryRead(out audio);

    internal ValueTask<bool> WaitToReadAsync(CancellationToken ct = default)
        => _audioChannel.Reader.WaitToReadAsync(ct);

    internal void CompleteWriter()
    {
        Interlocked.CompareExchange(ref _state, (int)FireRedSessionState.Finishing,
            (int)FireRedSessionState.Active);
        _audioChannel.Writer.TryComplete();
    }

    internal void AbortWork()
    {
        CompleteWriter();
        try { _workStop.Cancel(); } catch (ObjectDisposedException) { }
    }

    internal void DrainAndDispose()
    {
        while (_audioChannel.Reader.TryRead(out var item)) item.Dispose();
    }

    private void OnAudioChunkDropped(PooledAudio dropped)
    {
        try { dropped.Dispose(); } catch { }
        try
        {
            long total = Interlocked.Increment(ref _droppedFrames);
            long now = Environment.TickCount64;
            long last = Volatile.Read(ref _lastDropLogTicks);
            if (now - last >= 1000 && Interlocked.CompareExchange(ref _lastDropLogTicks, now, last) == last)
                _logger.LogWarning("ASR session {SessionId} audio buffer full, frame dropped (total dropped: {Total}).", SessionId, total);
        }
        catch { }
    }
}
