using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Core.Services.ASRService;

/// <summary>Common service facade for all ASR providers.</summary>
public abstract class BaseASRService : IASRService, IAsyncDisposable
{
    private readonly AsrSessionManager _sessions;
    private readonly object _disposeSync = new();
    private Task? _disposeTask;

    protected BaseASRService(int maxConcurrentSessions)
        => _sessions = new AsrSessionManager(maxConcurrentSessions);

    public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
    {
        EnsureServiceActive();
        return _sessions.StartAsync(CreateSession, ct);
    }

    public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
    {
        if (string.IsNullOrWhiteSpace(sessionId) || audioData.IsEmpty)
            return;
        if (_sessions.TryGet(sessionId, out var session))
            session.FeedAudio(audioData);
    }

    public Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
    {
        EnsureServiceActive();
        return _sessions.FinishAsync(sessionId, ct);
    }

    public Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
    {
        EnsureServiceActive();
        return _sessions.CancelAsync(sessionId, ct);
    }

    protected abstract AsrSession CreateSession(string sessionId);

    protected bool TryGetSession<TSession>(string sessionId, out TSession session)
        where TSession : AsrSession
    {
        if (_sessions.TryGet(sessionId, out var candidate) && candidate is TSession typed)
        {
            session = typed;
            return true;
        }

        _sessions.ThrowIfFailed(sessionId);
        session = null!;
        return false;
    }

    protected void EnsureServiceActive()
    {
        lock (_disposeSync)
            ObjectDisposedException.ThrowIf(_disposeTask is not null, this);
        _sessions.EnsureActive();
    }

    /// <summary>Stops provider-wide background work before sessions are cancelled.</summary>
    protected virtual void OnDisposeRequested() { }

    /// <summary>Releases provider-wide resources after all sessions are gone.</summary>
    protected virtual void DisposeProvider() { }

    public ValueTask DisposeAsync()
    {
        lock (_disposeSync)
        {
            _disposeTask ??= Task.Run(DisposeCoreAsync);
            return new ValueTask(_disposeTask);
        }
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    private async Task DisposeCoreAsync()
    {
        OnDisposeRequested();
        await _sessions.DisposeAsync().ConfigureAwait(false);
        DisposeProvider();
    }
}
