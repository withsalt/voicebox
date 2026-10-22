using System.Runtime.ExceptionServices;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Core.Services.ASRService;

/// <summary>
/// Indexes live sessions and enforces the service-wide concurrency limit.
/// Session lifecycle and provider resource ownership remain in <see cref="AsrSession"/>.
/// </summary>
internal sealed class AsrSessionManager
{
    private readonly object _sync = new();
    private readonly Dictionary<string, AsrSession> _sessions = new(StringComparer.Ordinal);
    private readonly Dictionary<string, StartEntry> _starting = new(StringComparer.Ordinal);
    private readonly Dictionary<string, FailureEntry> _failures = new(StringComparer.Ordinal);
    private readonly Queue<string> _failureOrder = new();
    private static readonly TimeSpan FailureRetention = TimeSpan.FromMinutes(5);
    private const int MaxRetainedFailures = 128;
    private readonly TimeProvider _timeProvider;
    private readonly int _capacity;
    private bool _disposed;
    private Task? _disposeTask;

    public AsrSessionManager(int capacity, TimeProvider? timeProvider = null)
    {
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _timeProvider = timeProvider ?? TimeProvider.System;
    }

    public async Task<SessionInfo> StartAsync(
        Func<string, AsrSession> createSession,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(createSession);
        ct.ThrowIfCancellationRequested();

        StartEntry entry;
        lock (_sync)
        {
            EnsureActiveNoLock();
            PruneReleasedNoLock();
            if (_sessions.Count + _starting.Count >= _capacity)
                throw new InvalidOperationException($"Reached the maximum concurrent session limit: {_capacity}.");

            string id = Guid.NewGuid().ToString("N");
            entry = new StartEntry(id);
            _starting.Add(id, entry);
        }

        AsrSession? session = null;
        try
        {
            session = createSession(entry.Id);
            bool stopForDisposal;
            lock (_sync)
                stopForDisposal = _disposed;
            if (stopForDisposal)
            {
                try { entry.Stop.Cancel(); }
                catch (ObjectDisposedException) { }
            }

            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, entry.Stop.Token);
            await session.StartAsync(linked.Token).ConfigureAwait(false);

            bool accepted;
            lock (_sync)
            {
                accepted = !_disposed && !ct.IsCancellationRequested;
                if (accepted)
                {
                    _starting.Remove(entry.Id);
                    _sessions.Add(entry.Id, session);
                    _ = WatchReleaseAsync(entry.Id, session);
                }
            }

            if (!accepted)
                throw new OperationCanceledException(_disposed ? entry.Stop.Token : ct);

            return new SessionInfo(entry.Id);
        }
        catch
        {
            if (session is not null)
            {
                try { await session.CancelAsync().ConfigureAwait(false); }
                catch { }
            }
            throw;
        }
        finally
        {
            entry.Stop.Dispose();

            lock (_sync)
            {
                _starting.Remove(entry.Id);
                entry.Completed.TrySetResult();
            }
        }
    }

    public bool TryGet(string sessionId, out AsrSession session)
    {
        lock (_sync)
        {
            if (!_disposed && _sessions.TryGetValue(sessionId, out session!))
                return true;
            session = null!;
            return false;
        }
    }

    public void EnsureActive()
    {
        lock (_sync) EnsureActiveNoLock();
    }

    public Task<AsrResult?> FinishAsync(string sessionId, CancellationToken ct)
        => Find(sessionId).FinishAsync(ct);

    public void ThrowIfFailed(string sessionId)
    {
        lock (_sync)
        {
            EnsureActiveNoLock();
            PruneReleasedNoLock();
            if (_failures.TryGetValue(sessionId, out var failure))
                failure.Error.Throw();
        }
    }

    public Task CancelAsync(string sessionId, CancellationToken ct)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);

        AsrSession? session;
        lock (_sync)
        {
            EnsureActiveNoLock();
            PruneReleasedNoLock();
            _sessions.TryGetValue(sessionId, out session);
        }

        // Cancellation is intentionally idempotent: callers often race with
        // automatic release and should not need a second existence check.
        return (session?.CancelAsync() ?? Task.CompletedTask).WaitAsync(ct);
    }

    public async Task DisposeAsync()
    {
        Task disposal;
        lock (_sync)
        {
            _disposeTask ??= Task.Run(DisposeCoreAsync);
            disposal = _disposeTask;
        }
        await disposal.ConfigureAwait(false);
    }

    private async Task DisposeCoreAsync()
    {
        StartEntry[] starts;
        AsrSession[] sessions;
        lock (_sync)
        {
            _disposed = true;
            starts = _starting.Values.ToArray();
            sessions = _sessions.Values.ToArray();
        }

        foreach (var start in starts)
        {
            try { start.Stop.Cancel(); }
            catch (ObjectDisposedException) { }
        }

        Task[] startTasks = starts.Select(start => start.Completed.Task).ToArray();
        Task[] sessionTasks = sessions.Select(CancelSafelyAsync).ToArray();
        await Task.WhenAll(startTasks.Concat(sessionTasks)).ConfigureAwait(false);
        lock (_sync)
        {
            _failures.Clear();
            _failureOrder.Clear();
        }
    }

    private async Task WatchReleaseAsync(string id, AsrSession session)
    {
        try { await session.ReleaseTask.ConfigureAwait(false); }
        catch { }
        lock (_sync) RemoveReleasedNoLock(id, session);
    }

    private AsrSession Find(string sessionId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        lock (_sync)
        {
            EnsureActiveNoLock();
            PruneReleasedNoLock();
            if (_sessions.TryGetValue(sessionId, out var session))
                return session;
            if (_failures.TryGetValue(sessionId, out var failure))
                failure.Error.Throw();
        }
        throw new InvalidOperationException($"Session not found or already finished: {sessionId}");
    }

    private void PruneReleasedNoLock()
    {
        foreach (var pair in _sessions.ToArray())
        {
            if (pair.Value.ReleaseTask.IsCompleted)
                RemoveReleasedNoLock(pair.Key, pair.Value);
        }
        PruneFailuresNoLock();
    }

    private void RemoveReleasedNoLock(string id, AsrSession session)
    {
        if (!_sessions.Remove(id) || _disposed || session.FailureInfo is not { } error)
            return;

        // Retain only the error, never the provider/session or its audio buffers.
        _failures.Add(id, new FailureEntry(error, _timeProvider.GetUtcNow() + FailureRetention));
        _failureOrder.Enqueue(id);
        PruneFailuresNoLock();
    }

    private void PruneFailuresNoLock()
    {
        DateTimeOffset now = _timeProvider.GetUtcNow();
        while (_failureOrder.TryPeek(out string? id) &&
               (_failures.Count > MaxRetainedFailures || _failures[id].ExpiresAt <= now))
        {
            _failureOrder.Dequeue();
            _failures.Remove(id);
        }
    }

    private static async Task CancelSafelyAsync(AsrSession session)
    {
        try { await session.CancelAsync().ConfigureAwait(false); }
        catch { }
    }

    private void EnsureActiveNoLock()
        => ObjectDisposedException.ThrowIf(_disposed, this);

    private sealed record FailureEntry(ExceptionDispatchInfo Error, DateTimeOffset ExpiresAt);

    private sealed class StartEntry(string id)
    {
        public string Id { get; } = id;
        public CancellationTokenSource Stop { get; } = new();
        public TaskCompletionSource Completed { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
