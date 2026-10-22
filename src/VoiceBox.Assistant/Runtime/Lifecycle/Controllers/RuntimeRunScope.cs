using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Owns the run-scoped cancellation source plus the started/disposed latches of one Runtime
/// instance. It holds no conversation, module or effect state; collaborators consult it only to
/// decide whether runtime-owned work may still start or continue.
/// </summary>
internal sealed class RuntimeRunScope :
    IRuntimeLivenessView,
    IRuntimeCancellationView
{
    private CancellationTokenSource? _runCts;
    private int _started;
    private int _disposed;

    public CancellationTokenSource? Cts => _runCts;

    public CancellationToken Token => _runCts?.Token ?? default;

    CancellationToken IRuntimeCancellationView.RunToken => Token;

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public bool IsStarted => Volatile.Read(ref _started) == 1;

    public bool IsCancelled => _runCts?.IsCancellationRequested == true;

    /// <summary>Disposed or the current run has been cancelled.</summary>
    public bool IsStopping => IsDisposed || IsCancelled;

    /// <summary>The input gate may only reopen inside a live, non-cancelled run.</summary>
    public bool CanOpenInput => !IsCancelled && !IsDisposed && IsStarted;

    public CancellationTokenSource Begin()
    {
        var cts = new CancellationTokenSource();
        _runCts = cts;
        Volatile.Write(ref _started, 1);
        return cts;
    }

    public bool IsCurrentRun(CancellationTokenSource? candidate) => ReferenceEquals(_runCts, candidate);

    public void ClearIfCurrent(CancellationTokenSource? candidate)
    {
        if (ReferenceEquals(_runCts, candidate))
            _runCts = null;
    }

    /// <summary>Detaches the source so concurrent readers fall back to a default token.</summary>
    public CancellationTokenSource? Detach()
    {
        CancellationTokenSource? current = _runCts;
        _runCts = null;
        return current;
    }

    public void Cancel()
    {
        try { _runCts?.Cancel(); }
        catch { }
    }

    public void MarkStopped() => Volatile.Write(ref _started, 0);

    /// <summary>Returns false when another caller already latched disposal.</summary>
    public bool TryLatchDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            return false;

        Volatile.Write(ref _disposed, 1);
        return true;
    }

    public void ThrowIfDisposed(object owner)
        => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, owner);
}
