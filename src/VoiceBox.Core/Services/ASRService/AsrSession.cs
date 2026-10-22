using System.Runtime.ExceptionServices;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Core.Services.ASRService;

/// <summary>
/// Owns the complete lifecycle of one ASR session.
/// </summary>
/// <remarks>
/// A provider supplies the hooks for its transport/model. The base class
/// serializes terminal operations, separates waiter cancellation from provider
/// cancellation, retains the first fatal error, and releases provider resources once.
/// </remarks>
public abstract class AsrSession : IDisposable, IAsyncDisposable
{
    private readonly object _sync = new();
    private readonly CancellationTokenSource _cancellation = new();
    private readonly CancellationToken _cancellationToken;
    private readonly TaskCompletionSource _startCompleted =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _finishCompleted =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _released =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private LifecycleState _state = LifecycleState.Created;
    private Task<AsrResult?>? _finishTask;
    private Task? _cancelTask;
    private Task? _releaseTask;
    private ExceptionDispatchInfo? _failure;
    private bool _cancellationRequested;
    private bool _finishCommitted;

    protected AsrSession(string sessionId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        SessionId = sessionId;
        CreatedAt = DateTime.UtcNow;
        _cancellationToken = _cancellation.Token;
    }

    public string SessionId { get; }

    public DateTime CreatedAt { get; }

    public CancellationToken CancellationToken => _cancellationToken;

    /// <summary>The first fatal error, retained even after provider resources are released.</summary>
    public Exception? FailureException => FailureInfo?.SourceException;

    internal ExceptionDispatchInfo? FailureInfo
    {
        get { lock (_sync) return _failure; }
    }

    internal Task ReleaseTask => _released.Task;

    /// <summary>
    /// Confucius4 keeps a completed decode alive when its caller cancels the wait,
    /// allowing a later Finish call to collect the result.
    /// </summary>
    protected virtual bool RetainAfterFinishWaitCancellation => false;

    internal async Task StartAsync(CancellationToken ct)
    {
        lock (_sync)
        {
            if (_state != LifecycleState.Created)
                throw new InvalidOperationException($"Session {SessionId} has already started.");
            _state = LifecycleState.Starting;
        }

        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, CancellationToken);
        try
        {
            await StartSessionAsync(linked.Token).ConfigureAwait(false);
            lock (_sync)
            {
                _failure?.Throw();
                if (_cancellationRequested)
                    throw new OperationCanceledException(CancellationToken);
                _state = LifecycleState.Active;
            }
        }
        catch (Exception ex)
        {
            if (ex is not OperationCanceledException || !linked.IsCancellationRequested)
                _ = FailAsync(ex);
            ThrowIfFailed();
            throw;
        }
        finally { _startCompleted.TrySetResult(); }
    }

    internal Task<AsrResult?> FinishAsync(CancellationToken ct)
    {
        Task<AsrResult?> terminal;
        bool requestCancellation = false;
        lock (_sync)
        {
            if (_failure is not null)
                _finishTask ??= Task.FromException<AsrResult?>(_failure.SourceException);
            else if (_state == LifecycleState.Released)
                throw new InvalidOperationException($"Session {SessionId} is no longer active.");

            if (_finishTask is null)
            {
                _state = LifecycleState.Finishing;
                if (_cancellationRequested || (!RetainAfterFinishWaitCancellation && ct.IsCancellationRequested))
                {
                    _finishTask = Task.FromCanceled<AsrResult?>(new CancellationToken(true));
                    _finishCompleted.TrySetResult();
                    requestCancellation = !_cancellationRequested;
                }
                else
                    _finishTask = Task.Factory.StartNew(
                        static state => ((AsrSession)state!).RunFinishAsync(),
                        this,
                        CancellationToken.None,
                        TaskCreationOptions.DenyChildAttach,
                        TaskScheduler.Default).Unwrap();
            }

            terminal = _finishTask;
        }

        if (requestCancellation)
            CancelAsync();

        return AwaitFinishAsync(terminal, ct);
    }

    private async Task<AsrResult?> AwaitFinishAsync(Task<AsrResult?> terminal, CancellationToken ct)
    {
        using var registration = RetainAfterFinishWaitCancellation
            ? default
            : ct.Register(static state => ((AsrSession)state!).CancelAsync(), this);

        try
        {
            AsrResult? result = await terminal.WaitAsync(ct).ConfigureAwait(false);
            await CompleteCleanupAsync().ConfigureAwait(false);
            return result;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            if (!RetainAfterFinishWaitCancellation)
                _ = CancelAsync();
            throw;
        }
        catch
        {
            try { await CompleteCleanupAsync().ConfigureAwait(false); }
            catch when (FailureException is not null) { }
            ThrowIfFailed();
            throw;
        }
    }

    /// <summary>
    /// Records the first fatal error and starts automatic cleanup. Background workers
    /// must not await this task from inside work that ReleaseSessionAsync waits for.
    /// The task confirms cleanup; Finish and result streams carry the original error.
    /// </summary>
    public Task FailAsync(Exception error)
    {
        ArgumentNullException.ThrowIfNull(error);
        lock (_sync)
        {
            if (_failure is not null)
                return BeginTerminationNoLock();
            if (_cancellationRequested || _finishCommitted || _state == LifecycleState.Released)
                return _cancelTask ?? _releaseTask ?? Task.CompletedTask;

            if (_state == LifecycleState.Created)
                _startCompleted.TrySetResult();
            _failure = ExceptionDispatchInfo.Capture(error);
            _state = LifecycleState.Faulted;
            _cancellationRequested = true;
            if (_finishTask is null)
                _finishCompleted.TrySetResult();
            return BeginTerminationNoLock();
        }
    }

    private Task BeginTerminationNoLock()
    {
        if (_cancelTask is not null)
            return _cancelTask;
        if (_releaseTask is not null)
            return _releaseTask;

        ExceptionDispatchInfo? failure = _failure;
        _cancelTask = Task.Run(async () =>
        {
            // Publish the error to provider result channels before cancelling work;
            // otherwise an abort could close them successfully or mask it as cancellation.
            Exception? cleanupError = null;
            if (failure is not null)
            {
                try { OnSessionFaulted(failure.SourceException); }
                catch (Exception ex) { cleanupError = ex; }
            }

            try { _cancellation.Cancel(); }
            catch (Exception ex) { cleanupError ??= ex; }

            try { await CancelSessionAsync().ConfigureAwait(false); }
            catch (Exception ex) { cleanupError ??= ex; }

            try { await ReleaseAsync().ConfigureAwait(false); }
            catch (Exception ex) { cleanupError ??= ex; }

            if (cleanupError is not null)
                ExceptionDispatchInfo.Capture(cleanupError).Throw();
        });
        ObserveFault(_cancelTask);
        return _cancelTask;
    }

    private Task CompleteCleanupAsync()
    {
        // Finish must not bypass fault publication or provider cancellation by
        // starting release independently of the already scheduled termination.
        lock (_sync)
            return _cancelTask ?? ReleaseAsync();
    }

    private static void ObserveFault(Task task)
    {
        _ = task.ContinueWith(static task => _ = task.Exception,
            CancellationToken.None,
            TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }

    internal Task CancelAsync()
    {
        lock (_sync)
        {
            if (_cancellationRequested || _finishCommitted || _state == LifecycleState.Released)
                return BeginTerminationNoLock();

            // Publish cancellation before scheduling callbacks and cleanup so
            // a concurrent Finish cannot commit a successful result afterwards.
            _cancellationRequested = true;
            if (_state == LifecycleState.Created)
                _startCompleted.TrySetResult();
            if (_state != LifecycleState.Released)
                _state = LifecycleState.Cancelling;
            if (_finishTask is null)
                _finishCompleted.TrySetResult();
            return BeginTerminationNoLock();
        }
    }

    internal Task ReleaseAsync()
    {
        lock (_sync)
        {
            if (_releaseTask is not null)
                return _releaseTask;

            _releaseTask = Task.Run(async () =>
            {
                // A worker can fail before its provider Start hook has returned.
                // Do not release resources while that hook is still publishing tasks.
                await _startCompleted.Task.ConfigureAwait(false);
                await _finishCompleted.Task.ConfigureAwait(false);
                try
                {
                    await ReleaseSessionAsync().ConfigureAwait(false);
                }
                finally
                {
                    lock (_sync) _state = LifecycleState.Released;
                    _cancellation.Dispose();
                    _released.TrySetResult();
                }
            });
            ObserveFault(_releaseTask);
            return _releaseTask;
        }
    }

    private async Task<AsrResult?> RunFinishAsync()
    {
        try
        {
            ThrowIfFailed();
            CancellationToken.ThrowIfCancellationRequested();
            AsrResult? result = await FinishSessionAsync().ConfigureAwait(false);
            lock (_sync)
            {
                _failure?.Throw();
                if (_cancellationRequested)
                    throw new OperationCanceledException(CancellationToken);
                _finishCommitted = true;
            }
            return result;
        }
        catch (Exception ex)
        {
            if (ex is not OperationCanceledException || !CancellationToken.IsCancellationRequested)
                _ = FailAsync(ex);
            ThrowIfFailed();
            throw;
        }
        finally
        {
            _finishCompleted.TrySetResult();
        }
    }

    private void ThrowIfFailed() => FailureInfo?.Throw();

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public ValueTask DisposeAsync() => new(CancelAsync());

    protected abstract Task StartSessionAsync(CancellationToken ct);

    public abstract void FeedAudio(ReadOnlySpan<float> audioData);

    protected abstract Task<AsrResult?> FinishSessionAsync();

    protected abstract Task CancelSessionAsync();

    /// <summary>Stops audio admission and completes provider result channels with the fatal error.</summary>
    protected virtual void OnSessionFaulted(Exception error) { }

    protected abstract Task ReleaseSessionAsync();
}
