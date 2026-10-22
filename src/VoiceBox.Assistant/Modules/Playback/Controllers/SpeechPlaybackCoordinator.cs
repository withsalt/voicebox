using VoiceBox.Assistant.Api.Playback;
using VoiceBox.Assistant.Modules.Response.Models;
using System.Runtime.ExceptionServices;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Assistant.Modules.Playback.Contracts;

namespace VoiceBox.Assistant.Modules.Playback.Controllers;

/// <summary>
/// Serializes all logical text-to-speech requests before they reach the device PCM queue.
/// The coordinator deliberately owns no TTS or audio provider; <see cref="PlaybackModule"/>
/// supplies the request executor so the existing pump remains the single synthesis boundary.
/// </summary>
internal sealed class SpeechPlaybackCoordinator : IDisposable
{
    private const int QueueCapacity = 32;

    private readonly object _gate = new();
    private readonly LinkedList<SpeechPlaybackRequest> _requests = [];
    private readonly SemaphoreSlim _availableItems = new(0);
    private readonly SemaphoreSlim _availableSlots = new(QueueCapacity, QueueCapacity);
    private readonly ISpeechPlaybackRequestExecutor _executor;
    private readonly IPlaybackDeviceQueuePort _deviceQueue;
    private readonly HashSet<long> _admittedTurnIds = [];
    private long _nextRequestId;
    private CancellationTokenSource? _runCancellation;
    private Task? _worker;
    private SpeechPlaybackRequest? _active;
    private TaskCompletionSource? _activeStopCompletion;
    private long _lastStoppedOwnerTurnId;
    private bool _started;
    private bool _stopping;
    private bool _disposed;

    public SpeechPlaybackCoordinator(
        ISpeechPlaybackRequestExecutor executor,
        IPlaybackDeviceQueuePort deviceQueue)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _deviceQueue = deviceQueue ?? throw new ArgumentNullException(nameof(deviceQueue));
    }

    public Task Start(CancellationToken runCancellation)
    {
        runCancellation.ThrowIfCancellationRequested();
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_started)
                throw new InvalidOperationException("Speech playback coordinator has already started.");

            _started = true;
            _stopping = false;
            _runCancellation = CancellationTokenSource.CreateLinkedTokenSource(runCancellation);
            _worker = RunAsync(_runCancellation.Token);
            return _worker;
        }
    }

    public async ValueTask<SpeechEnqueueResult> EnqueueAsync(
        long? ownerTurnId,
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken cancellationToken,
        SpeechPlaybackQueuePosition queuePosition = SpeechPlaybackQueuePosition.Tail)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (queuePosition is not SpeechPlaybackQueuePosition.Head and
            not SpeechPlaybackQueuePosition.Tail)
        {
            throw new ArgumentOutOfRangeException(nameof(queuePosition), queuePosition, null);
        }

        SpeechPlaybackRequest request;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_started || _stopping || _runCancellation is null)
                throw new InvalidOperationException("Speech playback coordinator is not running.");

            if (ownerTurnId is > 0 && !_admittedTurnIds.Add(ownerTurnId.Value))
            {
                throw new InvalidOperationException(
                    $"Playback operation for turn '{ownerTurnId.Value}' is already queued or active.");
            }

            request = new SpeechPlaybackRequest(
                Interlocked.Increment(ref _nextRequestId),
                ownerTurnId,
                text,
                cancellationToken,
                _runCancellation.Token);
        }

        bool slotReserved = false;
        try
        {
            await _availableSlots.WaitAsync(request.Cancellation.Token).ConfigureAwait(false);
            slotReserved = true;

            lock (_gate)
            {
                if (!_started || _stopping || _runCancellation is null)
                    throw new InvalidOperationException("Speech playback coordinator is not running.");

                if (queuePosition == SpeechPlaybackQueuePosition.Head)
                    _requests.AddFirst(request);
                else
                    _requests.AddLast(request);
            }

            _availableItems.Release();
            slotReserved = false;
            return new SpeechEnqueueResult(request.RequestId, request.CompletionSource.Task);
        }
        catch
        {
            if (slotReserved)
                _availableSlots.Release();
            RemoveAdmittedTurn(request);
            request.TrySetCanceled();
            request.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Cancels and removes device audio for the active request without touching pending requests.
    /// An explicit completion barrier prevents the worker from starting the next request until
    /// the previous request's device audio has been removed. Cancellation callbacks and the
    /// device provider both run after the coordinator gate has been released.
    /// </summary>
    public bool TryStopActivePlayback()
        => TryStopActivePlayback(expectedOwnerTurnId: null) is PlaybackStopOutcome.Stopped or PlaybackStopOutcome.AlreadyStopped;

    /// <summary>The turn owning the active request, or null when none (or the active request is host speech).</summary>
    public long? ActiveOwnerTurnId
    {
        get
        {
            lock (_gate)
                return _active?.OwnerTurnId;
        }
    }

    public bool TryDuckActivePlayback(long expectedOwnerTurnId, float gain, float originalGain)
    {
        lock (_gate)
        {
            if (expectedOwnerTurnId <= 0 || _stopping || _active is not { StopIssued: false } active ||
                active.OwnerTurnId != expectedOwnerTurnId || active.Cancellation.IsCancellationRequested)
                return false;
            if (active.RestoreGain.HasValue)
                return true;
            if (!float.IsFinite(gain) || !float.IsFinite(originalGain) ||
                gain < 0 || originalGain > 1 || gain >= originalGain)
                return false;

            // Gain setters only update the device's target. Serialize that update with owner changes
            // so a late candidate cannot change the next reply's volume.
            active.RestoreGain = originalGain;
            _deviceQueue.SetGain(gain);
            return true;
        }
    }

    public bool TryRestoreActivePlayback(long expectedOwnerTurnId)
    {
        lock (_gate)
        {
            if (expectedOwnerTurnId <= 0 || _active is not { StopIssued: false } active ||
                active.OwnerTurnId != expectedOwnerTurnId || active.Cancellation.IsCancellationRequested ||
                active.RestoreGain is not { } gain)
                return false;

            _deviceQueue.SetGain(gain);
            active.RestoreGain = null;
            return true;
        }
    }

    /// <summary>
    /// Owner-scoped stop: the owner check and the stop reservation happen under the same gate, so a
    /// newer request that became active after the decision was taken can never be cleared by it,
    /// and a request that was already stopped is never cleared twice. A null owner is the
    /// host-speech control path and stops whatever is active.
    /// </summary>
    public PlaybackStopOutcome TryStopActivePlayback(long? expectedOwnerTurnId)
    {
        SpeechPlaybackRequest active;
        TaskCompletionSource completion;
        lock (_gate)
        {
            active = _active!;
            if (active is null || (expectedOwnerTurnId is { } owner && active.OwnerTurnId != owner))
            {
                // The owner's request may already have finished stopped (owner stop or cancellation
                // clear); a repeated stop for that owner is then a no-op, not a foreign owner.
                return expectedOwnerTurnId is { } stoppedOwner && stoppedOwner == _lastStoppedOwnerTurnId
                    ? PlaybackStopOutcome.AlreadyStopped
                    : PlaybackStopOutcome.NotOwned;
            }
            if (active.StopIssued)
                return PlaybackStopOutcome.AlreadyStopped;

            active.StopIssued = true;
            completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            _activeStopCompletion = completion;
        }

        Exception? cancellationFailure = null;
        try { active.Cancellation.Cancel(); }
        catch (ObjectDisposedException) { }
        catch (Exception exception) { cancellationFailure = exception; }

        // Provider cancellation callbacks may throw; removing the old PCM is still required.
        // Keep the barrier until ClearActive consumes it, even if this stop finishes first.
        try { _deviceQueue.ClearQueue(); }
        catch (Exception exception)
        {
            Exception failure = cancellationFailure is null
                ? exception
                : new AggregateException(cancellationFailure, exception);
            completion.TrySetException(failure);
            ExceptionDispatchInfo.Throw(failure);
        }

        completion.TrySetResult();
        if (cancellationFailure is not null)
            ExceptionDispatchInfo.Throw(cancellationFailure);
        return PlaybackStopOutcome.Stopped;
    }

    public void CancelActivePlayback()
    {
        SpeechPlaybackRequest? active;
        lock (_gate)
            active = _active;

        if (active is not null)
        {
            try { active.Cancellation.Cancel(); }
            catch (ObjectDisposedException) { }
        }
    }

    public async Task StopAsync()
    {
        Task? worker;
        CancellationTokenSource? runToCancel = null;
        lock (_gate)
        {
            if (!_started || _stopping)
            {
                worker = _worker;
            }
            else
            {
                _stopping = true;
                runToCancel = _runCancellation;
                worker = _worker;
            }
        }

        Exception? cancellationFailure = null;
        // Cancellation invokes arbitrary provider callbacks synchronously. Run them outside
        // the gate, and always join/clean up the worker even when a callback throws.
        try { runToCancel?.Cancel(); }
        catch (ObjectDisposedException) { }
        catch (Exception exception) { cancellationFailure = exception; }
        try { CancelActivePlayback(); }
        catch (Exception exception)
        {
            cancellationFailure = cancellationFailure is null
                ? exception
                : new AggregateException(cancellationFailure, exception);
        }

        try
        {
            if (worker is not null)
            {
                try { await worker.ConfigureAwait(false); }
                catch (OperationCanceledException) { }
            }
        }
        catch (Exception exception) when (cancellationFailure is not null)
        {
            throw new AggregateException(cancellationFailure, exception);
        }
        finally
        {
            CancelPendingRequests();
            lock (_gate)
            {
                _started = false;
                _active = null;
                _worker = null;
                _runCancellation?.Dispose();
                _runCancellation = null;
            }
        }

        if (cancellationFailure is not null)
            ExceptionDispatchInfo.Throw(cancellationFailure);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, true) != false)
            return;

        try { StopAsync().GetAwaiter().GetResult(); }
        catch { }
        _availableItems.Dispose();
        _availableSlots.Dispose();
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                await _availableItems.WaitAsync(cancellationToken).ConfigureAwait(false);

                SpeechPlaybackRequest request;
                lock (_gate)
                {
                    LinkedListNode<SpeechPlaybackRequest>? first = _requests.First;
                    if (first is null)
                        continue;
                    request = first.Value;
                    _requests.RemoveFirst();
                    _active = request;
                }
                _availableSlots.Release();

                string? result = null;
                Exception? failure = null;
                bool cancelled = false;
                try
                {
                    if (request.Cancellation.IsCancellationRequested)
                    {
                        cancelled = true;
                    }
                    else
                    {
                        result = await _executor.ExecuteAsync(request, request.Cancellation.Token)
                            .ConfigureAwait(false);
                        request.Cancellation.Token.ThrowIfCancellationRequested();
                    }
                }
                catch (OperationCanceledException) when (request.Cancellation.IsCancellationRequested)
                {
                    cancelled = true;
                }
                catch (Exception exception)
                {
                    failure = exception;
                }
                finally
                {
                    try
                    {
                        try
                        {
                            try
                            {
                                await ClearActive(request, cancelled).ConfigureAwait(false);
                            }
                            finally
                            {
                                // Clear/stop must finish before restoring gain; the next request cannot
                                // start until this cleanup completes, including on shutdown or failure.
                                if (request.RestoreGain is { } gain)
                                {
                                    _deviceQueue.SetGain(gain);
                                    request.RestoreGain = null;
                                }
                            }
                        }
                        finally
                        {
                            // Release admission before publishing Completion, including when
                            // device cleanup fails, and always detach the linked cancellation.
                            RemoveAdmittedTurn(request);
                            request.Dispose();
                        }
                    }
                    catch (Exception exception)
                    {
                        request.TrySetException(exception);
                        throw;
                    }
                }

                if (cancelled)
                    request.TrySetCanceled();
                else if (failure is not null)
                    request.TrySetException(failure);
                else
                    request.TrySetResult(result ?? string.Empty);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        finally
        {
            CancellationTokenSource? run;
            lock (_gate)
            {
                _stopping = true;
                run = _runCancellation;
            }
            // A terminated worker cannot accept more requests. Wake slot waiters as well as
            // cancelling queued requests; cancellation callbacks must not replace its fault.
            try { run?.Cancel(); } catch { }
            CancelPendingRequests();
        }
    }

    private Task ClearActive(SpeechPlaybackRequest request, bool cancelled)
    {
        bool clearDevice;
        Task barrier;
        lock (_gate)
        {
            if (ReferenceEquals(_active, request))
                _active = null;
            barrier = _activeStopCompletion?.Task ?? Task.CompletedTask;
            _activeStopCompletion = null;
            // A request cancelled through its own token (the Kernel's CancelResponse for this owner,
            // or shutdown) may leave already-queued device audio behind. Remove it here, exactly
            // once, unless the owner-scoped stop already did; it still belongs to this request
            // because no later request has been started yet.
            clearDevice = cancelled && !request.StopIssued;
            if (clearDevice)
                request.StopIssued = true;
            if (request.StopIssued && request.OwnerTurnId is > 0)
                _lastStoppedOwnerTurnId = request.OwnerTurnId.Value;
        }

        if (clearDevice)
            _deviceQueue.ClearQueue();
        return barrier;
    }

    private void RemoveAdmittedTurn(SpeechPlaybackRequest request)
    {
        if (request.OwnerTurnId is not > 0)
            return;

        lock (_gate)
            _admittedTurnIds.Remove(request.OwnerTurnId.Value);
    }

    private void CancelPendingRequests()
    {
        SpeechPlaybackRequest[] pending;
        lock (_gate)
        {
            pending = _requests.ToArray();
            _requests.Clear();
            for (int i = 0; i < pending.Length; i++)
                _availableItems.Wait(0);
        }

        foreach (SpeechPlaybackRequest request in pending)
        {
            _availableSlots.Release();
            RemoveAdmittedTurn(request);
            request.TrySetCanceled();
            request.Dispose();
        }
    }
}
