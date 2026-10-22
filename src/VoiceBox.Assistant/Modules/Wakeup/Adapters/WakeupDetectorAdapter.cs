using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Adapters;

/// <summary>
/// Wake detector service adapter. Lifecycle intent is recorded under a small state lock, while
/// every provider operation runs outside that lock and is serialized by a provider gate. A stale
/// transition is discarded when a newer lifecycle request wins a race with it.
/// </summary>
internal sealed class WakeupDetectorAdapter
{
    private readonly IWakeupService _wakeup;
    private readonly IWakeupDetectorActivationView _activation;
    private readonly object _stateLock = new();
    private readonly SemaphoreSlim _providerGate = new(1, 1);

    private long _transitionVersion;
    private bool _requestedActive;
    private bool _disposed;

    // Provider callbacks are synchronous for the current service contract. Allow a callback to
    // request a nested Stop/Start without waiting on the semaphore held by its own provider call.
    private int _providerThreadId;

    public WakeupDetectorAdapter(IWakeupService wakeup, IWakeupDetectorActivationView activation)
    {
        _wakeup = wakeup ?? throw new ArgumentNullException(nameof(wakeup));
        _activation = activation ?? throw new ArgumentNullException(nameof(activation));
    }

    public void Subscribe(EventHandler<WakeupEventArgs> handler) => _wakeup.OnWakeup += handler;

    public void Unsubscribe(EventHandler<WakeupEventArgs> handler) => _wakeup.OnWakeup -= handler;

    /// <summary>Unconditional start used by the gate's own <c>Start</c> transition.</summary>
    public void StartUnconditional() => RunTransition(active: true, conditional: false);

    /// <summary>Unconditional stop used by the gate's own <c>Stop</c> transition.</summary>
    public void StopUnconditional() => RunTransition(active: false, conditional: false);

    public void FeedIfActive(ReadOnlySpan<float> audio)
    {
        bool enteredGate = EnterProviderGate();
        try
        {
            lock (_stateLock)
            {
                if (_disposed || !_requestedActive || !_activation.IsActive)
                    return;
            }

            InvokeProviderFeed(audio);
        }
        finally
        {
            ExitProviderGate(enteredGate);
        }
    }

    public void StartIfActive() => RunTransition(active: true, conditional: true);

    public void StopIfActive() => RunTransition(active: false, conditional: true);

    public void RestartIfActive()
    {
        long version;
        lock (_stateLock)
        {
            if (_disposed || !_activation.IsActive)
                return;

            _requestedActive = true;
            version = ++_transitionVersion;
        }

        bool enteredGate = EnterProviderGate();
        try
        {
            if (!IsCurrentTransition(version, active: true, conditional: true))
                return;

            InvokeProviderStop();

            // Stop/Dispose may win while provider Stop is running. Never resurrect the detector
            // after such a terminal transition.
            if (IsCurrentTransition(version, active: true, conditional: true))
                InvokeProviderStart();
        }
        finally
        {
            ExitProviderGate(enteredGate);
        }
    }

    public void Dispose()
    {
        lock (_stateLock)
        {
            if (_disposed)
                return;

            _disposed = true;
            _requestedActive = false;
            ++_transitionVersion;
        }

        // Do not dispose the semaphore: another thread may already be waiting on it. Marking the
        // adapter terminal above prevents new provider work; draining the current operation keeps
        // Dispose's postcondition that no provider call remains in flight. Reentrant disposal is
        // handled by EnterProviderGate without waiting on the current provider callback.
        bool enteredGate = EnterProviderGate();
        ExitProviderGate(enteredGate);
    }

    private void RunTransition(bool active, bool conditional)
    {
        long version;
        lock (_stateLock)
        {
            if (_disposed || (conditional && !_activation.IsActive))
                return;

            _requestedActive = active;
            version = ++_transitionVersion;
        }

        bool enteredGate = EnterProviderGate();
        try
        {
            if (!IsCurrentTransition(version, active, conditional))
                return;

            if (active)
                InvokeProviderStart();
            else
                InvokeProviderStop();
        }
        finally
        {
            ExitProviderGate(enteredGate);
        }
    }

    private bool IsCurrentTransition(long version, bool active, bool conditional)
    {
        lock (_stateLock)
        {
            return !_disposed &&
                _transitionVersion == version &&
                _requestedActive == active &&
                (!conditional || _activation.IsActive);
        }
    }

    private bool EnterProviderGate()
    {
        if (Volatile.Read(ref _providerThreadId) == Environment.CurrentManagedThreadId)
            return false;

        _providerGate.Wait();
        Volatile.Write(ref _providerThreadId, Environment.CurrentManagedThreadId);
        return true;
    }

    private void ExitProviderGate(bool enteredGate)
    {
        if (!enteredGate)
            return;

        Volatile.Write(ref _providerThreadId, 0);
        _providerGate.Release();
    }

    private void InvokeProviderStart() => _wakeup.Start();

    private void InvokeProviderStop() => _wakeup.Stop();

    private void InvokeProviderFeed(ReadOnlySpan<float> audio) => _wakeup.FeedAudio(audio);
}
