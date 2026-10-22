using Microsoft.Extensions.Logging;
using System.Runtime.ExceptionServices;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Realtime.Resampling;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.Telemetry;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Owns the runtime's Start, Stop and Dispose sequences: the run scope, the ordered lifecycle
/// barriers on the control plane, the module topology hand-off, and the two lifecycle state
/// publications TurnKernel does not model. Conversation semantics stay entirely with the Kernel.
/// </summary>
internal sealed class AssistantRuntimeLifecycle : IRuntimeStopPort
{
    private readonly object _lifecycleLock;
    private readonly RuntimeRunScope _runScope;
    private readonly AssistantRuntimeLifecycleCoordinator _modules;
    private readonly RuntimeConversationStateMirror _stateMirror;
    private readonly AssistantControlLoopHost _loopHost;
    private readonly ControlPlaneAvailability _availability;
    private readonly AssistantControlSignalGateway _controlSignals;
    private readonly ControlOperationRegistry _operations;
    private readonly AssistantControlIngress _ingress;
    private readonly AssistantRuntimeSubscriptionOwner _subscriptions;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly TurnCompletionCoordinator _turnCompletion;
    private readonly RecognitionTurnCoordinator _recognition;
    private readonly VadSpeakerGateCoordinator _speakerGate;
    private readonly BargeInDataPlaneController _bargeInDataPlane;
    private readonly INearEndResetPort _nearEndReset;
    private readonly PlaybackInterruptionController _playbackInterruption;
    private readonly OrderedPublicationBoundary _publications;
    private readonly TurnContextStore _turnContexts;
    private readonly AudioResamplerPool _resamplers;
    private readonly ResponseModule _responses;
    private readonly PlaybackModule _playback;
    private readonly InputGate _gate;
    private readonly bool _wakeupInstalled;
    private readonly ILogger _logger;
    private object _disposalOwner;
    private Task _startupTask = Task.CompletedTask;
    private Task? _stopTask;

    public AssistantRuntimeLifecycle(
        object lifecycleLock,
        RuntimeRunScope runScope,
        AssistantRuntimeLifecycleCoordinator modules,
        RuntimeConversationStateMirror stateMirror,
        AssistantControlLoopHost loopHost,
        ControlPlaneAvailability availability,
        AssistantControlSignalGateway controlSignals,
        ControlOperationRegistry operations,
        AssistantControlIngress ingress,
        AssistantRuntimeSubscriptionOwner subscriptions,
        AssistantTaskSupervisor tasks,
        TurnCompletionCoordinator turnCompletion,
        RecognitionTurnCoordinator recognition,
        VadSpeakerGateCoordinator speakerGate,
        BargeInDataPlaneController bargeInDataPlane,
        INearEndResetPort nearEndReset,
        PlaybackInterruptionController playbackInterruption,
        OrderedPublicationBoundary publications,
        TurnContextStore turnContexts,
        AudioResamplerPool resamplers,
        ResponseModule responses,
        PlaybackModule playback,
        InputGate gate,
        bool wakeupInstalled,
        ILogger logger)
    {
        _lifecycleLock = lifecycleLock ?? throw new ArgumentNullException(nameof(lifecycleLock));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _modules = modules ?? throw new ArgumentNullException(nameof(modules));
        _stateMirror = stateMirror ?? throw new ArgumentNullException(nameof(stateMirror));
        _loopHost = loopHost ?? throw new ArgumentNullException(nameof(loopHost));
        _availability = availability ?? throw new ArgumentNullException(nameof(availability));
        _controlSignals = controlSignals ?? throw new ArgumentNullException(nameof(controlSignals));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _subscriptions = subscriptions ?? throw new ArgumentNullException(nameof(subscriptions));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _turnCompletion = turnCompletion ?? throw new ArgumentNullException(nameof(turnCompletion));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _speakerGate = speakerGate ?? throw new ArgumentNullException(nameof(speakerGate));
        _bargeInDataPlane = bargeInDataPlane ?? throw new ArgumentNullException(nameof(bargeInDataPlane));
        _nearEndReset = nearEndReset ?? throw new ArgumentNullException(nameof(nearEndReset));
        _playbackInterruption = playbackInterruption ?? throw new ArgumentNullException(nameof(playbackInterruption));
        _publications = publications ?? throw new ArgumentNullException(nameof(publications));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _resamplers = resamplers ?? throw new ArgumentNullException(nameof(resamplers));
        _responses = responses ?? throw new ArgumentNullException(nameof(responses));
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _wakeupInstalled = wakeupInstalled;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _disposalOwner = this;
    }

    /// <summary>The facade is the object an ObjectDisposedException must name.</summary>
    public void Bind(object disposalOwner)
        => _disposalOwner = disposalOwner ?? throw new ArgumentNullException(nameof(disposalOwner));

    public AssistantRuntimeLifecycleState State => _modules.State;

    public void ThrowIfDisposed() => _runScope.ThrowIfDisposed(_disposalOwner);

    public async Task StartAsync(CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var startupCompletion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        CancellationTokenSource runCts;
        lock (_lifecycleLock)
        {
            ThrowIfDisposed();
            if (_runScope.IsStarted || !_modules.BeginStart())
                throw new InvalidOperationException("VoiceAssistant 已启动, 不可重复 Start。");

            runCts = _runScope.Begin();
            _startupTask = startupCompletion.Task;
            _stopTask = null;
        }

        try
        {
            _loopHost.EnsureStarted();
            await _controlSignals.EnqueueCommandAsync(RuntimeCommandKind.Start, ct).ConfigureAwait(false);
            _ = _tasks.Track(
                _recognition.StartAudioPump(runCts.Token),
                "recognition.audio-feed-pump",
                ownerTurnId: null,
                AssistantTaskFaultPolicy.ObserveAndLog);
            _subscriptions.Attach();
            using var startupCts = CancellationTokenSource.CreateLinkedTokenSource(ct, runCts.Token);
            await _modules.StartAsync(startupCts.Token).ConfigureAwait(false);
            _ = _tasks.Track(
                _playback.Start(runCts.Token),
                "playback.logical-queue",
                ownerTurnId: null,
                AssistantTaskFaultPolicy.ObserveAndLog);
            startupCts.Token.ThrowIfCancellationRequested();
            if (!_modules.CompleteStart())
                throw new OperationCanceledException(startupCts.Token);
            _gate.Open();

            bool stateChanged = _stateMirror.TryTransition(AssistantState.Idle, out AssistantState oldState);
            _logger.LogInformation(AssistantRuntimeEventIds.Started, "VoiceAssistant started.");
            startupCompletion.TrySetResult();

            if (!runCts.IsCancellationRequested && stateChanged)
                _turnCompletion.PublishStateChanged(0, oldState, AssistantState.Idle);
            if (!runCts.IsCancellationRequested && _wakeupInstalled)
            {
                long operationId = _operations.NextOperationId();
                await _controlSignals.EnqueueAsync(
                    AssistantSignal.WakeupModeChanged(
                        operationId,
                        AssistantWakeMode.Disabled,
                        AssistantWakeMode.WaitingForWakeWord),
                    AssistantSignalSource.HostLifecycle,
                    startupCts.Token).ConfigureAwait(false);
            }

            await _publications.DrainAsync(runCts.Token).ConfigureAwait(false);
        }
        catch
        {
            _gate.Close();
            _subscriptions.Detach();
            _runScope.Cancel();
            try { await _playback.StopAsync().ConfigureAwait(false); } catch { }
            // Startup can fail after the last module has started (for example, cancellation
            // at the final startup check). Those participants still need reverse-order stop.
            await _modules.RollbackStartupAsync().ConfigureAwait(false);
            try { runCts.Dispose(); } catch { }

            bool stateChanged;
            AssistantState oldState;
            lock (_lifecycleLock)
            {
                _runScope.ClearIfCurrent(runCts);
                _runScope.MarkStopped();
                _modules.FailStart();
                stateChanged = _stateMirror.TryTransition(AssistantState.Stopped, out oldState);
            }

            startupCompletion.TrySetResult();
            if (stateChanged)
                _turnCompletion.PublishStateChanged(0, oldState, AssistantState.Stopped);
            await _publications.DrainAsync().ConfigureAwait(false);
            throw;
        }
    }

    public Task StopAsync()
    {
        Task startupTask;
        CancellationTokenSource? runCts;
        TaskCompletionSource? stopCompletion;
        Task stopTask;
        lock (_lifecycleLock)
        {
            if (_stopTask is not null)
                return _stopTask;
            if (_runScope.IsDisposed)
                return Task.CompletedTask;

            stopTask = PrepareStopLocked(out startupTask, out runCts, out stopCompletion);
        }

        return StartPreparedStop(stopTask, startupTask, runCts, stopCompletion);
    }

    public void Dispose()
    {
        Task startupTask;
        CancellationTokenSource? runCts;
        TaskCompletionSource? stopCompletion;
        Task stopTask;
        lock (_lifecycleLock)
        {
            if (!_runScope.TryLatchDisposed())
                return;

            stopTask = PrepareStopLocked(out startupTask, out runCts, out stopCompletion);
        }

        stopTask = StartPreparedStop(stopTask, startupTask, runCts, stopCompletion);
        try { stopTask.GetAwaiter().GetResult(); } catch { }
        _operations.FailPending("disposed");
        try
        {
            _loopHost.EnsureStarted();
            if (_availability.CanEnqueueLifecycleCommand)
                _controlSignals.EnqueueCommandAsync(RuntimeCommandKind.Dispose).GetAwaiter().GetResult();
        }
        catch { }

        _loopHost.RequestStop();
        _recognition.ClearDeferred();
        _loopHost.WaitForCompletion();
        _loopHost.Dispose();
        try { _ingress.Dispose(); } catch { }
        try { _publications.DrainAsync().GetAwaiter().GetResult(); } catch { }
        try { _publications.DisposeAsync().AsTask().GetAwaiter().GetResult(); } catch { }
        try { _responses.Dispose(); } catch { }
        try { _playback.Dispose(); } catch { }
        try { _runScope.Detach()?.Dispose(); } catch { }
        try { _resamplers.Dispose(); } catch { }
        try { _turnContexts.Dispose(); } catch { }
        _modules.DisposeAssistantOwned();
        _modules.MarkDisposed();
    }

    private Task PrepareStopLocked(
        out Task startupTask,
        out CancellationTokenSource? runCts,
        out TaskCompletionSource? stopCompletion)
    {
        startupTask = _startupTask;
        runCts = null;
        stopCompletion = null;
        if (_stopTask is not null)
            return _stopTask;
        if (!_runScope.IsStarted ||
            _stateMirror.Current == AssistantState.Stopped ||
            !_modules.BeginStop())
        {
            return Task.CompletedTask;
        }

        runCts = _runScope.Cts;
        stopCompletion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _stopTask = stopCompletion.Task;
        return _stopTask;
    }

    private Task StartPreparedStop(
        Task stopTask,
        Task startupTask,
        CancellationTokenSource? runCts,
        TaskCompletionSource? stopCompletion)
    {
        if (stopCompletion is null)
            return stopTask;

        try { runCts?.Cancel(); } catch { }
        _operations.FailPending("stop-requested");
        _ = _tasks.Track(
            CompleteStopAfterStartupAsync(startupTask, runCts, stopCompletion),
            "runtime.stop-after-startup",
            ownerTurnId: null,
            AssistantTaskFaultPolicy.ObserveAndLog,
            awaitOnStop: false);
        return stopTask;
    }

    private async Task CompleteStopAfterStartupAsync(
        Task startupTask,
        CancellationTokenSource? runCts,
        TaskCompletionSource stopCompletion)
    {
        try
        {
            await startupTask.ConfigureAwait(false);
            bool shouldStop;
            lock (_lifecycleLock)
                shouldStop = _runScope.IsStarted && _runScope.IsCurrentRun(runCts);
            if (shouldStop)
                await StopRuntimeAsync().ConfigureAwait(false);
            else
                _modules.CompleteStop();
            stopCompletion.TrySetResult();
        }
        catch (Exception ex)
        {
            lock (_lifecycleLock)
            {
                if (ReferenceEquals(_stopTask, stopCompletion.Task) &&
                    _stateMirror.Current != AssistantState.Stopped)
                {
                    _stopTask = null;
                }
            }

            stopCompletion.TrySetException(ex);
        }
    }

    private async Task StopRuntimeAsync()
    {
        if (_availability.CanEnqueueLifecycleCommand)
            await _controlSignals.EnqueueCommandAsync(RuntimeCommandKind.Stop).ConfigureAwait(false);
        _logger.LogDebug("VoiceAssistant stopping...");
        _gate.Close();
        _speakerGate.Reset();
        _bargeInDataPlane.ResetAll();
        _nearEndReset.Reset();
        _runScope.Cancel();
        _responses.CancelBackgroundOperations();
        _recognition.ClearDeferred();
        CancelActiveResponse();
        Exception? playbackFailure = null;
        try { await _playback.StopAsync().ConfigureAwait(false); }
        catch (Exception exception) { playbackFailure = exception; }
        // A failed playback worker must not prevent capture shutdown, subscription removal,
        // or run-scope retirement. Preserve its failure after the remaining teardown.
        _subscriptions.Detach();
        await _modules.StopAsync().ConfigureAwait(false);

        string? abandonedSession = _recognition.TakeAbandonedSession();
        if (abandonedSession is not null)
            await _recognition.CleanupAbandonedSessionAsync(abandonedSession).ConfigureAwait(false);

        // 先置 null 再 Dispose: 让并发读取 run token 的调用方一旦读到 null 即走 default,
        // 缩小 Dispose 期间 ObjectDisposedException 的窗口。
        try { _runScope.Detach()?.Dispose(); } catch { }

        _modules.CompleteStop();
        TransitionToStopped();
        _runScope.MarkStopped();
        await _publications.DrainAsync().ConfigureAwait(false);
        _logger.LogInformation(AssistantRuntimeEventIds.Stopped, "VoiceAssistant stopped.");
        if (playbackFailure is not null)
            ExceptionDispatchInfo.Capture(playbackFailure).Throw();
    }

    private void TransitionToStopped()
    {
        if (_stateMirror.TryTransition(AssistantState.Stopped, out AssistantState old))
            _turnCompletion.PublishStateChanged(0, old, AssistantState.Stopped);
    }

    private void CancelActiveResponse()
    {
        ((IActiveResponseCancellationPort)_responses).CancelActiveResponse();
    }
}
