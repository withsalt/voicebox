using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.Telemetry;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Converges a terminal control-plane fault exactly once: it faults the lifecycle, fails every
/// waiting control operation, closes the input gate, cancels the run and schedules a single
/// supervised stop. It never mutates a turn.
/// </summary>
internal sealed class AssistantFaultConverger
{
    private readonly AssistantRuntimeLifecycleStateMachine _lifecycle;
    private readonly object _lifecycleLock;
    private readonly ControlOperationRegistry _operations;
    private readonly BargeInDataPlaneController _bargeInDataPlane;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly RuntimeRunScope _runScope;
    private readonly InputGate _gate;
    private readonly ILogger _logger;
    private IRuntimeStopPort? _stop;
    private int _stopScheduled;

    public AssistantFaultConverger(
        AssistantRuntimeLifecycleStateMachine lifecycle,
        object lifecycleLock,
        ControlOperationRegistry operations,
        BargeInDataPlaneController bargeInDataPlane,
        AssistantTaskSupervisor tasks,
        RuntimeRunScope runScope,
        InputGate gate,
        ILogger logger)
    {
        _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
        _lifecycleLock = lifecycleLock ?? throw new ArgumentNullException(nameof(lifecycleLock));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _bargeInDataPlane = bargeInDataPlane ?? throw new ArgumentNullException(nameof(bargeInDataPlane));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public void Bind(IRuntimeStopPort stop) => _stop = stop ?? throw new ArgumentNullException(nameof(stop));

    public void Converge(AssistantFailure failure)
    {
        lock (_lifecycleLock)
            _lifecycle.MarkFaulted();

        _operations.FailPending($"{failure.Code}:{failure.Component}:{failure.MessageKey}");
        Exception exception = _bargeInDataPlane.FatalFault ?? new InvalidOperationException(
            $"{failure.Code}:{failure.Component}:{failure.MessageKey}");
        _gate.Close();
        _runScope.Cancel();
        _logger.LogCritical(
            AssistantRuntimeEventIds.TerminalFault,
            exception,
            "Assistant control ingress terminated ({Code}, {Component}, {MessageKey}); stopping.",
            failure.Code,
            failure.Component,
            failure.MessageKey);
        if (Interlocked.Exchange(ref _stopScheduled, 1) != 0)
            return;

        _tasks.Track(
            Task.Run(StopAfterFaultAsync),
            "runtime.fatal-stop",
            ownerTurnId: null,
            AssistantTaskFaultPolicy.ObserveAndLog,
            awaitOnStop: false);
    }

    private async Task StopAfterFaultAsync()
    {
        try
        {
            await _stop!.StopAsync().ConfigureAwait(false);
        }
        catch (Exception stopError)
        {
            _logger.LogCritical(stopError, "VoiceAssistant failed to stop after a render-aware DTD fault.");
        }
    }
}
