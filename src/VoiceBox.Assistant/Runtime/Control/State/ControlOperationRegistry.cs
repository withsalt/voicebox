using VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.State;

/// <summary>
/// Bookkeeping for in-flight control-plane operations: identity allocation, the awaiting
/// producers of lifecycle commands and semantic signals, and the real-time interruption
/// admissions the consumer has not reduced yet. It performs no dispatch and holds no Turn state.
/// </summary>
internal sealed class ControlOperationRegistry : IInterruptionAdmissionProgressView
{
    private readonly object _lock = new();
    private readonly Dictionary<long, TaskCompletionSource> _commandCompletions = [];
    private readonly Dictionary<SignalCompletionKey, TaskCompletionSource<AssistantControlSignalResult>>
        _signalCompletions = [];
    private string? _signalTerminationReason;
    private long _nextOperationId;
    private int _pendingInterruptionAdmissions;

    public long NextOperationId() => Interlocked.Increment(ref _nextOperationId);

    /// <summary>Producers still awaiting a verdict. A completed or abandoned signal leaves none.</summary>
    public int PendingSignalCount
    {
        get { lock (_lock) return _signalCompletions.Count; }
    }

    public bool HasSettledAllInterruptionAdmissions
        => Volatile.Read(ref _pendingInterruptionAdmissions) == 0;

    public void ArmInterruptionAdmission() => Interlocked.Increment(ref _pendingInterruptionAdmissions);

    public void DisarmInterruptionAdmission() => Interlocked.Decrement(ref _pendingInterruptionAdmissions);

    /// <summary>
    /// Releases the admission recorded for a real-time interruption decision once the single
    /// control consumer has reduced it. Only the near-end confirmation producer arms the counter;
    /// <see cref="AssistantSignalKind.InterruptionResolved"/> sources never do.
    /// </summary>
    public void SettleInterruptionAdmission(AssistantSignal signal)
    {
        if (signal.Kind == AssistantSignalKind.ActiveNearEndInterruptionRequested)
            DisarmInterruptionAdmission();
    }

    public TaskCompletionSource RegisterCommand(long operationId)
    {
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_lock)
        {
            if (!_commandCompletions.TryAdd(operationId, completion))
                throw new InvalidOperationException($"Duplicate lifecycle operation id {operationId}.");
        }

        return completion;
    }

    public void ForgetCommand(long operationId)
    {
        lock (_lock)
            _commandCompletions.Remove(operationId);
    }

    public void CompleteCommand(long operationId)
    {
        lock (_lock)
        {
            if (_commandCompletions.TryGetValue(operationId, out TaskCompletionSource? completion))
                completion.TrySetResult();
        }
    }

    public void CancelCommand(long operationId, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            if (_commandCompletions.TryGetValue(operationId, out TaskCompletionSource? completion))
                completion.TrySetCanceled(cancellationToken);
        }
    }

    public TaskCompletionSource<AssistantControlSignalResult> RegisterSignal(AssistantSignal signal)
    {
        var completion = new TaskCompletionSource<AssistantControlSignalResult>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var key = new SignalCompletionKey(signal.OperationId, signal.Kind);
        lock (_lock)
        {
            if (_signalTerminationReason is { } reason)
            {
                throw new OperationCanceledException(
                    $"Assistant control signal terminated: {reason}.");
            }

            if (!_signalCompletions.TryAdd(key, completion))
            {
                throw new InvalidOperationException(
                    $"Duplicate control signal completion for operation {signal.OperationId}, kind {signal.Kind}.");
            }
        }

        return completion;
    }

    public void ForgetSignal(
        AssistantSignal signal,
        TaskCompletionSource<AssistantControlSignalResult> completion)
    {
        var key = new SignalCompletionKey(signal.OperationId, signal.Kind);
        lock (_lock)
        {
            if (_signalCompletions.TryGetValue(key, out TaskCompletionSource<AssistantControlSignalResult>? current) &&
                ReferenceEquals(current, completion))
            {
                _signalCompletions.Remove(key);
            }
        }
    }

    public void CompleteSignal(AssistantSignal signal, AssistantControlSignalResult result)
    {
        TaskCompletionSource<AssistantControlSignalResult>? completion;
        var key = new SignalCompletionKey(signal.OperationId, signal.Kind);
        lock (_lock)
            _signalCompletions.Remove(key, out completion);

        completion?.TrySetResult(result);
    }

    public void FailPending(string reason)
    {
        TaskCompletionSource[] commands;
        TaskCompletionSource<AssistantControlSignalResult>[] signals;
        lock (_lock)
        {
            _signalTerminationReason ??= reason;
            commands = [.. _commandCompletions.Values];
            signals = [.. _signalCompletions.Values];
            _commandCompletions.Clear();
            _signalCompletions.Clear();
        }

        foreach (TaskCompletionSource completion in commands)
        {
            completion.TrySetException(new OperationCanceledException(
                $"Assistant control operation terminated: {reason}."));
        }

        foreach (TaskCompletionSource<AssistantControlSignalResult> completion in signals)
        {
            completion.TrySetException(new OperationCanceledException(
                $"Assistant control signal terminated: {reason}."));
        }
    }

    private readonly record struct SignalCompletionKey(
        long OperationId,
        AssistantSignalKind SignalKind);
}
