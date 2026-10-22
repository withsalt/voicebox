using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Supervises every fire-and-forget runtime task so shutdown can await a named, owned,
/// fault-observed set instead of racing background continuations. Effect handlers register their
/// asynchronous work here rather than starting unmanaged tasks.
/// </summary>
internal sealed class AssistantTaskSupervisor : IEffectBackgroundScheduler
{
    private readonly object _lock = new();
    private readonly Dictionary<Task, AssistantTaskRegistration> _tasks = new();
    private readonly ILogger _logger;

    public AssistantTaskSupervisor(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public int ActiveTaskCount
    {
        get
        {
            lock (_lock)
                return _tasks.Count;
        }
    }

    public Task Track(
        Task task,
        string taskName,
        long? ownerTurnId,
        AssistantTaskFaultPolicy faultPolicy,
        bool awaitOnStop = true)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentException.ThrowIfNullOrWhiteSpace(taskName);

        var registration = new AssistantTaskRegistration(taskName, ownerTurnId, faultPolicy);
        if (awaitOnStop)
        {
            bool added;
            lock (_lock)
                added = _tasks.TryAdd(task, registration);

            if (!added)
                return task;

            _ = task.ContinueWith(
                static (completed, state) => ((AssistantTaskSupervisor)state!).Untrack(completed),
                this,
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
        else
        {
            _ = task.ContinueWith(
                completed =>
                {
                    if (!completed.IsFaulted)
                        return;

                    _logger.LogWarning(
                        completed.Exception?.Flatten(),
                        "VoiceAssistant background task {TaskName} for turn {OwnerTurnId} faulted.",
                        taskName,
                        ownerTurnId);
                },
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }

        return task;
    }

    void IEffectBackgroundScheduler.Schedule(string taskName, long ownerTurnId, Func<Task> work)
        => Track(Task.Run(work), taskName, ownerTurnId, AssistantTaskFaultPolicy.ObserveAndLog);

    public async Task WaitForCompletionAsync()
    {
        while (true)
        {
            Task[] tasks = Snapshot();
            if (tasks.Length == 0)
                return;

            try
            {
                await Task.WhenAll(tasks).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception)
            {
                // Each fault is observed and logged by Untrack. Awaiting WhenAll here
                // still guarantees that shutdown does not race the task's completion.
            }
        }
    }

    private void Untrack(Task task)
    {
        AggregateException? exception = task.IsFaulted ? task.Exception?.Flatten() : null;
        AssistantTaskRegistration registration;
        lock (_lock)
        {
            if (!_tasks.Remove(task, out registration))
                return;
        }

        if (exception is null)
            return;

        switch (registration.FaultPolicy)
        {
            case AssistantTaskFaultPolicy.ObserveAndLog:
                _logger.LogWarning(
                    exception,
                    "VoiceAssistant background task {TaskName} for turn {OwnerTurnId} faulted.",
                    registration.Name,
                    registration.OwnerTurnId);
                break;
        }
    }

    private Task[] Snapshot()
    {
        lock (_lock)
            return _tasks.Keys.ToArray();
    }
}
