using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class BackgroundTaskTrackerTests
{
    [Fact]
    public async Task Supervisor_TracksNamedTurnOwnedTasksUntilTheyAreObserved()
    {
        var logger = new RecordingLogger();
        var supervisor = new AssistantTaskSupervisor(logger);
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Task task = supervisor.Track(
            completion.Task,
            "recognition.turn",
            ownerTurnId: 42,
            AssistantTaskFaultPolicy.ObserveAndLog);

        Assert.Equal(1, supervisor.ActiveTaskCount);
        Task drain = supervisor.WaitForCompletionAsync();
        Assert.False(drain.IsCompleted);
        completion.TrySetResult();
        await task;
        await drain;

        Assert.Equal(0, supervisor.ActiveTaskCount);
        Assert.Empty(logger.Entries);
    }

    [Fact]
    public async Task Track_AlreadyFaultedTask_RecordsItsException()
    {
        var logger = new RecordingLogger();
        var tracker = new AssistantTaskSupervisor(logger);
        var expected = new InvalidOperationException("background failure");

        _ = tracker.Track(
            Task.FromException(expected),
            "recognition.turn",
            ownerTurnId: 1,
            AssistantTaskFaultPolicy.ObserveAndLog);
        await tracker.WaitForCompletionAsync();

        LogEntry entry = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Warning, entry.Level);
        var aggregate = Assert.IsType<AggregateException>(entry.Exception);
        Assert.Contains(expected, aggregate.Flatten().InnerExceptions);
    }

    private readonly record struct LogEntry(LogLevel Level, Exception? Exception);

    private sealed class RecordingLogger : ILogger
    {
        private readonly List<LogEntry> _entries = [];

        public IReadOnlyList<LogEntry> Entries
        {
            get
            {
                lock (_entries)
                    return _entries.ToArray();
            }
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            lock (_entries)
                _entries.Add(new LogEntry(logLevel, exception));
        }
    }
}
