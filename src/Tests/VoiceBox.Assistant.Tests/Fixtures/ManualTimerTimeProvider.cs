namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Deterministic TimeProvider for timeout and scheduling contracts.
/// Timers run synchronously when the test advances monotonic time.
/// </summary>
internal sealed class ManualTimerTimeProvider : TimeProvider
{
    private readonly object _lock = new();
    private readonly List<ManualTimer> _timers = [];
    private TaskCompletionSource _timerAdded = NewSignal();
    private long _timestamp;

    public override long TimestampFrequency => TimeSpan.TicksPerSecond;

    public override long GetTimestamp() => Volatile.Read(ref _timestamp);

    public override ITimer CreateTimer(
        TimerCallback callback,
        object? state,
        TimeSpan dueTime,
        TimeSpan period)
    {
        var timer = new ManualTimer(this, callback, state, dueTime, period);
        TaskCompletionSource added;
        lock (_lock)
        {
            _timers.Add(timer);
            added = _timerAdded;
            _timerAdded = NewSignal();
        }
        added.TrySetResult();
        return timer;
    }

    public async Task WaitForTimerCountAsync(
        int expected,
        CancellationToken cancellationToken = default)
    {
        while (true)
        {
            Task added;
            lock (_lock)
            {
                if (_timers.Count >= expected)
                    return;
                added = _timerAdded.Task;
            }
            await added.WaitAsync(cancellationToken);
        }
    }

    public void Advance(TimeSpan duration)
    {
        long now = Interlocked.Add(ref _timestamp, duration.Ticks);
        while (true)
        {
            ManualTimer[] due;
            lock (_lock)
                due = _timers.Where(timer => timer.TryTakeDue(now)).ToArray();
            if (due.Length == 0)
                return;
            foreach (ManualTimer timer in due)
                timer.Invoke();
        }
    }

    private static TaskCompletionSource NewSignal()
        => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private sealed class ManualTimer(
        ManualTimerTimeProvider owner,
        TimerCallback callback,
        object? state,
        TimeSpan dueTime,
        TimeSpan period) : ITimer
    {
        private long _due = dueTime == Timeout.InfiniteTimeSpan
            ? long.MaxValue
            : owner.GetTimestamp() + dueTime.Ticks;
        private long _period = period == Timeout.InfiniteTimeSpan
            ? long.MaxValue
            : period.Ticks;
        private int _disposed;

        public bool Change(TimeSpan dueTime, TimeSpan period)
        {
            _due = dueTime == Timeout.InfiniteTimeSpan
                ? long.MaxValue
                : owner.GetTimestamp() + dueTime.Ticks;
            _period = period == Timeout.InfiniteTimeSpan ? long.MaxValue : period.Ticks;
            return Volatile.Read(ref _disposed) == 0;
        }

        public bool TryTakeDue(long now)
        {
            if (Volatile.Read(ref _disposed) != 0 || now < Volatile.Read(ref _due))
                return false;
            _due = _period == long.MaxValue ? long.MaxValue : checked(_due + _period);
            return true;
        }

        public void Invoke() => callback(state);

        public void Dispose() => Interlocked.Exchange(ref _disposed, 1);

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
