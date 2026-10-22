using System.Threading.Channels;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests.ASR;

public sealed class AsrSessionLifecycleTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    [Fact]
    public async Task FatalFailure_AutomaticallyReleasesAndPreservesCauseAfterRemoval()
    {
        await using var service = new TestService();
        SessionInfo info = await service.StartSessionAsync();
        TestSession session = service.Latest;
        var failure = new InvalidDataException("feature extraction failed");

        _ = session.FailAsync(failure);
        await session.ReleaseTask.WaitAsync(Timeout);

        Assert.Same(failure, session.FailureException);
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidDataException>(
            () => session.Results.Reader.Completion));
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidDataException>(
            () => service.FinishSessionAsync(info.SessionId)));
        Assert.Same(failure, Assert.Throws<InvalidDataException>(
            () => service.GetSession(info.SessionId)));
        await service.CancelSessionAsync(info.SessionId);
        await service.StartSessionAsync();
        Assert.Equal(1, session.CancelCount);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task FailureDuringFinish_ReportsOriginalErrorAndWaitsForRelease()
    {
        var session = new TestSession("finish-race");
        var entered = Signal();
        var release = Signal();
        session.Finish = async () =>
        {
            entered.SetResult();
            await Task.Delay(System.Threading.Timeout.Infinite, session.CancellationToken);
            return null;
        };
        session.Release = () => release.Task;
        await session.StartAsync(default);
        Task<AsrResult?> finish = session.FinishAsync(default);
        await entered.Task.WaitAsync(Timeout);
        var failure = new IOException("receive failed");

        Task cleanup = session.FailAsync(failure);
        await session.ReleaseEntered.Task.WaitAsync(Timeout);
        Assert.False(finish.IsCompleted);
        release.SetResult();

        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => finish.WaitAsync(Timeout)));
        await cleanup.WaitAsync(Timeout);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task FinishOnFailedSession_CannotBypassFaultPublicationAndCancellation()
    {
        var session = new TestSession("cleanup-order");
        var cancel = Signal();
        session.Cancel = () => cancel.Task;
        await session.StartAsync(default);
        var failure = new IOException("pump failed");

        Task cleanup = session.FailAsync(failure);
        await session.CancelEntered.Task.WaitAsync(Timeout);
        Task<AsrResult?> finish = session.FinishAsync(default);
        Assert.False(session.ReleaseEntered.Task.IsCompleted);
        Assert.False(finish.IsCompleted);
        cancel.SetResult();

        await cleanup.WaitAsync(Timeout);
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => finish.WaitAsync(Timeout)));
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.Results.Reader.Completion));
    }

    [Fact]
    public async Task CleanupErrors_DoNotReplaceFirstFailureOrSkipRelease()
    {
        var session = new TestSession("cleanup-errors");
        var callbackError = new InvalidOperationException("cancellation callback failed");
        using var registration = session.CancellationToken.Register(() => throw callbackError);
        var hookError = new InvalidOperationException("fault notification failed");
        session.Fault = _ => throw hookError;
        session.Cancel = () => throw new InvalidOperationException("cancel failed");
        session.Release = () => throw new InvalidOperationException("release failed");
        await session.StartAsync(default);
        var failure = new IOException("original failure");

        Task cleanup = session.FailAsync(failure);
        Assert.Same(hookError, await Assert.ThrowsAsync<InvalidOperationException>(
            () => cleanup.WaitAsync(Timeout)));
        Assert.Same(failure, session.FailureException);
        Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.FinishAsync(default)));
        Assert.Equal(1, session.CancelCount);
        Assert.Equal(1, session.ReleaseCount);
        Assert.True(session.ReleaseTask.IsCompleted);
    }

    [Fact]
    public async Task FailureBeforeStartReturns_WaitsForPublishedWorkerBeforeReleasingCapacity()
    {
        var manager = new AsrSessionManager(1);
        var session = new TestSession("assigned-by-manager");
        var start = Signal();
        var worker = Signal();
        var failure = new IOException("worker failed during startup");
        session.Start = async ct =>
        {
            _ = session.FailAsync(failure);
            await start.Task;
        };
        session.Release = () => worker.Task;
        Task<SessionInfo> starting = manager.StartAsync(_ => session, default);
        try
        {
            await session.CancelEntered.Task.WaitAsync(Timeout);
            Assert.False(session.ReleaseEntered.Task.IsCompleted);
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => manager.StartAsync(id => new TestSession(id), default));
            start.SetResult();
            await session.ReleaseEntered.Task.WaitAsync(Timeout);
            Assert.False(starting.IsCompleted);
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => manager.StartAsync(id => new TestSession(id), default));
            worker.SetResult();
            Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => starting.WaitAsync(Timeout)));
            await manager.StartAsync(id => new TestSession(id), default);
        }
        finally
        {
            start.TrySetResult();
            worker.TrySetResult();
            await manager.DisposeAsync();
        }
        Assert.Equal(1, session.ReleaseCount);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task StartOrFinishHookFailure_AutomaticallyCleansUp(bool duringStart)
    {
        var session = new TestSession("hook-failure");
        var failure = new IOException("provider hook failed");
        if (duringStart) session.Start = _ => throw failure;
        else session.Finish = () => throw failure;

        if (duringStart)
            Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.StartAsync(default)));
        else
        {
            await session.StartAsync(default);
            Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.FinishAsync(default)));
        }

        await session.ReleaseTask.WaitAsync(Timeout);
        Assert.Same(failure, session.FailureException);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task ConcurrentFailureCancelAndFinish_ReleaseExactlyOnce()
    {
        var session = new TestSession("concurrent-terminal");
        await session.StartAsync(default);
        var failure = new IOException("first failure");
        Task cleanup = session.FailAsync(failure);

        Task[] callers = Enumerable.Range(0, 32).Select(index => Task.Run(async () =>
        {
            if (index % 3 == 0) await session.CancelAsync();
            else if (index % 3 == 1) await session.FailAsync(new Exception("secondary failure"));
            else Assert.Same(failure, await Assert.ThrowsAsync<IOException>(() => session.FinishAsync(default)));
        })).ToArray();
        await Task.WhenAll(callers.Append(cleanup)).WaitAsync(Timeout);

        Assert.Same(failure, session.FailureException);
        Assert.Equal(1, session.FaultCount);
        Assert.Equal(1, session.CancelCount);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task CancelCallbackFailure_StillReleasesAndDoesNotBecomeFatalFailure()
    {
        var session = new TestSession("explicit-cancel");
        await session.StartAsync(default);
        var error = new IOException("callback failed");
        using var registration = session.CancellationToken.Register(() => throw error);

        Task cleanup = session.CancelAsync();
        var thrown = await Assert.ThrowsAsync<AggregateException>(() => cleanup.WaitAsync(Timeout));
        Assert.Same(error, Assert.Single(thrown.InnerExceptions));
        Assert.Same(cleanup, session.FailAsync(new IOException("late shutdown error")));

        Assert.Null(session.FailureException);
        Assert.Equal(1, session.ReleaseCount);
        Assert.True(session.CancellationToken.IsCancellationRequested);
    }

    [Fact]
    public async Task FinishWaitCancellation_ReleasesOnlyAfterInFlightWorkExits()
    {
        var session = new TestSession("wait-cancel");
        var finishEntered = Signal();
        var stop = Signal();
        session.Finish = async () =>
        {
            finishEntered.SetResult();
            await stop.Task;
            session.CancellationToken.ThrowIfCancellationRequested();
            return null;
        };
        await session.StartAsync(default);
        using var caller = new CancellationTokenSource();
        Task<AsrResult?> finish = session.FinishAsync(caller.Token);
        await finishEntered.Task.WaitAsync(Timeout);
        caller.Cancel();
        try
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => finish.WaitAsync(Timeout));
            await session.CancelEntered.Task.WaitAsync(Timeout);
            Assert.False(session.ReleaseTask.IsCompleted);
            Assert.Null(session.FailureException);
        }
        finally { stop.SetResult(); }
        await session.ReleaseTask.WaitAsync(Timeout);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task RetainedFinishWaitCancellation_AllowsLaterResultCollection()
    {
        var session = new TestSession("retained-finish", retainWaitCancellation: true);
        var result = new AsrResult { Text = "complete" };
        var completion = new TaskCompletionSource<AsrResult?>(TaskCreationOptions.RunContinuationsAsynchronously);
        session.Finish = () => completion.Task;
        await session.StartAsync(default);
        using var caller = new CancellationTokenSource();
        Task<AsrResult?> first = session.FinishAsync(caller.Token);
        caller.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        Assert.False(session.CancellationToken.IsCancellationRequested);
        Assert.False(session.ReleaseTask.IsCompleted);

        completion.SetResult(result);
        Assert.Same(result, await session.FinishAsync(default).WaitAsync(Timeout));
        Assert.Equal(0, session.CancelCount);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task SuccessfulFinish_IgnoresLateFailure()
    {
        var session = new TestSession("success");
        var result = new AsrResult { Text = "ok" };
        session.Finish = () => Task.FromResult<AsrResult?>(result);
        await session.StartAsync(default);

        Assert.Same(result, await session.FinishAsync(default));
        await session.FailAsync(new IOException("late failure"));

        Assert.Null(session.FailureException);
        Assert.Equal(0, session.FaultCount);
        Assert.Equal(1, session.ReleaseCount);
    }

    [Fact]
    public async Task FailureHistory_ExpiresAndKeepsOnlyLatest128Errors()
    {
        var clock = new TestTimeProvider();
        var manager = new AsrSessionManager(1, clock);
        var ids = new List<string>();
        var errors = new List<IOException>();
        try
        {
            for (int index = 0; index < 129; index++)
            {
                TestSession? session = null;
                SessionInfo info = await manager.StartAsync(id => session = new TestSession(id), default);
                var error = new IOException($"failure {index}");
                await session!.FailAsync(error).WaitAsync(Timeout);
                ids.Add(info.SessionId);
                errors.Add(error);
            }
            await Assert.ThrowsAsync<InvalidOperationException>(() => manager.FinishAsync(ids[0], default));
            Assert.Same(errors[1], await Assert.ThrowsAsync<IOException>(() => manager.FinishAsync(ids[1], default)));
            Assert.Same(errors[^1], await Assert.ThrowsAsync<IOException>(() => manager.FinishAsync(ids[^1], default)));

            clock.Advance(TimeSpan.FromMinutes(5));
            await Assert.ThrowsAsync<InvalidOperationException>(() => manager.FinishAsync(ids[^1], default));
        }
        finally { await manager.DisposeAsync(); }
    }

    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private sealed class TestService() : BaseASRService(1)
    {
        public TestSession Latest { get; private set; } = null!;
        protected override AsrSession CreateSession(string id) => Latest = new TestSession(id);
        public AsrSession? GetSession(string id) => TryGetSession<TestSession>(id, out var session) ? session : null;
    }

    private sealed class TestSession(string id, bool retainWaitCancellation = false) : AsrSession(id)
    {
        public Func<CancellationToken, Task> Start { get; set; } = _ => Task.CompletedTask;
        public Func<Task<AsrResult?>> Finish { get; set; } = () => Task.FromResult<AsrResult?>(null);
        public Func<Task> Cancel { get; set; } = () => Task.CompletedTask;
        public Func<Task> Release { get; set; } = () => Task.CompletedTask;
        public Action<Exception>? Fault { get; set; }
        public Channel<AsrResult> Results { get; } = Channel.CreateUnbounded<AsrResult>();
        public TaskCompletionSource CancelEntered { get; } = Signal();
        public TaskCompletionSource ReleaseEntered { get; } = Signal();
        public int FaultCount;
        public int CancelCount;
        public int ReleaseCount;

        protected override bool RetainAfterFinishWaitCancellation => retainWaitCancellation;
        protected override Task StartSessionAsync(CancellationToken ct) => Start(ct);
        public override void FeedAudio(ReadOnlySpan<float> audioData) { }
        protected override Task<AsrResult?> FinishSessionAsync() => Finish();
        protected override void OnSessionFaulted(Exception error)
        {
            Interlocked.Increment(ref FaultCount);
            Results.Writer.TryComplete(error);
            Fault?.Invoke(error);
        }
        protected override Task CancelSessionAsync()
        {
            Interlocked.Increment(ref CancelCount);
            CancelEntered.TrySetResult();
            Results.Writer.TryComplete();
            return Cancel();
        }
        protected override Task ReleaseSessionAsync()
        {
            Interlocked.Increment(ref ReleaseCount);
            ReleaseEntered.TrySetResult();
            return Release();
        }
    }

    private sealed class TestTimeProvider : TimeProvider
    {
        private DateTimeOffset _now = DateTimeOffset.Parse("2026-10-14T00:00:00Z");
        public override DateTimeOffset GetUtcNow() => _now;
        public void Advance(TimeSpan duration) => _now += duration;
    }
}
