using System.Collections.Concurrent;
using VoiceBox.Assistant.Api.Playback;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeechPlaybackCoordinatorTests
{
    private sealed class DelegateRequestExecutor(
        Func<SpeechPlaybackRequest, CancellationToken, Task<string>> execute) :
        ISpeechPlaybackRequestExecutor
    {
        public Task<string> ExecuteAsync(
            SpeechPlaybackRequest request,
            CancellationToken cancellationToken)
            => execute(request, cancellationToken);
    }

    private sealed class CountingDeviceQueue : IPlaybackDeviceQueuePort
    {
        private int _clearCount;

        public int ClearCount => Volatile.Read(ref _clearCount);

        public float Gain { get; private set; } = 1;

        public List<float> GainChanges { get; } = [];

        public void ClearQueue() => Interlocked.Increment(ref _clearCount);

        public void SetGain(float gain)
        {
            Gain = gain;
            GainChanges.Add(gain);
        }
    }

    private sealed class BlockingDeviceQueue : IPlaybackDeviceQueuePort
    {
        public TaskCompletionSource Entered { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public TaskCompletionSource Release { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public void ClearQueue()
        {
            Entered.TrySetResult();
            Release.Task.GetAwaiter().GetResult();
        }

        public float Gain { get; private set; } = 1;

        public void SetGain(float gain) => Gain = gain;
    }

    private sealed class FailingDeviceQueue(Exception failure) : IPlaybackDeviceQueuePort
    {
        public void ClearQueue() => throw failure;

        public void SetGain(float gain) { }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CancellationDeviceFailure_CompletesActiveRequestAndClosesAdmission(bool fullQueue)
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var failure = new InvalidOperationException("device clear failed");
        SpeechPlaybackRequest? executed = null;
        using var coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (request, token) =>
            {
                executed = request;
                entered.TrySetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, token);
                return string.Empty;
            }),
            new FailingDeviceQueue(failure));
        using var cancellation = new CancellationTokenSource();
        Task worker = coordinator.Start(CancellationToken.None);
        SpeechEnqueueResult active = await coordinator.EnqueueAsync(7, TextStream("active"), cancellation.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        SpeechEnqueueResult pending = await coordinator.EnqueueAsync(8, TextStream("pending"), CancellationToken.None);
        var additional = new List<SpeechEnqueueResult>();
        Task<SpeechEnqueueResult>? blockedAdmission = null;
        if (fullQueue)
        {
            for (int i = 0; i < 31; i++)
                additional.Add(await coordinator.EnqueueAsync(9 + i, TextStream("pending"), CancellationToken.None));
            blockedAdmission = coordinator.EnqueueAsync(40, TextStream("blocked"), CancellationToken.None).AsTask();
            Assert.False(blockedAdmission.IsCompleted);
        }

        cancellation.Cancel();
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => worker.WaitAsync(TimeSpan.FromSeconds(2))));
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => active.Completion.WaitAsync(TimeSpan.FromSeconds(2))));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.Completion);
        foreach (SpeechEnqueueResult submission in additional)
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => submission.Completion);
        if (blockedAdmission is not null)
        {
            Exception? error = await Record.ExceptionAsync(
                () => blockedAdmission.WaitAsync(TimeSpan.FromSeconds(2)));
            Assert.True(error is OperationCanceledException or InvalidOperationException);
        }
        Assert.Throws<ObjectDisposedException>(() => executed!.Cancellation.Token);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await coordinator.EnqueueAsync(7, TextStream("late"), CancellationToken.None));

        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(coordinator.StopAsync));
        // Even a faulted worker must release its run scope during Stop.
        await coordinator.StopAsync();
    }

    [Fact]
    public async Task DuckingAndRestorationAreScopedToActiveRequestAndPreserveOriginalGain()
    {
        var firstStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var secondStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFirst = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseSecond = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var device = new CountingDeviceQueue();
        device.SetGain(0.6f);
        device.GainChanges.Clear();
        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, token) =>
        {
            if (request.OwnerTurnId == 7)
            {
                firstStarted.TrySetResult();
                await releaseFirst.Task.WaitAsync(token);
            }
            else
            {
                Assert.Equal(0.6f, device.Gain);
                secondStarted.TrySetResult();
                await releaseSecond.Task.WaitAsync(token);
            }
            return "done";
        }), device);
        _ = coordinator.Start(CancellationToken.None);
        var first = await coordinator.EnqueueAsync(7, TextStream("first"), CancellationToken.None);
        await firstStarted.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.False(coordinator.TryDuckActivePlayback(8, 0.15f, 0.6f));
        Assert.False(coordinator.TryDuckActivePlayback(7, 0.7f, 0.6f));
        Assert.True(coordinator.TryDuckActivePlayback(7, 0.15f, 0.6f));
        Assert.True(coordinator.TryDuckActivePlayback(7, 0.15f, 0.15f));
        Assert.Equal(new[] { 0.15f }, device.GainChanges);
        Assert.False(coordinator.TryRestoreActivePlayback(8));
        Assert.True(coordinator.TryRestoreActivePlayback(7));
        Assert.Equal(0.6f, device.Gain);
        Assert.True(coordinator.TryDuckActivePlayback(7, 0.15f, 0.6f));

        var second = await coordinator.EnqueueAsync(8, TextStream("second"), CancellationToken.None);
        releaseFirst.TrySetResult();
        await secondStarted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.False(coordinator.TryDuckActivePlayback(7, 0.15f, 0.6f));
        Assert.False(coordinator.TryRestoreActivePlayback(7));
        Assert.Equal(new[] { 0.15f, 0.6f, 0.15f, 0.6f }, device.GainChanges);
        Assert.Equal(0, device.ClearCount);
        releaseSecond.TrySetResult();
        await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(2));
    }

    [Fact]
    public async Task ConfirmedDuckingRestoresGainOnlyAfterDeviceAudioIsCleared()
    {
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var device = new BlockingDeviceQueue();
        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (_, token) =>
        {
            started.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return string.Empty;
        }), device);
        _ = coordinator.Start(CancellationToken.None);
        var request = await coordinator.EnqueueAsync(7, TextStream("active"), CancellationToken.None);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.True(coordinator.TryDuckActivePlayback(7, 0.15f, 1f));

        Task stop = Task.Run(() => coordinator.TryStopActivePlayback(7));
        try
        {
            await device.Entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.False(coordinator.TryRestoreActivePlayback(7));
            Assert.Equal(0.15f, device.Gain);
            Assert.False(request.Completion.IsCompleted);
        }
        finally
        {
            device.Release.TrySetResult();
        }
        await stop.WaitAsync(TimeSpan.FromSeconds(2));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => request.Completion.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(1f, device.Gain);
    }

    [Fact]
    public async Task RequestsAreExecutedInAdmissionOrderAndNeverOverlap()
    {
        var started = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var order = new ConcurrentQueue<string>();
        int active = 0;
        int maxActive = 0;

        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            string text = await ReadTextAsync(request.Text, cancellationToken);
            order.Enqueue(text);
            int current = Interlocked.Increment(ref active);
            InterlockedExtensions.Max(ref maxActive, current);
            try
            {
                if (text == "first")
                    started.TrySetResult(text);
                await release.Task.WaitAsync(cancellationToken);
                return text;
            }
            finally
            {
                Interlocked.Decrement(ref active);
            }
        }), new CountingDeviceQueue());

        _ = coordinator.Start(CancellationToken.None);
        var first = await coordinator.EnqueueAsync(null, TextStream("first"), CancellationToken.None);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(1));
        var second = await coordinator.EnqueueAsync(null, TextStream("second"), CancellationToken.None);

        await Task.Delay(50);
        Assert.DoesNotContain("second", order);

        release.TrySetResult();
        await Task.WhenAll(first.Completion, second.Completion);

        Assert.Equal(["first", "second"], order.ToArray());
        Assert.Equal(1, maxActive);
    }

    [Fact]
    public async Task HeadRequestsRunBeforeWaitingTailRequestsWithoutPreemptingActiveRequest()
    {
        var activeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var order = new ConcurrentQueue<string>();

        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            string text = await ReadTextAsync(request.Text, cancellationToken);
            order.Enqueue(text);
            if (text == "active")
            {
                activeStarted.TrySetResult();
                await release.Task.WaitAsync(cancellationToken);
            }

            return text;
        }), new CountingDeviceQueue());

        _ = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(
            null,
            TextStream("active"),
            CancellationToken.None);
        await activeStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        var tail = await coordinator.EnqueueAsync(
            null,
            TextStream("tail"),
            CancellationToken.None);
        var head = await coordinator.EnqueueAsync(
            null,
            TextStream("head"),
            CancellationToken.None,
            SpeechPlaybackQueuePosition.Head);

        Assert.Equal(["active"], order.ToArray());
        release.TrySetResult();
        await Task.WhenAll(active.Completion, head.Completion, tail.Completion);

        Assert.Equal(["active", "head", "tail"], order.ToArray());
    }

    [Fact]
    public async Task StoppingActiveRequestLeavesPendingRequestForTheWorker()
    {
        var activeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var activeRelease = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var nextStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var deviceQueue = new CountingDeviceQueue();

        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            string text = await ReadTextAsync(request.Text, cancellationToken);
            if (text == "active")
            {
                activeStarted.TrySetResult();
                await activeRelease.Task.WaitAsync(cancellationToken);
            }
            else
            {
                nextStarted.TrySetResult();
            }

            return text;
        }), deviceQueue);

        _ = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(null, TextStream("active"), CancellationToken.None);
        await activeStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        var pending = await coordinator.EnqueueAsync(null, TextStream("pending"), CancellationToken.None);

        Assert.True(coordinator.TryStopActivePlayback());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
        await nextStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        await pending.Completion;
        Assert.Equal(1, deviceQueue.ClearCount);
        Assert.False(coordinator.TryStopActivePlayback());
    }

    [Fact]
    public async Task OwnerScopedStop_IsIdempotentPerRequestAndIgnoresOtherOwners()
    {
        var activeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var deviceQueue = new CountingDeviceQueue();
        using var coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (request, cancellationToken) =>
            {
                string text = await ReadTextAsync(request.Text, cancellationToken);
                activeStarted.TrySetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                return text;
            }),
            deviceQueue);

        _ = coordinator.Start(CancellationToken.None);
        SpeechEnqueueResult active = await coordinator.EnqueueAsync(7, TextStream("active"), CancellationToken.None);
        await activeStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal(7, coordinator.ActiveOwnerTurnId);

        // A stale decision for another owner never touches this reply.
        Assert.Equal(PlaybackStopOutcome.NotOwned, coordinator.TryStopActivePlayback(6));
        Assert.Equal(0, deviceQueue.ClearCount);

        // The Kernel's StopPlayback and ClearPlaybackQueue effects both reach this gate; the device
        // is cleared exactly once.
        Assert.Equal(PlaybackStopOutcome.Stopped, coordinator.TryStopActivePlayback(7));
        Assert.Equal(PlaybackStopOutcome.AlreadyStopped, coordinator.TryStopActivePlayback(7));
        Assert.Equal(1, deviceQueue.ClearCount);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
    }

    [Fact]
    public async Task CancelledRequest_ClearsItsQueuedDeviceAudioOnce_AndLateOwnerStopIsANoOp()
    {
        var activeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var deviceQueue = new CountingDeviceQueue();
        using var coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (request, cancellationToken) =>
            {
                string text = await ReadTextAsync(request.Text, cancellationToken);
                activeStarted.TrySetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                return text;
            }),
            deviceQueue);

        _ = coordinator.Start(CancellationToken.None);
        using var responseCts = new CancellationTokenSource();
        SpeechEnqueueResult active = await coordinator.EnqueueAsync(9, TextStream("active"), responseCts.Token);
        await activeStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));

        // The response source is cancelled (Kernel CancelResponse for this owner) before any
        // owner-scoped stop reached the coordinator: the PCM already handed to the device must not
        // keep sounding.
        responseCts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
        Assert.Equal(1, deviceQueue.ClearCount);

        // The Kernel's stop effects for the same owner arrive afterwards: nothing to do, but they
        // are recognised as this owner's stop rather than a foreign one.
        Assert.Equal(PlaybackStopOutcome.AlreadyStopped, coordinator.TryStopActivePlayback(9));
        Assert.Equal(PlaybackStopOutcome.NotOwned, coordinator.TryStopActivePlayback(8));
        Assert.Equal(1, deviceQueue.ClearCount);
    }

    [Fact]
    public async Task DeviceClearRunsOutsideCoordinatorGateAndBlocksTheNextRequest()
    {
        var activeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var pendingStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var deviceQueue = new BlockingDeviceQueue();
        using var coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (request, cancellationToken) =>
            {
                string text = await ReadTextAsync(request.Text, cancellationToken);
                if (text == "active")
                {
                    activeStarted.TrySetResult();
                    await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                }
                else
                {
                    pendingStarted.TrySetResult();
                }

                return text;
            }),
            deviceQueue);

        _ = coordinator.Start(CancellationToken.None);
        SpeechEnqueueResult active = await coordinator.EnqueueAsync(
            null,
            TextStream("active"),
            CancellationToken.None);
        await activeStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        SpeechEnqueueResult pending = await coordinator.EnqueueAsync(
            null,
            TextStream("pending"),
            CancellationToken.None);

        Task<bool> stop = Task.Run(coordinator.TryStopActivePlayback);
        await deviceQueue.Entered.Task.WaitAsync(TimeSpan.FromSeconds(1));

        // Admission still acquires the coordinator gate while the provider is blocked. The next
        // executor cannot start until the device-clear completion barrier is released.
        SpeechEnqueueResult late = await coordinator.EnqueueAsync(
            null,
            TextStream("late"),
            CancellationToken.None).AsTask().WaitAsync(TimeSpan.FromSeconds(1));
        Assert.False(pendingStarted.Task.IsCompleted);

        deviceQueue.Release.TrySetResult();
        Assert.True(await stop.WaitAsync(TimeSpan.FromSeconds(1)));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
        await Task.WhenAll(pending.Completion, late.Completion).WaitAsync(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task DuplicateTurnAdmissionIsRejectedUntilTheFirstRequestCompletes()
    {
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            await release.Task.WaitAsync(cancellationToken);
            return await ReadTextAsync(request.Text, cancellationToken);
        }), new CountingDeviceQueue());

        _ = coordinator.Start(CancellationToken.None);
        var first = await coordinator.EnqueueAsync(7, TextStream("first"), CancellationToken.None);

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await coordinator.EnqueueAsync(7, TextStream("duplicate"), CancellationToken.None));

        release.TrySetResult();
        await first.Completion;

        var next = await coordinator.EnqueueAsync(7, TextStream("next"), CancellationToken.None);
        await next.Completion;
    }

    [Fact]
    public async Task CompletionContinuationCanRequeueTheSameTurnWithoutAdmissionRace()
    {
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            await release.Task.WaitAsync(cancellationToken);
            return await ReadTextAsync(request.Text, cancellationToken);
        }), new CountingDeviceQueue());

        _ = coordinator.Start(CancellationToken.None);
        var first = await coordinator.EnqueueAsync(11, TextStream("first"), CancellationToken.None);
        var continuation = first.Completion.ContinueWith(
            _ => coordinator.EnqueueAsync(11, TextStream("second"), CancellationToken.None).AsTask(),
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default).Unwrap();

        release.TrySetResult();
        var second = await continuation.WaitAsync(TimeSpan.FromSeconds(1));
        await second.Completion;
    }

    [Fact]
    public async Task StopCancelsActiveAndPendingRequests()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var coordinator = new SpeechPlaybackCoordinator(new DelegateRequestExecutor(async (request, cancellationToken) =>
        {
            entered.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            return string.Empty;
        }), new CountingDeviceQueue());

        _ = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(null, TextStream("active"), CancellationToken.None);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(1));
        var pending = await coordinator.EnqueueAsync(null, TextStream("pending"), CancellationToken.None);

        await coordinator.StopAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await pending.Completion);
    }

    [Fact]
    public async Task StopCancellationCallbacks_RunOutsideCoordinatorGate()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        bool acquiredGate = false;
        SpeechPlaybackCoordinator? coordinator = null;
        using var lifetime = coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (_, token) =>
            {
                using var registration = token.Register(() =>
                {
                    // Bound the probe so the pre-fix regression cannot hang the test process.
                    Task probe = Task.Run(() => { long? owner = coordinator!.ActiveOwnerTurnId; });
                    acquiredGate = probe.Wait(TimeSpan.FromSeconds(2));
                });
                entered.TrySetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, token);
                return string.Empty;
            }), new CountingDeviceQueue());
        _ = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(1, TextStream("active"), CancellationToken.None);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));

        await coordinator.StopAsync().WaitAsync(TimeSpan.FromSeconds(5));

        Assert.True(acquiredGate);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => active.Completion);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExplicitStopDeviceFailure_ClosesQueueRegardlessOfExecutorExitOrder(bool exitBeforeClear)
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var failure = new InvalidOperationException("device clear failed");
        int executionCount = 0;
        SpeechPlaybackCoordinator? coordinator = null;
        using var lifetime = coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (_, token) =>
            {
                Interlocked.Increment(ref executionCount);
                entered.TrySetResult();
                await release.Task;
                token.ThrowIfCancellationRequested();
                return string.Empty;
            }), new DelegateDeviceQueue(() =>
            {
                if (exitBeforeClear)
                {
                    release.TrySetResult();
                    Assert.True(SpinWait.SpinUntil(
                        () => coordinator!.ActiveOwnerTurnId is null, TimeSpan.FromSeconds(2)));
                }
                throw failure;
            }));
        Task worker = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(1, TextStream("active"), CancellationToken.None);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        var pending = await coordinator.EnqueueAsync(2, TextStream("pending"), CancellationToken.None);

        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() => coordinator.TryStopActivePlayback(1)));
        release.TrySetResult();

        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => active.Completion.WaitAsync(TimeSpan.FromSeconds(2))));
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => worker.WaitAsync(TimeSpan.FromSeconds(2))));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.Completion);
        Assert.Equal(1, executionCount);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await coordinator.EnqueueAsync(3, TextStream("late"), CancellationToken.None));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ThrowingCancellationCallback_DoesNotSkipDeviceOrShutdownCleanup(bool shutdown)
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var device = new CountingDeviceQueue();
        var failure = new InvalidOperationException("provider cancellation callback failed");
        using var coordinator = new SpeechPlaybackCoordinator(
            new DelegateRequestExecutor(async (request, token) =>
            {
                if (request.OwnerTurnId == 1)
                {
                    using var registration = token.Register(() => throw failure);
                    entered.TrySetResult();
                    await Task.Delay(Timeout.InfiniteTimeSpan, token);
                }
                return "done";
            }), device);
        _ = coordinator.Start(CancellationToken.None);
        var active = await coordinator.EnqueueAsync(1, TextStream("active"), CancellationToken.None);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        var pending = await coordinator.EnqueueAsync(2, TextStream("pending"), CancellationToken.None);

        Exception? error = shutdown
            ? await Record.ExceptionAsync(() => coordinator.StopAsync().WaitAsync(TimeSpan.FromSeconds(2)))
            : Record.Exception(() => coordinator.TryStopActivePlayback(1));
        Assert.Contains(failure, Assert.IsType<AggregateException>(error).Flatten().InnerExceptions);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => active.Completion.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(1, device.ClearCount);
        if (shutdown)
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending.Completion);
            _ = coordinator.Start(CancellationToken.None);
            var next = await coordinator.EnqueueAsync(3, TextStream("restart"), CancellationToken.None);
            await next.Completion.WaitAsync(TimeSpan.FromSeconds(2));
        }
        else
        {
            await pending.Completion.WaitAsync(TimeSpan.FromSeconds(2));
        }
    }

    private sealed class DelegateDeviceQueue(Action clear) : IPlaybackDeviceQueuePort
    {
        public void ClearQueue() => clear();

        public void SetGain(float gain) { }
    }

    private static async Task<string> ReadTextAsync(
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken cancellationToken)
    {
        await foreach (SpeakableTextDelta delta in text.WithCancellation(cancellationToken))
            return delta.Text;
        return string.Empty;
    }

    private static async IAsyncEnumerable<SpeakableTextDelta> TextStream(string text)
    {
        yield return new SpeakableTextDelta(text);
        await Task.CompletedTask;
    }

    private static class InterlockedExtensions
    {
        public static void Max(ref int location, int value)
        {
            while (true)
            {
                int current = Volatile.Read(ref location);
                if (current >= value)
                    return;
                if (Interlocked.CompareExchange(ref location, value, current) == current)
                    return;
            }
        }
    }
}
