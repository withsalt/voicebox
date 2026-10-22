using Microsoft.Extensions.Logging.Abstractions;
using System.Diagnostics;
using System.Reflection;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Runtime.Publications;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.Publications.Models;
using Xunit;
using Xunit.Abstractions;

namespace VoiceBox.Assistant.Tests;

public sealed class OrderedPublicationBoundaryTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FaultedWorker_ReleasesUndeliverableEventsAndRejectsNewPublication(bool checkAdmission)
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var boundary = new OrderedPublicationBoundary(dispatcher, gapTimeout: TimeSpan.FromMilliseconds(20));
        boundary.ReserveSequence();
        long second = boundary.ReserveSequence();
        var publication = new AssistantTextPublication(new AssistantTextEventArgs { Delta = "retained", IsThinking = false });
        boundary.PublishCommitted(second, publication);
        try
        {
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => boundary.Completion.WaitAsync(TimeSpan.FromSeconds(2)));
            if (checkAdmission)
            {
                long next = boundary.NextPublicationSequence;
                Assert.Throws<InvalidOperationException>(() => boundary.PublishStream(publication));
                Assert.Throws<InvalidOperationException>(() => boundary.PublishCommitted(second, publication));
                Assert.Throws<InvalidOperationException>(() => boundary.ReserveSequence());
                Assert.Equal(next, boundary.NextPublicationSequence);
            }
            else
            {
                var pending = (SortedDictionary<long, AssistantPublication>)typeof(OrderedPublicationBoundary)
                    .GetField("_pending", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(boundary)!;
                Assert.Empty(pending);
            }
        }
        finally
        {
            await Assert.ThrowsAsync<InvalidOperationException>(() => boundary.DisposeAsync().AsTask());
        }
    }

    [Fact]
    public async Task TextDeliveryWithoutDrainWaiters_DoesNotAllocateProgressSourcesPerEvent()
    {
        const int count = 10_000;
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var delivered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var release = new ManualResetEventSlim();
        int received = 0;
        long before = 0;
        long allocated = 0;
        dispatcher.AssistantText += (_, _) =>
        {
            int current = ++received;
            if (current == 1)
            {
                entered.TrySetResult();
                release.Wait();
            }
            if (current == 1_002)
                before = GC.GetAllocatedBytesForCurrentThread();
            if (current == count + 1_001)
            {
                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                delivered.TrySetResult();
            }
        };
        await using var boundary = new OrderedPublicationBoundary(dispatcher);
        var publication = new AssistantTextPublication(new AssistantTextEventArgs { Delta = "delta", IsThinking = false });
        boundary.PublishStream(publication);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            for (int i = 0; i < count + 1_000; i++)
                boundary.PublishStream(publication);
        }
        finally
        {
            release.Set();
        }
        // This completion comes from the subscriber, so no Drain waiter needs progress sources
        // while the worker delivers the measured events.
        await delivered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(count + 1_001, received);
        output.WriteLine($"Publication delivery thread allocated {allocated} bytes across {count - 1} measured intervals.");
        Assert.InRange(allocated, 0, 4_096);
        await boundary.DrainAsync().WaitAsync(TimeSpan.FromSeconds(2));
    }

    [Fact]
    public async Task DeferredPublicationWorker_CanDisposeBeforeStart()
    {
        var boundary = new OrderedPublicationBoundary(
            new AssistantEventDispatcher(new object(), NullLogger.Instance), startWorker: false);
        Assert.False(boundary.Completion.IsCompleted);
        Assert.Equal(0, boundary.PublisherThreadId);
        await boundary.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));
        Assert.True(boundary.Completion.IsCompletedSuccessfully);
    }

    [Fact]
    public async Task SharedProgressSource_ReleasesAllConcurrentWaitersInOrder()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        int delivered = 0;
        dispatcher.AssistantText += (_, _) => delivered++;
        await using var boundary = new OrderedPublicationBoundary(dispatcher, gapTimeout: TimeSpan.FromSeconds(2));
        var publication = new AssistantTextPublication(new AssistantTextEventArgs { Delta = "delta", IsThinking = false });
        long[] sequences = Enumerable.Range(0, 32).Select(_ => boundary.ReserveSequence()).ToArray();
        Task[] waiters = sequences.Select(sequence => boundary.WaitUntilDeliveredAsync(sequence)).ToArray();
        foreach (long sequence in sequences.Reverse())
            boundary.PublishCommitted(sequence, publication);
        await Task.WhenAll(waiters).WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(sequences.Length, delivered);
        Assert.Equal(sequences[^1], boundary.DeliveredSequence);
    }

    [Fact]
    public async Task OutOfOrderArrival_ReleasesStrictlyByPublicationSequenceOnOneContext()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var observed = new List<(AssistantState State, int Thread)>();
        dispatcher.StateChanged += (_, value) =>
        {
            lock (observed)
                observed.Add((value.To, Environment.CurrentManagedThreadId));
        };
        await using var boundary = new OrderedPublicationBoundary(dispatcher);
        long first = boundary.ReserveSequence();
        long second = boundary.ReserveSequence();

        boundary.PublishCommitted(second, new StateChangedPublication(AssistantState.Listening, AssistantState.Recognizing));
        boundary.PublishCommitted(first, new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));
        await boundary.DrainAsync().WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Equal([AssistantState.Listening, AssistantState.Recognizing], observed.Select(static value => value.State));
        Assert.Single(observed.Select(static value => value.Thread).Distinct());
        Assert.Equal(boundary.PublisherThreadId, observed[0].Thread);
    }

    [Fact]
    public async Task SubscriberBlock_DoesNotBlockProducerAndDrainWaitsForRelease()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        dispatcher.StateChanged += (_, _) =>
        {
            entered.TrySetResult();
            release.Task.GetAwaiter().GetResult();
        };
        await using var boundary = new OrderedPublicationBoundary(dispatcher);

        long before = Stopwatch.GetTimestamp();
        boundary.PublishStream(new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));
        TimeSpan producerElapsed = Stopwatch.GetElapsedTime(before);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(1));
        Task drain = boundary.DrainAsync();

        Assert.False(drain.IsCompleted);
        Assert.True(producerElapsed < TimeSpan.FromMilliseconds(20));
        release.TrySetResult();
        await drain.WaitAsync(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task ReservedSequenceGap_FaultsInsteadOfSkippingOrWaitingForever()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var boundary = new OrderedPublicationBoundary(
            dispatcher,
            gapTimeout: TimeSpan.FromMilliseconds(20));
        _ = boundary.ReserveSequence();
        long second = boundary.ReserveSequence();
        boundary.PublishCommitted(second, new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => boundary.Completion.WaitAsync(TimeSpan.FromSeconds(1)));
        Assert.Contains("gap at 1", error.Message, StringComparison.Ordinal);
        await Assert.ThrowsAsync<InvalidOperationException>(() => boundary.DisposeAsync().AsTask());
    }

    [Fact]
    public async Task Drain_PropagatesWorkerGapFaultInsteadOfWaitingForProgressForever()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var boundary = new OrderedPublicationBoundary(
            dispatcher,
            gapTimeout: TimeSpan.FromMilliseconds(20));
        _ = boundary.ReserveSequence();
        long second = boundary.ReserveSequence();
        boundary.PublishCommitted(second, new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => boundary.DrainAsync().WaitAsync(TimeSpan.FromSeconds(1)));

        Assert.Contains("gap at 1", error.Message, StringComparison.Ordinal);
        await Assert.ThrowsAsync<InvalidOperationException>(() => boundary.DisposeAsync().AsTask());
    }

    [Fact]
    public async Task Drain_PreservesFacadeSenderAndPerSubscriberFaultIsolation()
    {
        var sender = new object();
        var dispatcher = new AssistantEventDispatcher(sender, NullLogger.Instance);
        object? observedSender = null;
        dispatcher.StateChanged += (_, _) => throw new InvalidOperationException("subscriber");
        dispatcher.StateChanged += (actual, _) => observedSender = actual;
        await using var boundary = new OrderedPublicationBoundary(dispatcher);

        boundary.PublishStream(new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));
        await boundary.DrainAsync().WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Same(sender, observedSender);
        Assert.Equal(1, boundary.DeliveredSequence);
        Assert.Equal(2, boundary.NextPublicationSequence);
    }

    [Fact]
    public async Task InvalidCommittedSequence_IsRejectedWithoutReservingOrSkipping()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        await using var boundary = new OrderedPublicationBoundary(dispatcher);
        var publication = new StateChangedPublication(AssistantState.Idle, AssistantState.Listening);

        Assert.Throws<ArgumentOutOfRangeException>(() => boundary.PublishCommitted(0, publication));
        Assert.Throws<ArgumentOutOfRangeException>(() => boundary.PublishCommitted(-1, publication));
        Assert.Throws<ArgumentOutOfRangeException>(() => boundary.PublishCommitted(1, publication));
        Assert.Equal(1, boundary.ReserveSequence());
        Assert.Throws<ArgumentOutOfRangeException>(() => boundary.PublishCommitted(2, publication));
        boundary.PublishCommitted(1, publication);
        await boundary.DrainAsync().WaitAsync(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task WaitUntilDelivered_AtCurrentSequenceReturnsImmediately()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        await using var boundary = new OrderedPublicationBoundary(dispatcher);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => boundary.WaitUntilDeliveredAsync(-1));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => boundary.WaitUntilDeliveredAsync(1));
        await boundary.WaitUntilDeliveredAsync(0).WaitAsync(TimeSpan.FromSeconds(1));

        long sequence = boundary.PublishStream(
            new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));
        await boundary.WaitUntilDeliveredAsync(sequence).WaitAsync(TimeSpan.FromSeconds(1));
        await boundary.WaitUntilDeliveredAsync(sequence).WaitAsync(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task DisposeWithUnpublishedReservation_ReportsFinalDrainGap()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var boundary = new OrderedPublicationBoundary(dispatcher);
        _ = boundary.ReserveSequence();

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => boundary.DisposeAsync().AsTask());
        Assert.Contains("gap at 1 during drain", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GapDeadlineUsesInjectedClockAcrossUnrelatedPublicationWakeups()
    {
        var dispatcher = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var time = new ManualTimerTimeProvider();
        time.Advance(TimeSpan.FromTicks(1));
        var boundary = new OrderedPublicationBoundary(
            dispatcher,
            timeProvider: time,
            gapTimeout: TimeSpan.FromMilliseconds(20));
        _ = boundary.ReserveSequence();
        long second = boundary.ReserveSequence();
        long third = boundary.ReserveSequence();

        boundary.PublishCommitted(second, new StateChangedPublication(AssistantState.Idle, AssistantState.Listening));
        await time.WaitForTimerCountAsync(1).WaitAsync(TimeSpan.FromSeconds(1));
        time.Advance(TimeSpan.FromMilliseconds(10));
        boundary.PublishCommitted(third, new StateChangedPublication(AssistantState.Listening, AssistantState.Recognizing));
        await time.WaitForTimerCountAsync(2).WaitAsync(TimeSpan.FromSeconds(1));
        time.Advance(TimeSpan.FromMilliseconds(10));

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => boundary.Completion.WaitAsync(TimeSpan.FromSeconds(1)));
        Assert.Contains("gap at 1", error.Message, StringComparison.Ordinal);
    }
}
