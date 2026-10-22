using System.Collections.Concurrent;
using System.Diagnostics;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AssistantControlPlaneTests
{
    [Fact]
    public void CriticalLane_CapacityMinusOneDoesNotFault()
    {
        using var ingress = new AssistantControlIngress();

        for (int index = 0; index < AssistantControlIngress.CriticalCapacity - 1; index++)
        {
            Assert.True(ingress.TryEnqueueCritical(
                AssistantSignal.SpeechStarted(enableBargeIn: true),
                index + 1,
                AssistantSignalSource.CaptureCallback,
                out _));
        }

        Assert.False(ingress.HasTerminalFault);
    }

    [Fact]
    public void CriticalEnvelope_PreservesIngressSequenceAndSourceTimestampOnSignal()
    {
        using var ingress = new AssistantControlIngress();
        const long sourceTimestamp = 123456;

        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.InterruptionRequested(1, 2),
            sourceTimestamp,
            AssistantSignalSource.DuplexCallback,
            out long sequence));

        Assert.True(ingress.TryRead(out ControlEnvelope envelope));
        Assert.Equal(sequence, envelope.Signal.SourceSequence);
        Assert.Equal(sourceTimestamp, envelope.Signal.SourceTimestamp);
    }

    [Fact]
    public async Task PostBarrierCriticalSignal_IsRepresentedByAnOrderedAbandonEnvelope()
    {
        using var ingress = new AssistantControlIngress();
        long barrier = await ingress.EnqueueAsync(
            new RuntimeCommand(RuntimeCommandKind.Stop, 1),
            sourceTimestamp: 1);

        Assert.False(ingress.TryEnqueueCritical(
            AssistantSignal.InterruptionRequested(1, 2),
            sourceTimestamp: 2,
            AssistantSignalSource.DuplexCallback,
            out long abandonedSequence));

        Assert.True(ingress.TryRead(out ControlEnvelope command));
        Assert.Equal(barrier, command.IngressSequence);
        Assert.True(ingress.TryRead(out ControlEnvelope abandoned));
        Assert.Equal(abandonedSequence, abandoned.IngressSequence);
        Assert.True(ingress.IsAfterBarrier(abandoned));
    }

    [Fact]
    public void CriticalLane_CapacityAndOverflowProduceOneTerminalFaultWithoutDroppingAcceptedSignals()
    {
        using var ingress = new AssistantControlIngress();
        var accepted = new List<long>();

        for (int index = 0; index < AssistantControlIngress.CriticalCapacity; index++)
        {
            Assert.True(ingress.TryEnqueueCritical(
                AssistantSignal.SpeechStarted(enableBargeIn: true),
                index + 1,
                AssistantSignalSource.CaptureCallback,
                out long sequence));
            accepted.Add(sequence);
        }

        Assert.False(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechStarted(enableBargeIn: true),
            AssistantControlIngress.CriticalCapacity + 1,
            AssistantSignalSource.CaptureCallback,
            out _));
        Assert.True(ingress.HasTerminalFault);

        var drained = new List<ControlEnvelope>();
        while (ingress.TryRead(out ControlEnvelope envelope))
            drained.Add(envelope);

        Assert.Equal(accepted, drained.Take(accepted.Count).Select(static value => value.IngressSequence));
        ControlEnvelope fault = Assert.Single(drained, static value => value.Kind == ControlEnvelopeKind.TerminalFault);
        Assert.Equal("VBX-CTRL-001", fault.Failure.Code);
        Assert.Equal(AssistantControlIngress.CriticalCapacity + 1, fault.IngressSequence);
    }

    [Fact]
    public void BoundedQueue_RejectsZeroCapacityAndDisposeIsIdempotent()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BoundedMpscQueue<int>(0));

        var queue = new BoundedMpscQueue<int>(1);
        queue.Dispose();
        queue.Dispose();
        Assert.Throws<ObjectDisposedException>(() => queue.TryWrite(1));
    }

    [Fact]
    public void Ingress_DisposedAndTerminalAdmissionAreBothRejectedWithoutThrowing()
    {
        var disposed = new AssistantControlIngress();
        disposed.Dispose();
        disposed.Dispose();
        Assert.False(disposed.TryTerminate(
            new AssistantFailure("VBX-CTRL-TEST", "lifecycle", "test", "disposed"),
            1,
            out _));
        Assert.False(disposed.TryEnqueueCritical(
            AssistantSignal.SpeechStarted(enableBargeIn: false),
            1,
            AssistantSignalSource.CaptureCallback,
            out _));

        using var terminal = new AssistantControlIngress();
        Assert.True(terminal.TryTerminate(
            new AssistantFailure("VBX-CTRL-TEST", "fault", "test", "terminal"),
            1,
            out _));
        Assert.False(terminal.TryTerminate(
            new AssistantFailure("VBX-CTRL-TEST", "fault", "test", "duplicate"),
            2,
            out _));
        Assert.False(terminal.TryEnqueueCritical(
            AssistantSignal.SpeechEnded(1, TimeSpan.Zero),
            2,
            AssistantSignalSource.CaptureCallback,
            out _));
    }

    [Fact]
    public void OperationFailed_RejectsNonFailureSignalKinds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => AssistantSignal.OperationFailed(
            AssistantSignalKind.SpeechStarted,
            1,
            1,
            new AssistantFailure("VBX-CTRL-TEST", "operation", "test", "invalid-kind")));
    }

    [Fact]
    public void AssistantSignal_FactoryPreservesAllConstructorFields()
    {
        AssistantSignal signal = AssistantSignal.RecognitionCompleted(
            turnId: 42,
            operationId: 99,
            text: "recognized",
            speechDuration: TimeSpan.FromMilliseconds(12));

        Assert.Equal(AssistantSignalKind.RecognitionCompleted, signal.Kind);
        Assert.Equal(42, signal.TurnId);
        Assert.Equal(99, signal.OperationId);
        Assert.Equal(TimeSpan.FromMilliseconds(12).Ticks, signal.DurationTicks);
        Assert.False(signal.Flag);
        Assert.Equal(default, signal.State);
        Assert.Equal("recognized", signal.Text);
        Assert.Null(signal.Reason);
        Assert.Equal(default(AssistantFailure), signal.Failure);
    }

    [Fact]
    public void OperationFailed_AcceptsEveryFailureSignalKind()
    {
        AssistantSignalKind[] kinds =
        [
            AssistantSignalKind.RecognitionFailed,
            AssistantSignalKind.ResponseFailed,
            AssistantSignalKind.PlaybackFailed,
            AssistantSignalKind.RuntimeFault,
        ];

        foreach (AssistantSignalKind kind in kinds)
        {
            AssistantSignal signal = AssistantSignal.OperationFailed(
                kind,
                turnId: 1,
                operationId: 2,
                new AssistantFailure("VBX-CTRL-TEST", "operation", "test", "failure"));
            Assert.Equal(kind, signal.Kind);
        }
    }

    [Fact]
    public async Task TwoLanes_MergeByGlobalSequenceAndRejectTurnSignalsAfterStopBarrier()
    {
        using var ingress = new AssistantControlIngress();
        var handler = new RecordingHandler(expectedHandled: 2);
        var loop = new AssistantControlLoop(ingress, handler);
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        Task run = loop.Start(stop.Token);

        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechStarted(enableBargeIn: false),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.CaptureCallback,
            out long speechSequence));
        long stopSequence = await ingress.EnqueueAsync(
            new RuntimeCommand(RuntimeCommandKind.Stop),
            Stopwatch.GetTimestamp(),
            stop.Token);
        Assert.False(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechEnded(1, TimeSpan.FromMilliseconds(20)),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.CaptureCallback,
            out long rejectedSequence));

        await handler.ExpectedHandled.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal([speechSequence, stopSequence], handler.Sequences);
        Assert.Equal(stopSequence + 1, rejectedSequence);
        Assert.Equal(stopSequence, ingress.BarrierSequence);

        stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
    }

    [Fact]
    public async Task FatalAndStop_AreConsumedOnceInIngressOrder()
    {
        using var ingress = new AssistantControlIngress();
        var handler = new RecordingHandler(expectedHandled: 2);
        var loop = new AssistantControlLoop(ingress, handler);

        long stopSequence = await ingress.EnqueueAsync(
            new RuntimeCommand(RuntimeCommandKind.Stop, 1),
            1);
        Assert.True(ingress.TryTerminate(
            new AssistantFailure("VBX-CTRL-TEST", "fault", "test", "fatal-plus-stop"),
            2,
            out long faultSequence));

        await loop.Start(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Equal(stopSequence + 1, faultSequence);
        Assert.Equal([stopSequence], handler.Sequences);
        Assert.Equal("VBX-CTRL-TEST", Assert.Single(handler.Faults).Code);
    }

    [Fact]
    public async Task BoundedMpscQueue_ConcurrentProducersPreserveEveryAcceptedItem()
    {
        const int ProducerCount = 4;
        const int ItemsPerProducer = 10_000;
        var queue = new BoundedMpscQueue<long>(256);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        int completedProducers = 0;
        var consumed = new HashSet<long>();

        Task consumer = Task.Run(() =>
        {
            while (Volatile.Read(ref completedProducers) != ProducerCount ||
                   consumed.Count != ProducerCount * ItemsPerProducer)
            {
                timeout.Token.ThrowIfCancellationRequested();
                if (queue.TryRead(out long value))
                    Assert.True(consumed.Add(value), $"Duplicate value {value}.");
                else
                    Thread.Yield();
            }
        }, timeout.Token);

        Task[] producers = Enumerable.Range(0, ProducerCount)
            .Select(producer => Task.Run(() =>
            {
                try
                {
                    for (int index = 0; index < ItemsPerProducer; index++)
                    {
                        long value = ((long)producer << 32) | (uint)index;
                        while (!queue.TryWrite(value))
                        {
                            timeout.Token.ThrowIfCancellationRequested();
                            Thread.Yield();
                        }
                    }
                }
                finally
                {
                    Interlocked.Increment(ref completedProducers);
                }
            }, timeout.Token))
            .ToArray();

        await Task.WhenAll(producers).WaitAsync(timeout.Token);
        await consumer.WaitAsync(timeout.Token);
        Assert.Equal(ProducerCount * ItemsPerProducer, consumed.Count);
    }

    [Fact]
    public async Task CompletionAndLifecycleLane_ReservesTwentyFourPlusEightAdmissions()
    {
        using var ingress = new AssistantControlIngress();

        for (int index = 0; index < 24; index++)
        {
            await ingress.EnqueueSignalAsync(
                AssistantSignal.RecognitionCompleted(index + 1, index + 1, "ok"),
                index + 1,
                AssistantSignalSource.Recognition);
        }
        for (int index = 0; index < 8; index++)
        {
            await ingress.EnqueueAsync(
                new RuntimeCommand(RuntimeCommandKind.Start, index + 1),
                index + 1);
        }

        var drained = new List<ControlEnvelope>();
        while (ingress.TryRead(out ControlEnvelope envelope))
            drained.Add(envelope);

        Assert.Equal(32, drained.Count);
        Assert.Equal(24, drained.Count(static value => value.Kind == ControlEnvelopeKind.AssistantSignal));
        Assert.Equal(8, drained.Count(static value => value.Kind == ControlEnvelopeKind.RuntimeCommand));
        Assert.Equal(Enumerable.Range(1, 32).Select(static value => (long)value),
            drained.Select(static value => value.IngressSequence));
    }

    [Fact]
    public async Task EqualSourceTimestampsAreAcceptedButDecreasingAsyncTimestampTerminatesIngress()
    {
        using var ingress = new AssistantControlIngress();

        await ingress.EnqueueSignalAsync(
            AssistantSignal.RecognitionCompleted(1, 1, "first"),
            10,
            AssistantSignalSource.Recognition);
        await ingress.EnqueueSignalAsync(
            AssistantSignal.RecognitionCompleted(1, 2, "second"),
            10,
            AssistantSignalSource.Recognition);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await ingress.EnqueueSignalAsync(
                AssistantSignal.RecognitionCompleted(1, 3, "stale"),
                9,
                AssistantSignalSource.Recognition));

        Assert.Contains("source timestamp", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.True(ingress.HasTerminalFault);
        var drained = new List<ControlEnvelope>();
        while (ingress.TryRead(out ControlEnvelope envelope))
            drained.Add(envelope);
        Assert.Equal([1L, 2L, 3L], drained.Select(static value => value.IngressSequence));
        Assert.Equal("VBX-CTRL-002", drained[^1].Failure.Code);
    }

    [Fact]
    public async Task SameTimestampDifferentLaneOrder_ReplaysTheIngressSequenceExactly()
    {
        const long Timestamp = 42;
        using var ingress = new AssistantControlIngress();
        var handler = new RecordingHandler(expectedHandled: 3);
        var loop = new AssistantControlLoop(ingress, handler);
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        Task run = loop.Start(stop.Token);

        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechStarted(enableBargeIn: false),
            Timestamp,
            AssistantSignalSource.CaptureCallback,
            out long first));
        long second = await ingress.EnqueueSignalAsync(
            AssistantSignal.RecognitionCompleted(1, 1, "ok"),
            Timestamp,
            AssistantSignalSource.Recognition,
            stop.Token);
        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechEnded(1, TimeSpan.Zero),
            Timestamp,
            AssistantSignalSource.CaptureCallback,
            out long third));

        await handler.ExpectedHandled.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal([first, second, third], handler.Sequences);

        stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
    }

    [Fact]
    public void CriticalProducer_SteadyStateTryEnqueueAndReadAllocateZeroBytes()
    {
        using var ingress = new AssistantControlIngress();
        AssistantSignal signal = AssistantSignal.SpeechStarted(enableBargeIn: false);
        Assert.True(ingress.TryEnqueueCritical(
            signal,
            1,
            AssistantSignalSource.DuplexCallback,
            out _));
        Assert.True(ingress.TryRead(out _));

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int index = 0; index < 1_000; index++)
        {
            Assert.True(ingress.TryEnqueueCritical(
                signal,
                index + 2,
                AssistantSignalSource.DuplexCallback,
                out _));
            Assert.True(ingress.TryRead(out _));
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Theory]
    [InlineData(160)]
    [InlineData(256)]
    [InlineData(320)]
    public void CriticalProducer_P99AndMaxStayInsideCallbackPeriodBudgets(
        int callbackSamples)
    {
        const int SampleRate = 16_000;
        const int Iterations = 1_000;
        using var ingress = new AssistantControlIngress();
        AssistantSignal signal = AssistantSignal.InterruptionRequested(1, 1);
        Assert.True(ingress.TryEnqueueCritical(
            signal,
            1,
            AssistantSignalSource.DuplexCallback,
            out _));
        Assert.True(ingress.TryRead(out _));

        var elapsedTicks = new long[Iterations];
        for (int index = 0; index < Iterations; index++)
        {
            long started = Stopwatch.GetTimestamp();
            if (!ingress.TryEnqueueCritical(
                    signal,
                    index + 2,
                    AssistantSignalSource.DuplexCallback,
                    out _) ||
                !ingress.TryRead(out _))
            {
                throw new Xunit.Sdk.XunitException("Critical ingress steady-state admission failed.");
            }
            elapsedTicks[index] = Stopwatch.GetTimestamp() - started;
        }

        Array.Sort(elapsedTicks);
        double callbackPeriodMilliseconds = callbackSamples * 1000d / SampleRate;
        double tickToMilliseconds = 1000d / Stopwatch.Frequency;
        int p99Index = (int)Math.Ceiling(Iterations * 0.99) - 1;
        double p99Milliseconds = elapsedTicks[p99Index] * tickToMilliseconds;
        double maxMilliseconds = elapsedTicks[^1] * tickToMilliseconds;

        Assert.True(
            p99Milliseconds <= callbackPeriodMilliseconds,
            $"{callbackSamples}-sample ingress p99 {p99Milliseconds:F3}ms exceeded " +
            $"its {callbackPeriodMilliseconds:F3}ms callback period.");
        Assert.True(
            maxMilliseconds <= callbackPeriodMilliseconds * 2,
            $"{callbackSamples}-sample ingress max {maxMilliseconds:F3}ms exceeded " +
            "two callback periods.");
    }

    [Theory]
    [InlineData(160, 0.0104)]
    [InlineData(256, 0.0171)]
    [InlineData(320, 0.0063)]
    public async Task InterruptionSourceToEffectLatency_AddsNoMoreThanOneCallbackPeriod(
        int callbackSamples,
        double maxBudgetMilliseconds)
    {
        const int SampleRate = 16_000;
        const int Iterations = 100;
        using var ingress = new AssistantControlIngress();
        using var handler = new LatencyHandler();
        var loop = new AssistantControlLoop(ingress, handler);
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task run = loop.Start(stop.Token);

        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.InterruptionRequested(1, 1),
            1,
            AssistantSignalSource.DuplexCallback,
            out long warmupSequence));
        await handler.WaitForSequenceAsync(warmupSequence, stop.Token);

        var elapsedTicks = new long[Iterations];
        for (int index = 0; index < Iterations; index++)
        {
            long started = Stopwatch.GetTimestamp();
            Assert.True(ingress.TryEnqueueCritical(
                AssistantSignal.InterruptionRequested(1, index + 2),
                index + 2,
                AssistantSignalSource.DuplexCallback,
                out long sequence));
            await handler.WaitForSequenceAsync(sequence, stop.Token);
            elapsedTicks[index] = Stopwatch.GetTimestamp() - started;
        }

        double tickToMilliseconds = 1000d / Stopwatch.Frequency;
        double maxMilliseconds = elapsedTicks.Max() * tickToMilliseconds;
        double callbackPeriodMilliseconds = callbackSamples * 1000d / SampleRate;
        Assert.True(
            maxMilliseconds <= maxBudgetMilliseconds + callbackPeriodMilliseconds,
            $"{callbackSamples}-sample interruption max {maxMilliseconds:F3}ms exceeded budget " +
            $"{maxBudgetMilliseconds:F4}ms plus one " +
            $"{callbackPeriodMilliseconds:F3}ms callback period.");

        stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
    }

    [Fact]
    public async Task ControlLoop_CriticalSignalIsConsumedByTheSingleNonAudioConsumer()
    {
        using var ingress = new AssistantControlIngress();
        using var handler = new ConsumerThreadCaptureHandler();
        var loop = new AssistantControlLoop(ingress, handler);
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        Task run = loop.Start(stop.Token);

        for (int index = 1; index <= 3; index++)
        {
            Assert.True(ingress.TryEnqueueCritical(
                AssistantSignal.InterruptionRequested(1, index),
                index,
                AssistantSignalSource.DuplexCallback,
                out long sequence));
            await handler.WaitForSequenceAsync(sequence, stop.Token);
        }

        // The consumer must be one dedicated non-audio thread that never changes and never
        // borrows the shared pool, so a real-time decision cannot queue behind pooled work.
        Assert.False(handler.HandledOnThreadPool);
        Assert.NotEqual(0, loop.ConsumerThreadId);
        int[] handledThreadIds;
        lock (handler.HandledThreadIds)
            handledThreadIds = [.. handler.HandledThreadIds];
        Assert.Equal(3, handledThreadIds.Length);
        Assert.Single(handledThreadIds.Distinct());
        Assert.Equal(loop.ConsumerThreadId, handledThreadIds[0]);
        Assert.NotEqual(Environment.CurrentManagedThreadId, handledThreadIds[0]);

        stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
    }

    [Fact]
    public async Task SequenceGap_ConvergesToOneTerminalFault()
    {
        var ingress = new FakeIngressReader(
            ControlEnvelope.FromSignal(
                2,
                2,
                AssistantSignalSource.Recognition,
                AssistantSignal.RecognitionCompleted(1, 1, "late")));
        var handler = new RecordingHandler(expectedHandled: 1);
        var loop = new AssistantControlLoop(
            ingress,
            handler,
            gapTimeout: TimeSpan.FromMilliseconds(20));

        await loop.Start(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Empty(handler.Sequences);
        AssistantFailure fault = Assert.Single(handler.Faults);
        Assert.Equal("VBX-CTRL-005", fault.Code);
    }

    [Fact]
    public async Task SequenceGap_UsesOneDeadlineAcrossUnrelatedWakeups()
    {
        var ingress = new DelayedWakeIngressReader(
            wakeDelay: TimeSpan.FromMilliseconds(15),
            wakeCount: 3,
            ControlEnvelope.FromSignal(
                2,
                2,
                AssistantSignalSource.Recognition,
                AssistantSignal.RecognitionCompleted(1, 1, "late")));
        var handler = new RecordingHandler(expectedHandled: 1);
        var loop = new AssistantControlLoop(
            ingress,
            handler,
            gapTimeout: TimeSpan.FromMilliseconds(30));
        long started = Stopwatch.GetTimestamp();

        await loop.Start(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(1));

        Assert.True(Stopwatch.GetElapsedTime(started) < TimeSpan.FromMilliseconds(60));
        Assert.Equal("VBX-CTRL-005", Assert.Single(handler.Faults).Code);
    }

    [Fact]
    public async Task SequenceGap_DeadlineUsesInjectedClockAcrossUnrelatedWakeups()
    {
        var time = new ManualTimerTimeProvider();
        time.Advance(TimeSpan.FromTicks(1));
        var ingress = new PulsedIngressReader(
            ControlEnvelope.FromSignal(
                2,
                2,
                AssistantSignalSource.Recognition,
                AssistantSignal.RecognitionCompleted(1, 1, "late")));
        var handler = new RecordingHandler(expectedHandled: 1);
        var loop = new AssistantControlLoop(
            ingress,
            handler,
            timeProvider: time,
            gapTimeout: TimeSpan.FromMilliseconds(20));
        Task run = loop.Start(CancellationToken.None);

        await time.WaitForTimerCountAsync(1).WaitAsync(TimeSpan.FromSeconds(1));
        time.Advance(TimeSpan.FromMilliseconds(10));
        ingress.Pulse();
        await time.WaitForTimerCountAsync(2).WaitAsync(TimeSpan.FromSeconds(1));
        time.Advance(TimeSpan.FromMilliseconds(10));

        await run.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal("VBX-CTRL-005", Assert.Single(handler.Faults).Code);
    }

    [Fact]
    public async Task ControlLoop_HandlerFaultIsConvertedToTerminalFailure()
    {
        var ingress = new FakeIngressReader(
            ControlEnvelope.FromSignal(
                1,
                1,
                AssistantSignalSource.Recognition,
                AssistantSignal.RecognitionCompleted(1, 1, "fault")));
        var handler = new ThrowingHandler(throwCancellation: false);
        var loop = new AssistantControlLoop(ingress, handler);

        await loop.Start(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Equal("VBX-CTRL-010", Assert.Single(handler.Faults).Code);
    }

    [Fact]
    public async Task ControlLoop_HandlerCancellationPropagatesWithoutTerminalFailure()
    {
        using var cancellation = new CancellationTokenSource();
        var ingress = new FakeIngressReader(
            ControlEnvelope.FromSignal(
                1,
                1,
                AssistantSignalSource.Recognition,
                AssistantSignal.RecognitionCompleted(1, 1, "cancel")));
        var handler = new ThrowingHandler(throwCancellation: true, cancellation);
        var loop = new AssistantControlLoop(ingress, handler);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => loop.Start(cancellation.Token).WaitAsync(TimeSpan.FromSeconds(1)));
        Assert.Empty(handler.Faults);
    }

    [Fact]
    public void NonMonotonicSourceTimestamp_ProducesTerminalFault()
    {
        using var ingress = new AssistantControlIngress();
        Assert.True(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechStarted(enableBargeIn: false),
            10,
            AssistantSignalSource.CaptureCallback,
            out _));

        Assert.False(ingress.TryEnqueueCritical(
            AssistantSignal.SpeechEnded(1, TimeSpan.Zero),
            9,
            AssistantSignalSource.CaptureCallback,
            out _));

        var drained = new List<ControlEnvelope>();
        while (ingress.TryRead(out ControlEnvelope envelope))
            drained.Add(envelope);
        Assert.Contains(drained, static value => value.Failure.Code == "VBX-CTRL-002");
    }

    private sealed class RecordingHandler : IAssistantControlLoopHandler
    {
        private readonly int _expectedHandled;
        private readonly TaskCompletionSource _handled = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public RecordingHandler(int expectedHandled)
        {
            _expectedHandled = expectedHandled;
        }

        public ConcurrentQueue<long> Sequences { get; } = new();
        public ConcurrentQueue<AssistantFailure> Faults { get; } = new();
        public Task ExpectedHandled => _handled.Task;

        public ValueTask HandleRuntimeCommandAsync(
            RuntimeCommand command,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            Record(envelope.IngressSequence);
            return ValueTask.CompletedTask;
        }

        public ValueTask HandleAssistantSignalAsync(
            AssistantSignal signal,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            Record(envelope.IngressSequence);
            return ValueTask.CompletedTask;
        }

        public ValueTask HandleTerminalFaultAsync(
            AssistantFailure failure,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            Faults.Enqueue(failure);
            if (Sequences.Count + Faults.Count >= _expectedHandled)
                _handled.TrySetResult();
            return ValueTask.CompletedTask;
        }

        public ValueTask HandleAbandonedAfterBarrierAsync(
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => ValueTask.CompletedTask;

        private void Record(long sequence)
        {
            Sequences.Enqueue(sequence);
            if (Sequences.Count + Faults.Count >= _expectedHandled)
                _handled.TrySetResult();
        }
    }

    private sealed class LatencyHandler : IAssistantControlLoopHandler, IDisposable
    {
        private readonly SemaphoreSlim _handled = new(0);
        private long _lastSequence;

        public async Task WaitForSequenceAsync(long sequence, CancellationToken cancellationToken)
        {
            while (Volatile.Read(ref _lastSequence) < sequence)
                await _handled.WaitAsync(cancellationToken);
        }

        public ValueTask HandleRuntimeCommandAsync(
            RuntimeCommand command,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleAssistantSignalAsync(
            AssistantSignal signal,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleTerminalFaultAsync(
            AssistantFailure failure,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleAbandonedAfterBarrierAsync(
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public void Dispose() => _handled.Dispose();

        private ValueTask Record(long sequence)
        {
            Volatile.Write(ref _lastSequence, sequence);
            _handled.Release();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class ConsumerThreadCaptureHandler : IAssistantControlLoopHandler, IDisposable
    {
        private readonly SemaphoreSlim _handled = new(0);
        private long _lastSequence;

        public bool HandledOnThreadPool { get; private set; }

        public List<int> HandledThreadIds { get; } = [];

        public async Task WaitForSequenceAsync(long sequence, CancellationToken cancellationToken)
        {
            while (Volatile.Read(ref _lastSequence) < sequence)
                await _handled.WaitAsync(cancellationToken);
        }

        public ValueTask HandleRuntimeCommandAsync(
            RuntimeCommand command,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleAssistantSignalAsync(
            AssistantSignal signal,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleTerminalFaultAsync(
            AssistantFailure failure,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public ValueTask HandleAbandonedAfterBarrierAsync(
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => Record(envelope.IngressSequence);

        public void Dispose() => _handled.Dispose();

        private ValueTask Record(long sequence)
        {
            HandledOnThreadPool = Thread.CurrentThread.IsThreadPoolThread;
            lock (HandledThreadIds)
                HandledThreadIds.Add(Environment.CurrentManagedThreadId);
            Volatile.Write(ref _lastSequence, sequence);
            _handled.Release();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class FakeIngressReader : IAssistantControlIngressReader
    {
        private readonly Queue<ControlEnvelope> _items;

        public FakeIngressReader(params ControlEnvelope[] items)
        {
            _items = new Queue<ControlEnvelope>(items);
        }

        public bool TryRead(out ControlEnvelope envelope)
            => _items.TryDequeue(out envelope);

        public void WaitForData(CancellationToken cancellationToken)
        {
            cancellationToken.WaitHandle.WaitOne();
            cancellationToken.ThrowIfCancellationRequested();
        }

        public ValueTask WaitForDataAsync(CancellationToken cancellationToken)
            => new(Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken));

        public bool IsAfterBarrier(in ControlEnvelope envelope) => false;
    }

    private sealed class DelayedWakeIngressReader : IAssistantControlIngressReader
    {
        private readonly Queue<ControlEnvelope> _items;
        private readonly TimeSpan _wakeDelay;
        private int _wakeCount;

        public DelayedWakeIngressReader(
            TimeSpan wakeDelay,
            int wakeCount,
            params ControlEnvelope[] items)
        {
            _items = new Queue<ControlEnvelope>(items);
            _wakeDelay = wakeDelay;
            _wakeCount = wakeCount;
        }

        public bool TryRead(out ControlEnvelope envelope)
            => _items.TryDequeue(out envelope);

        public void WaitForData(CancellationToken cancellationToken)
            => WaitForDataAsync(cancellationToken).AsTask().GetAwaiter().GetResult();

        public ValueTask WaitForDataAsync(CancellationToken cancellationToken)
        {
            if (Interlocked.Decrement(ref _wakeCount) >= 0)
                return new(Task.Delay(_wakeDelay, cancellationToken));
            return new(Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken));
        }

        public bool IsAfterBarrier(in ControlEnvelope envelope) => false;
    }

    private sealed class PulsedIngressReader : IAssistantControlIngressReader
    {
        private readonly Queue<ControlEnvelope> _items;
        private readonly SemaphoreSlim _available = new(0);

        public PulsedIngressReader(params ControlEnvelope[] items)
            => _items = new Queue<ControlEnvelope>(items);

        public bool TryRead(out ControlEnvelope envelope)
            => _items.TryDequeue(out envelope);

        public void WaitForData(CancellationToken cancellationToken)
            => _available.Wait(cancellationToken);

        public ValueTask WaitForDataAsync(CancellationToken cancellationToken)
            => new(_available.WaitAsync(cancellationToken));

        public bool IsAfterBarrier(in ControlEnvelope envelope) => false;

        public void Pulse() => _available.Release();
    }

    private sealed class ThrowingHandler : IAssistantControlLoopHandler
    {
        private readonly bool _throwCancellation;
        private readonly CancellationTokenSource? _cancellation;

        public ThrowingHandler(bool throwCancellation, CancellationTokenSource? cancellation = null)
        {
            _throwCancellation = throwCancellation;
            _cancellation = cancellation;
        }

        public ConcurrentQueue<AssistantFailure> Faults { get; } = new();

        public ValueTask HandleRuntimeCommandAsync(
            RuntimeCommand command,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => ValueTask.CompletedTask;

        public ValueTask HandleAssistantSignalAsync(
            AssistantSignal signal,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            if (_throwCancellation)
            {
                _cancellation!.Cancel();
                throw new OperationCanceledException(cancellationToken);
            }

            throw new InvalidOperationException("handler-test-fault");
        }

        public ValueTask HandleTerminalFaultAsync(
            AssistantFailure failure,
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
        {
            Faults.Enqueue(failure);
            return ValueTask.CompletedTask;
        }

        public ValueTask HandleAbandonedAfterBarrierAsync(
            ControlEnvelope envelope,
            CancellationToken cancellationToken)
            => ValueTask.CompletedTask;
    }
}
