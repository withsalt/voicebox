using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

internal sealed class AssistantControlLoop
{
    private const string ConsumerThreadName = "voicebox-assistant-control-loop";

    private readonly IAssistantControlIngressReader _ingress;
    private readonly IAssistantControlLoopHandler _handler;
    private readonly TimeProvider _timeProvider;
    private readonly TimeSpan _gapTimeout;
    private readonly SortedDictionary<long, ControlEnvelope> _pending = [];
    private long _nextExpected = 1;
    private int _consumerThreadId;
    private int _started;

    public AssistantControlLoop(
        IAssistantControlIngressReader ingress,
        IAssistantControlLoopHandler handler,
        TimeProvider? timeProvider = null,
        TimeSpan? gapTimeout = null)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _handler = handler ?? throw new ArgumentNullException(nameof(handler));
        _timeProvider = timeProvider ?? TimeProvider.System;
        _gapTimeout = gapTimeout ?? TimeSpan.FromMilliseconds(40);
    }

    public int ConsumerThreadId => Volatile.Read(ref _consumerThreadId);

    public Task Start(CancellationToken cancellationToken)
    {
        if (Interlocked.Exchange(ref _started, 1) != 0)
            throw new InvalidOperationException("Assistant control loop can only be started once.");

        // The consumer owns a dedicated thread instead of a pooled work item. Interruption
        // decisions must reach the reducer within the 5.4.1 callback-period budget, and a
        // pooled wake-up can queue behind unrelated recognition/response work whenever the
        // host constrains the pool.
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var consumer = new Thread(() => RunConsumer(completion, cancellationToken))
        {
            IsBackground = true,
            Name = ConsumerThreadName,
        };
        consumer.Start();
        return completion.Task;
    }

    private void RunConsumer(TaskCompletionSource completion, CancellationToken cancellationToken)
    {
        try
        {
            Run(cancellationToken);
            completion.TrySetResult();
        }
        catch (OperationCanceledException exception)
        {
            completion.TrySetCanceled(exception.CancellationToken);
        }
        catch (Exception exception)
        {
            completion.TrySetException(exception);
        }
    }

    private void Run(CancellationToken cancellationToken)
    {
        long gapStarted = 0;
        Volatile.Write(ref _consumerThreadId, Environment.CurrentManagedThreadId);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            while (_ingress.TryRead(out ControlEnvelope envelope))
                _pending.Add(envelope.IngressSequence, envelope);

            if (_pending.Remove(_nextExpected, out ControlEnvelope next))
            {
                gapStarted = 0;
                if (!TryDispatch(next, cancellationToken))
                    return;
                _nextExpected++;
                if (next.Kind == ControlEnvelopeKind.TerminalFault)
                    return;
                continue;
            }

            if (_pending.Count > 0)
            {
                if (gapStarted == 0)
                    gapStarted = _timeProvider.GetTimestamp();
                TimeSpan elapsed = _timeProvider.GetElapsedTime(gapStarted);
                TimeSpan remaining = _gapTimeout - elapsed;
                if (remaining <= TimeSpan.Zero)
                {
                    HandleGapFault(elapsed, cancellationToken);
                    return;
                }

                // The gap watchdog stays on the injected clock, so this rare recovery branch
                // keeps its timer-based deadline instead of an OS wait.
                Task data = _ingress.WaitForDataAsync(cancellationToken).AsTask();
                Task deadline = Task.Delay(remaining, _timeProvider);
                if (Task.WhenAny(data, deadline).GetAwaiter().GetResult() != data)
                {
                    HandleGapFault(
                        _timeProvider.GetElapsedTime(gapStarted),
                        cancellationToken);
                    return;
                }
                data.GetAwaiter().GetResult();
                continue;
            }

            gapStarted = 0;
            _ingress.WaitForData(cancellationToken);
        }
    }

    private void HandleGapFault(
        TimeSpan elapsed,
        CancellationToken cancellationToken)
    {
        var failure = new AssistantFailure(
            "VBX-CTRL-005",
            "ordering",
            "AssistantControlLoop",
            $"ingress-sequence-gap:{_nextExpected}:elapsed-ticks={elapsed.Ticks}");
        // A watchdog fault occurs outside the normal envelope stream. Latch the production
        // ingress before notifying the handler so concurrent producers immediately observe a
        // terminal control plane and cannot enqueue work after the loop has stopped. Test readers
        // that intentionally expose only the read contract still receive the synthetic callback.
        long faultSequence = _nextExpected;
        if (_ingress is IAssistantControlIngressTerminator terminator &&
            terminator.TryTerminate(failure, _timeProvider.GetTimestamp(), out long latchedSequence) &&
            latchedSequence > 0)
        {
            faultSequence = latchedSequence;
        }
        ControlEnvelope fault = ControlEnvelope.FromTerminalFault(
            faultSequence,
            _timeProvider.GetTimestamp(),
            failure);
        Complete(_handler.HandleTerminalFaultAsync(failure, fault, cancellationToken));
    }

    private ValueTask DispatchAsync(ControlEnvelope envelope, CancellationToken cancellationToken)
    {
        if (_ingress.IsAfterBarrier(in envelope))
            return _handler.HandleAbandonedAfterBarrierAsync(envelope, cancellationToken);

        return envelope.Kind switch
        {
            ControlEnvelopeKind.RuntimeCommand => _handler.HandleRuntimeCommandAsync(
                envelope.Command,
                envelope,
                cancellationToken),
            ControlEnvelopeKind.AssistantSignal => _handler.HandleAssistantSignalAsync(
                envelope.Signal,
                envelope,
                cancellationToken),
            ControlEnvelopeKind.TerminalFault => _handler.HandleTerminalFaultAsync(
                envelope.Failure,
                envelope,
                cancellationToken),
            _ => throw new ArgumentOutOfRangeException(nameof(envelope), envelope.Kind, null),
        };
    }

    private bool TryDispatch(
        ControlEnvelope envelope,
        CancellationToken cancellationToken)
    {
        try
        {
            Complete(DispatchAsync(envelope, cancellationToken));
            return true;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception)
        {
            var failure = new AssistantFailure(
                "VBX-CTRL-010",
                "handler",
                "AssistantControlLoop",
                exception.GetType().Name);
            ControlEnvelope fault = ControlEnvelope.FromTerminalFault(
                envelope.IngressSequence,
                _timeProvider.GetTimestamp(),
                failure);
            Complete(_handler.HandleTerminalFaultAsync(failure, fault, cancellationToken));
            return false;
        }
    }

    /// <summary>
    /// Drives a handler to completion on the dedicated consumer thread. The loop already
    /// processes envelopes strictly one at a time, so waiting here preserves FIFO ordering while
    /// keeping the consumer off the shared thread pool.
    /// </summary>
    private static void Complete(ValueTask handler)
    {
        if (handler.IsCompletedSuccessfully)
        {
            handler.GetAwaiter().GetResult();
            return;
        }

        handler.AsTask().GetAwaiter().GetResult();
    }
}
