using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Publications.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Models;

namespace VoiceBox.Assistant.Runtime.Publications.Dispatching;

internal sealed class OrderedPublicationBoundary : IAssistantPublicEventSink, IAsyncDisposable
{
    private const string PublisherThreadName = "voicebox-assistant-publication-boundary";

    private readonly AssistantEventDispatcher _publisher;
    private readonly TimeProvider _timeProvider;
    private readonly TimeSpan _gapTimeout;
    private readonly PublicationIngress _ingress;
    private readonly IPublicationFaultSink? _faults;
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _worker;
    private readonly Thread _workerThread;
    private readonly object _admissionLock = new();
    private readonly object _progressLock = new();
    private readonly SortedDictionary<long, AssistantPublication> _pending = [];
    private TaskCompletionSource? _progress;
    private bool _started;
    private long _reservedSequence;
    private long _deliveredSequence;
    private int _publisherThreadId;
    private int _completed;
    private int _terminated;
    private int _faultReported;
    private int _faulted;

    public OrderedPublicationBoundary(
        AssistantEventDispatcher publisher,
        TimeProvider? timeProvider = null,
        TimeSpan? gapTimeout = null,
        IPublicationFaultSink? faults = null,
        bool startWorker = true)
    {
        _publisher = publisher ?? throw new ArgumentNullException(nameof(publisher));
        _timeProvider = timeProvider ?? TimeProvider.System;
        _gapTimeout = gapTimeout ?? TimeSpan.FromMilliseconds(40);
        _faults = faults;
        _ingress = new PublicationIngress();
        // R6 requires a single publication context. It is a dedicated thread rather than a
        // pooled work item so a released event reaches subscribers immediately even when the
        // host constrains the thread pool; producer threads still never block on delivery.
        var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _workerThread = new Thread(() => RunWorker(completion))
        {
            IsBackground = true,
            Name = PublisherThreadName,
        };
        _worker = completion.Task;
        if (startWorker)
            Start();
    }

    /// <summary>Composition defers the thread until every collaborator has been constructed.</summary>
    public void Start()
    {
        lock (_admissionLock)
        {
            if (_started)
                return;
            _workerThread.Start();
            _started = true;
        }
    }

    public long NextPublicationSequence => Volatile.Read(ref _reservedSequence) + 1;

    public long DeliveredSequence => Volatile.Read(ref _deliveredSequence);

    public int PublisherThreadId => Volatile.Read(ref _publisherThreadId);

    public Task Completion => _worker;

    /// <summary>True after the worker has emitted its terminal publication watermark.</summary>
    public bool IsFaulted => Volatile.Read(ref _faulted) != 0;

    public long RaiseStateChanged(AssistantState from, AssistantState to)
        => PublishStream(new StateChangedPublication(from, to));

    public long RaiseWakeModeChanged(AssistantWakeMode from, AssistantWakeMode to)
        => PublishStream(new WakeModeChangedPublication(from, to));

    public long RaiseWakeupDetected(AssistantWakeupDetectedEventArgs args)
        => PublishStream(new WakeupDetectedPublication(args));

    public long RaiseUserSpoke(UserSpokeEventArgs args)
        => PublishStream(new UserSpokePublication(args));

    public long RaiseBargeInValidation(BargeInValidationEventArgs args)
        => PublishStream(new BargeInValidationPublication(args));

    public long RaiseFaulted(AssistantFaultedEventArgs args)
        => PublishStream(new FaultedPublication(args));

    public long RaiseAssistantText(AssistantTextEventArgs args)
        => PublishStream(new AssistantTextPublication(args));

    public long RaiseAssistantResponseGenerated(AssistantResponseGeneratedEventArgs args)
        => PublishStream(new AssistantResponseGeneratedPublication(args));

    public long RaiseAssistantReplied(AssistantReplyEventArgs args)
        => PublishStream(new AssistantRepliedPublication(args));

    public long ReserveSequence()
    {
        ThrowIfUnavailable();
        return Interlocked.Increment(ref _reservedSequence);
    }

    public void PublishCommitted(long publicationSequence, AssistantPublication publication)
    {
        ArgumentNullException.ThrowIfNull(publication);
        ThrowIfUnavailable();
        if (publicationSequence <= 0 || publicationSequence > Volatile.Read(ref _reservedSequence))
            throw new ArgumentOutOfRangeException(nameof(publicationSequence));
        if (!_ingress.TryWrite(new PublicationEnvelope(publicationSequence, publication)))
            throw new InvalidOperationException("Ordered publication boundary is not accepting events.");
    }

    public long PublishStream(AssistantPublication publication)
    {
        ArgumentNullException.ThrowIfNull(publication);
        lock (_admissionLock)
        {
            ThrowIfUnavailable();
            long sequence = Interlocked.Increment(ref _reservedSequence);
            if (!_ingress.TryWrite(new PublicationEnvelope(sequence, publication)))
                throw new InvalidOperationException("Ordered publication boundary is not accepting events.");
            return sequence;
        }
    }

    public async Task DrainAsync(CancellationToken cancellationToken = default)
    {
        long target = Volatile.Read(ref _reservedSequence);
        await WaitUntilDeliveredAsync(target, cancellationToken).ConfigureAwait(false);
    }

    public async Task WaitUntilDeliveredAsync(
        long publicationSequence,
        CancellationToken cancellationToken = default)
    {
        if (publicationSequence < 0 || publicationSequence > Volatile.Read(ref _reservedSequence))
            throw new ArgumentOutOfRangeException(nameof(publicationSequence));

        long target = publicationSequence;
        while (Volatile.Read(ref _deliveredSequence) < target)
        {
            if (_worker.IsCompleted)
            {
                await _worker.ConfigureAwait(false);
                return;
            }

            Task progress;
            lock (_progressLock)
            {
                // Delivery advances before SignalProgress swaps the TCS. Recheck while
                // holding that same swap lock so a waiter cannot attach to the next,
                // unsignaled TCS after its requested sequence was already delivered.
                if (Volatile.Read(ref _deliveredSequence) >= target)
                    return;
                progress = (_progress ??= CreateProgressSource()).Task;
            }
            Task completed = await Task.WhenAny(progress, _worker)
                .WaitAsync(cancellationToken)
                .ConfigureAwait(false);
            if (completed == _worker)
            {
                await _worker.ConfigureAwait(false);
                return;
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _completed, 1) != 0)
            return;

        _ingress.Complete();
        Start();
        try
        {
            await _worker.ConfigureAwait(false);
        }
        finally
        {
            _stop.Cancel();
            _stop.Dispose();
        }
    }

    private void RunWorker(TaskCompletionSource completion)
    {
        try
        {
            try
            {
                Run();
            }
            finally
            {
                // Once the only consumer exits, these events can never be delivered. Close
                // admission and release their payloads before exposing worker completion.
                Volatile.Write(ref _terminated, 1);
                _ingress.Complete(discardPending: true);
                _pending.Clear();
            }
            completion.TrySetResult();
        }
        catch (OperationCanceledException exception)
        {
            completion.TrySetCanceled(exception.CancellationToken);
        }
        catch (Exception exception)
        {
            ReportFault(exception);
            completion.TrySetException(exception);
        }
    }

    private void ThrowIfUnavailable()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _completed) != 0, this);
        if (Volatile.Read(ref _terminated) != 0)
            throw new InvalidOperationException("Ordered publication boundary is no longer running.");
    }

    private void ReportFault(Exception exception)
    {
        if (_faults is null || Volatile.Read(ref _completed) != 0 ||
            Interlocked.Exchange(ref _faultReported, 1) != 0)
        {
            return;
        }

        // The worker cannot wait for a missing sequence after a gap. Emit the terminal
        // watermark on this same publication context before latching the control plane; the
        // subsequent Kernel fault effect observes IsFaulted and does not enqueue a second event.
        Volatile.Write(ref _faulted, 1);
        _publisher.RaiseFaulted(new AssistantFaultedEventArgs
        {
            Component = "OrderedPublicationBoundary",
            Exception = exception,
        });
        _faults.Report(new AssistantFailure(
            "VBX-CTRL-021",
            "runtime",
            "OrderedPublicationBoundary",
            exception is InvalidOperationException
                ? "publication-boundary-fault"
                : "publication-boundary-worker-fault"));
    }

    private void Run()
    {
        try
        {
            long nextExpected = 1;
            long gapStarted = 0;
            Volatile.Write(ref _publisherThreadId, Environment.CurrentManagedThreadId);
            while (_ingress.WaitToRead(_stop.Token))
            {
                while (_ingress.TryRead(out PublicationEnvelope envelope))
                {
                    if (!_pending.TryAdd(envelope.PublicationSequence, envelope.Publication))
                        throw new InvalidOperationException(
                            $"Duplicate PublicationSequence {envelope.PublicationSequence}.");
                }

                while (_pending.Remove(nextExpected, out AssistantPublication? publication))
                {
                    gapStarted = 0;
                    Dispatch(publication);
                    Volatile.Write(ref _deliveredSequence, nextExpected);
                    SignalProgress();
                    nextExpected++;
                }

                if (_pending.Count > 0)
                {
                    if (gapStarted == 0)
                        gapStarted = _timeProvider.GetTimestamp();
                    TimeSpan remaining = _gapTimeout - _timeProvider.GetElapsedTime(gapStarted);
                    if (remaining <= TimeSpan.Zero)
                    {
                        throw new InvalidOperationException(
                            $"PublicationSequence gap at {nextExpected}.");
                    }

                    // The gap watchdog stays on the injected clock, so this rare recovery
                    // branch keeps its timer-based deadline instead of an OS wait.
                    Task<bool> data = _ingress.WaitToReadAsync(_stop.Token).AsTask();
                    Task deadline = Task.Delay(remaining, _timeProvider);
                    if (Task.WhenAny(data, deadline).GetAwaiter().GetResult() != data)
                    {
                        throw new InvalidOperationException(
                            $"PublicationSequence gap at {nextExpected}.");
                    }
                    if (!data.GetAwaiter().GetResult())
                        break;
                }
                else
                {
                    gapStarted = 0;
                }
            }

            while (_pending.Remove(nextExpected, out AssistantPublication? publication))
            {
                Dispatch(publication);
                Volatile.Write(ref _deliveredSequence, nextExpected);
                SignalProgress();
                nextExpected++;
            }

            if (_pending.Count != 0 || nextExpected <= Volatile.Read(ref _reservedSequence))
                throw new InvalidOperationException($"PublicationSequence gap at {nextExpected} during drain.");
        }
        catch
        {
            SignalProgress();
            throw;
        }
    }

    private void Dispatch(AssistantPublication publication)
    {
        switch (publication)
        {
            case StateChangedPublication value:
                _publisher.RaiseStateChanged(value.From, value.To);
                break;
            case WakeModeChangedPublication value:
                _publisher.RaiseWakeModeChanged(value.From, value.To);
                break;
            case WakeupDetectedPublication value:
                _publisher.RaiseWakeupDetected(value.Args);
                break;
            case UserSpokePublication value:
                _publisher.RaiseUserSpoke(value.Args);
                break;
            case BargeInValidationPublication value:
                _publisher.RaiseBargeInValidation(value.Args);
                break;
            case FaultedPublication value:
                _publisher.RaiseFaulted(value.Args);
                break;
            case AssistantTextPublication value:
                _publisher.RaiseAssistantText(value.Args);
                break;
            case AssistantResponseGeneratedPublication value:
                _publisher.RaiseAssistantResponseGenerated(value.Args);
                break;
            case AssistantRepliedPublication value:
                _publisher.RaiseAssistantReplied(value.Args);
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(publication),
                    publication.GetType().Name,
                    null);
        }
    }

    private void SignalProgress()
    {
        TaskCompletionSource? completed;
        lock (_progressLock)
        {
            completed = _progress;
            _progress = null;
        }
        completed?.TrySetResult();
    }

    private static TaskCompletionSource CreateProgressSource()
        => new(TaskCreationOptions.RunContinuationsAsynchronously);
}
