using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

internal sealed class AssistantControlIngress :
    IAssistantControlIngressReader,
    IAssistantControlIngressTerminator,
    IPublicationFaultSink,
    IDisposable
{
    private const int CriticalLaneCapacity = 8;
    private const int CompletionAndLifecycleLaneCapacity = 32;
    private const int CompletionCapacity = 24;
    private const int LifecycleReserve = CompletionAndLifecycleLaneCapacity - CompletionCapacity;

    private readonly BoundedMpscQueue<ControlEnvelope> _critical;
    private readonly BoundedMpscQueue<ControlEnvelope> _completionAndLifecycle;
    private readonly SemaphoreSlim _available = new(0);
    private readonly SemaphoreSlim _completionSlots = new(CompletionCapacity, CompletionCapacity);
    private readonly SemaphoreSlim _lifecycleSlots = new(LifecycleReserve, LifecycleReserve);
    private readonly long[] _lastSourceTimestamps = new long[Enum.GetValues<AssistantSignalSource>().Length];
    private long _nextIngressSequence;
    private long _barrierSequence;
    private ControlEnvelope _terminalFault;
    private int _hasTerminalFault;
    private int _terminalConsumed;
    private int _disposed;

    public AssistantControlIngress()
    {
        _critical = new BoundedMpscQueue<ControlEnvelope>(CriticalLaneCapacity);
        _completionAndLifecycle = new BoundedMpscQueue<ControlEnvelope>(
            CompletionAndLifecycleLaneCapacity);
    }

    public static int CriticalCapacity => CriticalLaneCapacity;

    public static int CompletionAndLifecycleCapacity => CompletionAndLifecycleLaneCapacity;

    public long BarrierSequence => Volatile.Read(ref _barrierSequence);

    public bool HasTerminalFault => Volatile.Read(ref _hasTerminalFault) != 0;

    public bool TryTerminate(
        AssistantFailure failure,
        long sourceTimestamp,
        out long ingressSequence)
    {
        ingressSequence = 0;
        if (Volatile.Read(ref _disposed) != 0 || Volatile.Read(ref _hasTerminalFault) != 0)
            return false;

        long sequence = NextSequence();
        if (!RecordTerminalFault(sequence, sourceTimestamp, failure))
            return false;

        ingressSequence = sequence;
        return true;
    }

    void IPublicationFaultSink.Report(AssistantFailure failure)
        => TryTerminate(failure, System.Diagnostics.Stopwatch.GetTimestamp(), out _);

    public bool TryEnqueueCritical(
        AssistantSignal signal,
        long sourceTimestamp,
        AssistantSignalSource source,
        out long ingressSequence)
    {
        ingressSequence = 0;
        if (Volatile.Read(ref _disposed) != 0 || Volatile.Read(ref _hasTerminalFault) != 0)
            return false;

        if (!TryAcceptSourceTimestamp(source, sourceTimestamp))
        {
            RecordTerminalFault(
                NextSequence(),
                sourceTimestamp,
                new AssistantFailure(
                    "VBX-CTRL-002",
                    "ordering",
                    source.ToString(),
                    "non-monotonic-source-timestamp"));
            return false;
        }

        long sequence = NextSequence();
        long barrier = BarrierSequence;
        if (barrier != 0 && sequence > barrier)
        {
            // Taking a sequence is a commitment. Preserve it as an abandoned envelope so the
            // single consumer can advance FIFO without mistaking a post-barrier rejection for a
            // missing ingress item.
            ControlEnvelope abandoned = ControlEnvelope.FromSignal(
                sequence,
                sourceTimestamp,
                source,
                signal);
            if (_critical.TryWrite(abandoned))
            {
                _available.Release();
            }
            else
            {
                RecordTerminalFault(
                    sequence,
                    sourceTimestamp,
                    new AssistantFailure(
                        "VBX-CTRL-001",
                        "capacity",
                        "AssistantControlIngress",
                        "post-barrier-abandon-overflow"));
            }
            ingressSequence = sequence;
            return false;
        }

        ControlEnvelope envelope = ControlEnvelope.FromSignal(sequence, sourceTimestamp, source, signal);
        if (!_critical.TryWrite(envelope))
        {
            RecordTerminalFault(
                sequence,
                sourceTimestamp,
                new AssistantFailure(
                    "VBX-CTRL-001",
                    "capacity",
                    "AssistantControlIngress",
                    "critical-lane-overflow"));
            return false;
        }

        ingressSequence = sequence;
        _available.Release();
        return true;
    }

    public async ValueTask<long> EnqueueAsync(
        RuntimeCommand command,
        long sourceTimestamp,
        CancellationToken cancellationToken = default)
    {
        ThrowIfUnavailable();
        EnsureSourceTimestamp(AssistantSignalSource.HostLifecycle, sourceTimestamp);
        await _lifecycleSlots.WaitAsync(cancellationToken).ConfigureAwait(false);
        long sequence = 0;
        try
        {
            sequence = NextSequence();
            if (command.IsBarrier)
                SetBarrier(sequence);
            ControlEnvelope envelope = ControlEnvelope.FromCommand(
                sequence,
                sourceTimestamp,
                AssistantSignalSource.HostLifecycle,
                command);
            if (!_completionAndLifecycle.TryWrite(envelope))
                throw new InvalidOperationException("Reserved lifecycle admission was unavailable.");
            _available.Release();
            return sequence;
        }
        catch
        {
            _lifecycleSlots.Release();
            if (sequence != 0)
            {
                RecordTerminalFault(
                    sequence,
                    sourceTimestamp,
                    new AssistantFailure(
                        "VBX-CTRL-003",
                        "lifecycle",
                        "AssistantControlIngress",
                        "lifecycle-admission-failed"));
            }
            throw;
        }
    }

    public async ValueTask<long> EnqueueSignalAsync(
        AssistantSignal signal,
        long sourceTimestamp,
        AssistantSignalSource source,
        CancellationToken cancellationToken = default)
    {
        ThrowIfUnavailable();
        EnsureSourceTimestamp(source, sourceTimestamp);
        await _completionSlots.WaitAsync(cancellationToken).ConfigureAwait(false);
        long sequence = 0;
        try
        {
            sequence = NextSequence();
            long barrier = BarrierSequence;
            if (barrier != 0 && sequence > barrier)
                throw new InvalidOperationException("Assistant control signal was rejected after the lifecycle barrier.");
            ControlEnvelope envelope = ControlEnvelope.FromSignal(
                sequence,
                sourceTimestamp,
                source,
                signal);
            if (!_completionAndLifecycle.TryWrite(envelope))
                throw new InvalidOperationException("Reserved completion admission was unavailable.");
            _available.Release();
            return sequence;
        }
        catch
        {
            _completionSlots.Release();
            if (sequence != 0)
            {
                RecordTerminalFault(
                    sequence,
                    sourceTimestamp,
                    new AssistantFailure(
                        "VBX-CTRL-004",
                        "completion",
                        source.ToString(),
                        "completion-admission-failed"));
            }
            throw;
        }
    }

    public bool TryRead(out ControlEnvelope envelope)
    {
        bool hasCritical = _critical.TryPeek(out ControlEnvelope critical);
        bool hasCompletion = _completionAndLifecycle.TryPeek(out ControlEnvelope completion);
        bool hasFault = Volatile.Read(ref _hasTerminalFault) == 2 &&
            Volatile.Read(ref _terminalConsumed) == 0;
        long faultSequence = hasFault ? _terminalFault.IngressSequence : long.MaxValue;
        long criticalSequence = hasCritical ? critical.IngressSequence : long.MaxValue;
        long completionSequence = hasCompletion ? completion.IngressSequence : long.MaxValue;

        if (faultSequence < criticalSequence && faultSequence < completionSequence)
        {
            if (Interlocked.Exchange(ref _terminalConsumed, 1) == 0)
            {
                envelope = _terminalFault;
                ConsumeAvailabilitySignal();
                return true;
            }
        }

        if (criticalSequence < completionSequence)
        {
            if (_critical.TryRead(out envelope))
            {
                ConsumeAvailabilitySignal();
                return true;
            }
        }
        else if (hasCompletion && _completionAndLifecycle.TryRead(out envelope))
        {
            if (envelope.Kind == ControlEnvelopeKind.RuntimeCommand)
                _lifecycleSlots.Release();
            else
                _completionSlots.Release();
            ConsumeAvailabilitySignal();
            return true;
        }

        if (hasFault && Interlocked.Exchange(ref _terminalConsumed, 1) == 0)
        {
            envelope = _terminalFault;
            ConsumeAvailabilitySignal();
            return true;
        }

        envelope = default;
        return false;
    }

    public void WaitForData(CancellationToken cancellationToken)
        => _available.Wait(cancellationToken);

    public ValueTask WaitForDataAsync(CancellationToken cancellationToken)
        => new(_available.WaitAsync(cancellationToken));

    public bool IsAfterBarrier(in ControlEnvelope envelope)
    {
        long barrier = BarrierSequence;
        return barrier != 0 &&
            envelope.IngressSequence > barrier &&
            envelope.Kind == ControlEnvelopeKind.AssistantSignal;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        _available.Dispose();
        _completionSlots.Dispose();
        _lifecycleSlots.Dispose();
        _critical.Dispose();
        _completionAndLifecycle.Dispose();
    }

    private long NextSequence() => Interlocked.Increment(ref _nextIngressSequence);

    private void SetBarrier(long sequence)
        => Interlocked.CompareExchange(ref _barrierSequence, sequence, 0);

    private bool TryAcceptSourceTimestamp(AssistantSignalSource source, long timestamp)
    {
        int index = (int)source;
        while (true)
        {
            long current = Volatile.Read(ref _lastSourceTimestamps[index]);
            if (timestamp < current)
                return false;
            if (Interlocked.CompareExchange(ref _lastSourceTimestamps[index], timestamp, current) == current)
                return true;
        }
    }

    private void EnsureSourceTimestamp(AssistantSignalSource source, long timestamp)
    {
        if (TryAcceptSourceTimestamp(source, timestamp))
            return;

        RecordTerminalFault(
            NextSequence(),
            timestamp,
            new AssistantFailure(
                "VBX-CTRL-002",
                "ordering",
                source.ToString(),
                "non-monotonic-source-timestamp"));
        throw new InvalidOperationException(
            $"Assistant control source timestamp decreased for '{source}'.");
    }

    private void ConsumeAvailabilitySignal()
        => _available.Wait(0);

    private bool RecordTerminalFault(long sequence, long timestamp, AssistantFailure failure)
    {
        if (Interlocked.CompareExchange(ref _hasTerminalFault, 1, 0) != 0)
            return false;

        _terminalFault = ControlEnvelope.FromTerminalFault(sequence, timestamp, failure);
        Volatile.Write(ref _hasTerminalFault, 2);
        _available.Release();
        return true;
    }

    private void ThrowIfUnavailable()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Volatile.Read(ref _hasTerminalFault) != 0)
            throw new InvalidOperationException("Assistant control ingress is terminated.");
    }
}
