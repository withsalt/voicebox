using System.Diagnostics;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Queues;

/// <summary>
/// A bounded side queue that carries the payload of a data-plane observation next to its
/// correlated control signal. The producer writes the payload before admitting the signal; the
/// single consumer takes it back under the same operation identity. A mismatch means the queue and
/// the signal stream diverged, which can only be an ordering-invariant break, so it converges as a
/// terminal control fault rather than silently dropping or reordering a fact.
/// </summary>
internal sealed class CorrelatedObservationQueue<T>
{
    private readonly BoundedMpscQueue<Entry> _entries;
    private readonly AssistantControlIngress _ingress;
    private readonly string _component;
    private readonly string _overflowCode;
    private readonly string _mismatchCode;
    private readonly string _mismatchMessageKey;

    public CorrelatedObservationQueue(
        AssistantControlIngress ingress,
        int capacity,
        string component,
        string overflowCode,
        string mismatchCode,
        string mismatchMessageKey)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _entries = new BoundedMpscQueue<Entry>(capacity);
        _component = component ?? throw new ArgumentNullException(nameof(component));
        _overflowCode = overflowCode ?? throw new ArgumentNullException(nameof(overflowCode));
        _mismatchCode = mismatchCode ?? throw new ArgumentNullException(nameof(mismatchCode));
        _mismatchMessageKey = mismatchMessageKey ?? throw new ArgumentNullException(nameof(mismatchMessageKey));
    }

    /// <summary>
    /// Records a payload for a signal that is about to be admitted. Allocation-free and
    /// non-blocking, so a real-time producer may call it. Overflow is a capacity-invariant break.
    /// </summary>
    public bool TryWrite(long operationId, T payload, string overflowMessageKey)
    {
        if (_entries.TryWrite(new Entry(operationId, payload)))
            return true;

        Terminate(_overflowCode, "capacity", overflowMessageKey);
        return false;
    }

    /// <summary>Takes the payload the admitted signal refers to, on the control consumer.</summary>
    public bool TryTake(long operationId, out T payload)
    {
        if (_entries.TryRead(out Entry entry) && entry.OperationId == operationId)
        {
            payload = entry.Payload;
            return true;
        }

        payload = default!;
        Terminate(_mismatchCode, "ordering", _mismatchMessageKey);
        return false;
    }

    private void Terminate(string code, string category, string messageKey)
        => _ingress.TryTerminate(
            new AssistantFailure(code, category, _component, messageKey),
            Stopwatch.GetTimestamp(),
            out _);

    private readonly record struct Entry(long OperationId, T Payload);
}
