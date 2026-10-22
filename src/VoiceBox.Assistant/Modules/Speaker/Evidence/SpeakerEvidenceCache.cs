using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Evidence;

/// <summary>
/// Embedding/score cache keyed by <c>(streamEpoch, windowStart, windowEnd)</c>. Embeddings are
/// extracted once per window; per-bank scores are cheap comparisons stored beside them. Entries
/// expire with the identity ring; a discontinuity never clears audio that arrived after the gap.
/// </summary>
internal sealed class SpeakerEvidenceCache
{
    private readonly object _lock = new();
    private readonly Dictionary<WindowKey, SpeakerWindowEvidence> _entries = new();
    private readonly HashSet<WindowKey> _inFlight = new();
    private TaskCompletionSource<bool>? _updateSignal;

    public bool TryGet(long streamEpoch, long start, long end, out SpeakerWindowEvidence evidence)
    {
        lock (_lock)
            return _entries.TryGetValue(new WindowKey(streamEpoch, start, end), out evidence!);
    }

    public bool IsInFlight(long streamEpoch, long start, long end)
    {
        lock (_lock)
            return _inFlight.Contains(new WindowKey(streamEpoch, start, end));
    }

    /// <summary>Reserves a window for inference. False when it is cached or already in flight.</summary>
    public bool TryReserve(long streamEpoch, long start, long end)
    {
        lock (_lock)
        {
            var key = new WindowKey(streamEpoch, start, end);
            if (_entries.ContainsKey(key) || _inFlight.Contains(key))
                return false;
            _inFlight.Add(key);
            return true;
        }
    }

    public void Release(long streamEpoch, long start, long end, bool notify = true)
    {
        lock (_lock)
        {
            _inFlight.Remove(new WindowKey(streamEpoch, start, end));
            if (notify) PulseLocked();
        }
    }

    public void Store(SpeakerWindowEvidence evidence)
    {
        ArgumentNullException.ThrowIfNull(evidence);
        lock (_lock)
        {
            var key = new WindowKey(evidence.StreamEpoch, evidence.StartSample, evidence.EndSampleExclusive);
            _entries[key] = evidence;
            _inFlight.Remove(key);
            PulseLocked();
        }
    }

    /// <summary>Drops windows no longer covered by the ring or belonging to another stream.</summary>
    public void Trim(long streamEpoch, long retainedStart)
    {
        lock (_lock)
        {
            if (_entries.Count == 0)
                return;
            List<WindowKey>? expired = null;
            foreach (WindowKey key in _entries.Keys)
            {
                if (key.StreamEpoch != streamEpoch || key.Start < retainedStart)
                    (expired ??= []).Add(key);
            }
            if (expired is null)
                return;
            foreach (WindowKey key in expired)
                _entries.Remove(key);
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _entries.Clear();
            _inFlight.Clear();
            PulseLocked();
        }
    }

    public int Count
    {
        get { lock (_lock) return _entries.Count; }
    }

    /// <summary>Task completed on the next cache change; captured atomically with a read.</summary>
    public Task UpdateSignal
    {
        get { lock (_lock) return (_updateSignal ??= NewSignal()).Task; }
    }

    // Allocate a completion source only when a consumer actually waits; silence must not allocate
    // a new task every 10ms simply to notify a nonexistent waiter.
    public void NotifyAudioAvailable() => Pulse();

    public void Pulse()
    {
        lock (_lock)
            PulseLocked();
    }

    private void PulseLocked()
    {
        TaskCompletionSource<bool>? previous = _updateSignal;
        _updateSignal = null;
        previous?.TrySetResult(true);
    }

    private static TaskCompletionSource<bool> NewSignal()
        => new(TaskCreationOptions.RunContinuationsAsynchronously);

    private readonly record struct WindowKey(long StreamEpoch, long Start, long End);
}
