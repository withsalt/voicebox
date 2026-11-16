using VoiceBox.Assistant.Runtime.Publications.Models;

namespace VoiceBox.Assistant.Runtime.Publications.Dispatching;

internal sealed class PublicationIngress
{
    private readonly Queue<PublicationEnvelope> _items = new(64);
    private readonly SemaphoreSlim _available = new(0);
    private readonly object _lock = new();
    private bool _completed;

    public bool TryWrite(PublicationEnvelope envelope)
    {
        lock (_lock)
        {
            if (_completed)
                return false;
            _items.Enqueue(envelope);
        }
        _available.Release();
        return true;
    }

    public bool TryRead(out PublicationEnvelope envelope)
    {
        lock (_lock)
        {
            if (!_items.TryDequeue(out envelope))
                return false;
        }
        _available.Wait(0);
        return true;
    }

    public bool WaitToRead(CancellationToken cancellationToken)
    {
        while (true)
        {
            lock (_lock)
            {
                if (_items.Count > 0)
                    return true;
                if (_completed)
                    return false;
            }
            _available.Wait(cancellationToken);
        }
    }

    public async ValueTask<bool> WaitToReadAsync(CancellationToken cancellationToken)
    {
        while (true)
        {
            lock (_lock)
            {
                if (_items.Count > 0)
                    return true;
                if (_completed)
                    return false;
            }
            await _available.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    public void Complete(bool discardPending = false)
    {
        lock (_lock)
        {
            _completed = true;
            if (discardPending)
                _items.Clear();
        }
        _available.Release();
    }
}
