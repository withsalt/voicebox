namespace VoiceBox.Assistant.Runtime.Control.Queues;

internal sealed class BoundedMpscQueue<T> : IDisposable
{
    private readonly T[] _items;
    private readonly long[] _published;
    private readonly SemaphoreSlim _slots;
    private long _claimedPosition;
    private long _readPosition;
    private int _disposed;

    public BoundedMpscQueue(int capacity)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        _items = new T[capacity];
        _published = new long[capacity];
        _slots = new SemaphoreSlim(capacity, capacity);
    }

    public bool TryWrite(T item)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_slots.Wait(0))
            return false;

        long position = Interlocked.Increment(ref _claimedPosition) - 1;
        int index = (int)(position % _items.Length);
        _items[index] = item;
        Volatile.Write(ref _published[index], position + 1);
        return true;
    }

    public bool TryPeek(out T item)
    {
        long position = Volatile.Read(ref _readPosition);
        int index = (int)(position % _items.Length);
        if (Volatile.Read(ref _published[index]) != position + 1)
        {
            item = default!;
            return false;
        }

        item = _items[index];
        return true;
    }

    public bool TryRead(out T item)
    {
        long position = Volatile.Read(ref _readPosition);
        int index = (int)(position % _items.Length);
        if (Volatile.Read(ref _published[index]) != position + 1)
        {
            item = default!;
            return false;
        }

        item = _items[index];
        _items[index] = default!;
        Volatile.Write(ref _published[index], 0);
        Volatile.Write(ref _readPosition, position + 1);
        _slots.Release();
        return true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _slots.Dispose();
    }
}
