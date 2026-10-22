namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class FloatFifo
    {
        private float[] _buffer;
        private int _head;
        private int _count;

        public FloatFifo(int capacity)
        {
            _buffer = new float[Math.Max(16, capacity)];
        }

        public int Count => _count;

        public void Write(ReadOnlySpan<float> source)
        {
            if (source.IsEmpty) return;
            EnsureCapacity(_count + source.Length);

            int tail = (_head + _count) % _buffer.Length;
            int first = Math.Min(source.Length, _buffer.Length - tail);
            source[..first].CopyTo(_buffer.AsSpan(tail));
            if (first < source.Length)
                source[first..].CopyTo(_buffer);
            _count += source.Length;
        }

        public void WriteZeros(int count)
        {
            if (count <= 0) return;
            EnsureCapacity(_count + count);

            int tail = (_head + _count) % _buffer.Length;
            int first = Math.Min(count, _buffer.Length - tail);
            _buffer.AsSpan(tail, first).Clear();
            if (first < count)
                _buffer.AsSpan(0, count - first).Clear();
            _count += count;
        }

        public void Read(Span<float> destination)
        {
            if (destination.Length > _count)
                throw new InvalidOperationException($"FloatFifo underflow: need {destination.Length}, have {_count}.");

            int first = Math.Min(destination.Length, _buffer.Length - _head);
            _buffer.AsSpan(_head, first).CopyTo(destination);
            if (first < destination.Length)
                _buffer.AsSpan(0, destination.Length - first).CopyTo(destination[first..]);

            _head = (_head + destination.Length) % _buffer.Length;
            _count -= destination.Length;
            if (_count == 0) _head = 0;
        }

        public void Clear()
        {
            _head = 0;
            _count = 0;
        }

        private void EnsureCapacity(int required)
        {
            if (required <= _buffer.Length) return;

            int newCapacity = _buffer.Length * 2;
            while (newCapacity < required) newCapacity *= 2;

            var next = new float[newCapacity];
            int first = Math.Min(_count, _buffer.Length - _head);
            Array.Copy(_buffer, _head, next, 0, first);
            if (first < _count)
                Array.Copy(_buffer, 0, next, first, _count - first);

            _buffer = next;
            _head = 0;
        }
    }
}
