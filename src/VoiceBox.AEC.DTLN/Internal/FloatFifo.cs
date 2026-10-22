namespace VoiceBox.AEC.DTLN.Internal
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
            EnsureCapacityForAdditional(source.Length);

            int tail = (_head + _count) % _buffer.Length;
            int first = Math.Min(source.Length, _buffer.Length - tail);
            source[..first].CopyTo(_buffer.AsSpan(tail));
            if (first < source.Length)
                source[first..].CopyTo(_buffer);
            _count += source.Length;
        }

        public void WriteReplacingNonFinite(ReadOnlySpan<float> source)
        {
            if (source.IsEmpty) return;
            EnsureCapacityForAdditional(source.Length);

            int tail = (_head + _count) % _buffer.Length;
            int first = Math.Min(source.Length, _buffer.Length - tail);
            CopyReplacingNonFinite(source[..first], _buffer.AsSpan(tail, first));
            if (first < source.Length)
                CopyReplacingNonFinite(source[first..], _buffer.AsSpan(0, source.Length - first));
            _count += source.Length;
        }

        public void WriteZeros(int count)
        {
            if (count <= 0) return;
            EnsureCapacityForAdditional(count);

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

        private void EnsureCapacityForAdditional(int additionalCount)
        {
            if (_count > Array.MaxLength - additionalCount)
                throw new InvalidOperationException($"FloatFifo capacity overflow: have {_count}, adding {additionalCount}.");

            EnsureCapacity(_count + additionalCount);
        }

        private void EnsureCapacity(int required)
        {
            if (required <= _buffer.Length) return;

            int newCapacity = _buffer.Length;
            while (newCapacity < required)
            {
                if (newCapacity > Array.MaxLength / 2)
                {
                    newCapacity = Array.MaxLength;
                    break;
                }

                newCapacity *= 2;
            }

            if (newCapacity < required)
                throw new InvalidOperationException($"FloatFifo capacity overflow: required {required}.");

            var next = new float[newCapacity];
            int first = Math.Min(_count, _buffer.Length - _head);
            Array.Copy(_buffer, _head, next, 0, first);
            if (first < _count)
                Array.Copy(_buffer, 0, next, first, _count - first);
            _buffer = next;
            _head = 0;
        }

        private static void CopyReplacingNonFinite(ReadOnlySpan<float> source, Span<float> destination)
        {
            for (int i = 0; i < source.Length; i++)
            {
                float sample = source[i];
                destination[i] = float.IsFinite(sample) ? sample : 0f;
            }
        }
    }
}
