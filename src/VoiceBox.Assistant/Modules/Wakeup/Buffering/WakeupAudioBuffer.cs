namespace VoiceBox.Assistant.Modules.Wakeup.Buffering
{
    /// <summary>
    /// 固定容量的 float PCM 环形缓冲。用于休眠期保留最近一小段音频,
    /// 唤醒命中后补送给 VAD, 避免"唤醒词 + 后续命令"同句时丢掉后半句。
    /// </summary>
    internal sealed class WakeupAudioBuffer
    {
        private readonly object _lock = new();
        private readonly float[] _buffer;
        private int _writeIndex;
        private int _count;

        public WakeupAudioBuffer(int capacitySamples)
        {
            if (capacitySamples < 0)
                throw new ArgumentOutOfRangeException(nameof(capacitySamples), "Capacity must be >= 0.");

            _buffer = capacitySamples == 0 ? Array.Empty<float>() : new float[capacitySamples];
        }

        public int Count
        {
            get { lock (_lock) return _count; }
        }

        public void Append(ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty || _buffer.Length == 0)
                return;

            lock (_lock)
            {
                if (audio.Length >= _buffer.Length)
                {
                    audio.Slice(audio.Length - _buffer.Length).CopyTo(_buffer);
                    _writeIndex = 0;
                    _count = _buffer.Length;
                    return;
                }

                int first = Math.Min(audio.Length, _buffer.Length - _writeIndex);
                audio.Slice(0, first).CopyTo(_buffer.AsSpan(_writeIndex, first));
                int remaining = audio.Length - first;
                if (remaining > 0)
                    audio.Slice(first, remaining).CopyTo(_buffer.AsSpan(0, remaining));

                _writeIndex = (_writeIndex + audio.Length) % _buffer.Length;
                _count = Math.Min(_buffer.Length, _count + audio.Length);
            }
        }

        public float[] SnapshotAndClear()
        {
            lock (_lock)
            {
                float[] result = SnapshotLocked();
                ClearLocked();
                return result;
            }
        }

        public void Clear()
        {
            lock (_lock)
                ClearLocked();
        }

        private void ClearLocked()
        {
            _writeIndex = 0;
            _count = 0;
        }

        private float[] SnapshotLocked()
        {
            if (_count == 0)
                return Array.Empty<float>();

            var result = new float[_count];
            int start = (_writeIndex - _count + _buffer.Length) % _buffer.Length;
            int first = Math.Min(_count, _buffer.Length - start);
            Array.Copy(_buffer, start, result, 0, first);
            int remaining = _count - first;
            if (remaining > 0)
                Array.Copy(_buffer, 0, result, first, remaining);
            return result;
        }
    }
}
