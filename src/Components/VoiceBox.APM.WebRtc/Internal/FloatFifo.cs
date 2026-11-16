namespace VoiceBox.APM.WebRtc.Internal
{
    /// <summary>
    /// 轻量数组环形缓冲（单生产者-单消费者，同一线程使用）。
    /// 用于把任意帧长的音频重分块成固定大小的处理块。预分配容量，必要时翻倍扩容。
    /// </summary>
    internal sealed class FloatFifo
    {
        private float[] _buf;
        private int _head;   // 读位置
        private int _count;  // 当前样本数

        public FloatFifo(int capacity)
        {
            _buf = new float[Math.Max(capacity, 16)];
        }

        public int Count => _count;

        /// <summary>写入一段样本。</summary>
        public void Write(ReadOnlySpan<float> src)
        {
            if (src.IsEmpty) return;
            EnsureCapacity(_count + src.Length);

            int tail = (_head + _count) % _buf.Length;
            int firstChunk = Math.Min(src.Length, _buf.Length - tail);
            src.Slice(0, firstChunk).CopyTo(_buf.AsSpan(tail));
            if (firstChunk < src.Length)
                src.Slice(firstChunk).CopyTo(_buf.AsSpan(0));

            _count += src.Length;
        }

        /// <summary>写入 <paramref name="count"/> 个值为 0 的样本（用于预填延迟）。</summary>
        public void WriteZeros(int count)
        {
            if (count <= 0) return;
            EnsureCapacity(_count + count);

            int tail = (_head + _count) % _buf.Length;
            for (int i = 0; i < count; i++)
            {
                _buf[tail] = 0f;
                tail = (tail + 1) % _buf.Length;
            }
            _count += count;
        }

        /// <summary>读取 <paramref name="dst"/>.Length 个样本到 dst。要求 <see cref="Count"/> 足够。</summary>
        public void Read(Span<float> dst)
        {
            if (dst.Length > _count)
                throw new InvalidOperationException($"FloatFifo underflow: need {dst.Length}, have {_count}");

            int firstChunk = Math.Min(dst.Length, _buf.Length - _head);
            _buf.AsSpan(_head, firstChunk).CopyTo(dst);
            if (firstChunk < dst.Length)
                _buf.AsSpan(0, dst.Length - firstChunk).CopyTo(dst.Slice(firstChunk));

            _head = (_head + dst.Length) % _buf.Length;
            _count -= dst.Length;
        }

        public void Clear()
        {
            _head = 0;
            _count = 0;
        }

        private void EnsureCapacity(int required)
        {
            if (required <= _buf.Length) return;

            int newCap = _buf.Length * 2;
            while (newCap < required) newCap *= 2;

            var newBuf = new float[newCap];
            // 把现有数据线性化拷到新缓冲头部
            int firstChunk = Math.Min(_count, _buf.Length - _head);
            Array.Copy(_buf, _head, newBuf, 0, firstChunk);
            if (firstChunk < _count)
                Array.Copy(_buf, 0, newBuf, firstChunk, _count - firstChunk);

            _buf = newBuf;
            _head = 0;
        }
    }
}
