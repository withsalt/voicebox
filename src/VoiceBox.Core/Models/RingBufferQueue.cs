namespace VoiceBox.Core.Models
{
    /// <summary>
    /// 环形缓冲队列实现
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class RingBufferQueue<T>
    {
        private readonly T[] _buffer;
        private int _head;
        private int _tail;
        private int _count;
        private readonly object _syncRoot = new object();
        private readonly int _capacity;

        public int Count => Volatile.Read(ref _count);
        public int Capacity => _capacity;

        public RingBufferQueue(int capacity)
        {
            if (capacity <= 0)
                throw new ArgumentException("Capacity must be greater than 0", nameof(capacity));

            _capacity = capacity;
            _buffer = new T[capacity];
            _head = 0;
            _tail = 0;
            _count = 0;
        }

        public void Enqueue(T item)
        {
            lock (_syncRoot)
            {
                InternalEnqueue(item);
            }
        }

        /// <summary>
        /// 将数组中的元素写入队列。
        /// </summary>
        /// <param name="items">要添加的元素数组。</param>
        public void EnqueueRange(T[] items)
        {
            if (items == null || items.Length == 0)
                return;

            EnqueueRange(new ReadOnlySpan<T>(items));
        }

        /// <summary>
        /// 将列表中的元素写入队列。
        /// </summary>
        /// <param name="items"></param>
        public void EnqueueRange(ICollection<T> items)
        {
            if (items == null || items.Count == 0)
                return;

            if (items is T[] array)
            {
                EnqueueRange(array);
                return;
            }

            lock (_syncRoot)
            {
                foreach (var item in items)
                {
                    InternalEnqueue(item);
                }
            }
        }

        /// <summary>
        /// 使用 ReadOnlySpan<T> 实现高效范围写入
        /// </summary>
        /// <param name="items">要添加的元素范围。</param>
        public void EnqueueRange(ReadOnlySpan<T> items)
        {
            if (items.IsEmpty)
            {
                return;
            }

            lock (_syncRoot)
            {
                int itemsToCopy = items.Length;

                if (itemsToCopy > _capacity)
                {
                    items = items.Slice(itemsToCopy - _capacity);
                    itemsToCopy = _capacity;
                }

                int newCount = _count + itemsToCopy;
                if (newCount > _capacity)
                {
                    _head = (_head + (newCount - _capacity)) % _capacity;
                    _count = _capacity;
                }
                else
                {
                    _count = newCount;
                }

                int spaceToEnd = _capacity - _tail;
                if (itemsToCopy <= spaceToEnd)
                {
                    items.CopyTo(_buffer.AsSpan(_tail));
                }
                else
                {
                    int firstPartLength = spaceToEnd;
                    int secondPartLength = itemsToCopy - firstPartLength;

                    items.Slice(0, firstPartLength).CopyTo(_buffer.AsSpan(_tail));
                    items.Slice(firstPartLength).CopyTo(_buffer.AsSpan(0));
                }

                _tail = (_tail + itemsToCopy) % _capacity;
            }
        }

        /// <summary>
        /// 从队列中移除并返回指定数量的元素。
        /// </summary>
        /// <param name="count">要移除的元素数量。</param>
        /// <returns>一个包含从队列头部移除的元素的数组。</returns>
        public T[] DequeueRange(int count)
        {
            // 检查输入参数是否有效
            if (count <= 0)
            {
                return Array.Empty<T>();
            }

            lock (_syncRoot)
            {
                if (_count == 0)
                {
                    return Array.Empty<T>();
                }

                int itemsToDequeue = Math.Min(count, _count);
                T[] result = new T[itemsToDequeue];

                if (_head + itemsToDequeue <= _capacity)
                {
                    // 情况1：数据是连续的，没有环绕
                    // 直接从 _head 开始复制 itemsToDequeue 个元素
                    Array.Copy(_buffer, _head, result, 0, itemsToDequeue);
                }
                else
                {
                    // 情况2：数据是环绕的，分为两部分
                    // 第一部分：从 _head 到数组末尾
                    int firstPartLength = _capacity - _head;
                    Array.Copy(_buffer, _head, result, 0, firstPartLength);

                    // 第二部分：从数组开头到剩余部分
                    int secondPartLength = itemsToDequeue - firstPartLength;
                    Array.Copy(_buffer, 0, result, firstPartLength, secondPartLength);
                }

                // 清理已出队位置的数据，以释放引用
                // 这一步对于引用类型很重要，可以防止内存泄漏
                int start = _head;
                int end = (_head + itemsToDequeue) % _capacity;
                if (start < end)
                {
                    Array.Clear(_buffer, start, itemsToDequeue);
                }
                else // 环绕情况
                {
                    Array.Clear(_buffer, start, _capacity - start);
                    Array.Clear(_buffer, 0, end);
                }

                // 更新头指针和计数
                _head = end;
                _count -= itemsToDequeue;

                return result;
            }
        }

        public bool TryDequeue(out T result)
        {
            lock (_syncRoot)
            {
                if (_count == 0)
                {
                    result = default!;
                    return false;
                }

                result = _buffer[_head];
                _buffer[_head] = default!;
                _head = (_head + 1) % _capacity;
                _count--;
                return true;
            }
        }

        public bool TryPeek(out T result)
        {
            lock (_syncRoot)
            {
                if (_count == 0)
                {
                    result = default!;
                    return false;
                }

                result = _buffer[_head];
                return true;
            }
        }

        public void Clear()
        {
            lock (_syncRoot)
            {
                Array.Clear(_buffer, 0, _buffer.Length);
                _head = 0;
                _tail = 0;
                _count = 0;
            }
        }

        public T[] ToArray()
        {
            lock (_syncRoot)
            {
                if (_count == 0) return Array.Empty<T>();

                T[] result = new T[_count];

                if (_head < _tail)
                {
                    Array.Copy(_buffer, _head, result, 0, _count);
                }
                else
                {
                    int segment1 = _capacity - _head;
                    Array.Copy(_buffer, _head, result, 0, segment1);
                    Array.Copy(_buffer, 0, result, segment1, _count - segment1);
                }

                return result;
            }
        }

        /// <summary>
        /// 将环形缓冲区中的元素复制到现有的一维数组中，从指定的数组索引开始。
        /// </summary>
        /// <param name="array">作为从环形缓冲区复制的元素的目标的一维数组。该数组必须具有从零开始的索引。</param>
        /// <param name="arrayIndex">`array` 中从零开始的索引，应在此处开始复制。</param>
        /// <exception cref="ArgumentNullException">如果 `array` 为 null。</exception>
        /// <exception cref="ArgumentOutOfRangeException">如果 `arrayIndex` 小于 0。</exception>
        /// <exception cref="ArgumentException">如果目标数组 `array` 从 `arrayIndex` 开始没有足够的空间来容纳所有元素。</exception>
        public void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            lock (_syncRoot)
            {
                if (arrayIndex < 0 || arrayIndex > array.Length)
                    throw new ArgumentOutOfRangeException(nameof(arrayIndex), "索引必须为非负数，并且不能大于数组的长度。");

                if (array.Length - arrayIndex < _count)
                    throw new ArgumentException("从指定的数组索引开始，目标数组的空间不足以容纳所有元素。");

                if (_count == 0)
                    return;

                // 情况1：数据是连续的块（头指针在尾指针之前）
                if (_head < _tail)
                {
                    Array.Copy(_buffer, _head, array, arrayIndex, _count);
                }
                else // 情况2：数据环绕（尾指针在头指针之前或重合，但队列不为空）
                {
                    // 复制第一部分：从头指针到内部数组的末尾
                    int firstPartLength = _capacity - _head;
                    Array.Copy(_buffer, _head, array, arrayIndex, firstPartLength);

                    // 复制第二部分：从内部数组的开头到尾指针之前
                    int secondPartLength = _count - firstPartLength;
                    Array.Copy(_buffer, 0, array, arrayIndex + firstPartLength, secondPartLength);
                }
            }
        }

        private void InternalEnqueue(T item)
        {
            // 添加元素到队列尾部
            _buffer[_tail] = item;
            _tail = (_tail + 1) % _capacity;

            // 队列满时移除头部的元素
            if (_count < _capacity)
            {
                _count++;
            }
            else
            {
                // 移动头指针覆盖旧数据
                _head = (_head + 1) % _capacity;
            }
        }
    }
}
