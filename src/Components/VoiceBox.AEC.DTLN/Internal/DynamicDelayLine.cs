namespace VoiceBox.AEC.DTLN.Internal
{
    /// <summary>
    /// 可变长度延迟线,支持运行时调整延迟。
    /// 延迟切换时在 <see cref="TransitionSamples"/> 个样本内做交叉淡化,避免咔哒音。
    /// 仅在单一实时回调线程访问,非线程安全。
    /// </summary>
    internal sealed class DynamicDelayLine
    {
        private const int TransitionSamples = 64; // 4ms @ 16kHz,平滑切换窗口

        private readonly float[] _buffer;
        private readonly int _maxDelay;
        private int _writeIndex;
        private int _currentDelay;
        private int _targetDelay;
        private int _transitionRemaining;
        // 累计已写入样本数 (不绕回)。冷启动判定用它而非会绕回的 _writeIndex: 流式中段 _writeIndex
        // 绕回到小值时, 旧条件 `_writeIndex < _targetDelay` 会误判为冷启动, 跳过交叉淡化产生咔哒。
        private long _totalWritten;

        public int CurrentDelay => _currentDelay;
        public int MaxDelay => _maxDelay;

        public DynamicDelayLine(int initialDelay, int maxDelay)
        {
            if (maxDelay < 0)
                throw new ArgumentOutOfRangeException(nameof(maxDelay));
            if (initialDelay < 0 || initialDelay > maxDelay)
                throw new ArgumentOutOfRangeException(nameof(initialDelay));

            _maxDelay = maxDelay;
            _buffer = new float[maxDelay + TransitionSamples + 1];
            _currentDelay = initialDelay;
            _targetDelay = initialDelay;
        }

        /// <summary>
        /// 设置目标延迟。实际延迟在接下来的 <see cref="TransitionSamples"/> 个样本内平滑过渡到目标值。
        /// 超出 [0, MaxDelay] 范围或与当前目标相同则忽略。
        /// </summary>
        public void SetDelay(int newDelay)
        {
            if (newDelay < 0 || newDelay > _maxDelay)
                return;
            if (newDelay == _targetDelay)
                return;

            _targetDelay = newDelay;
            _transitionRemaining = TransitionSamples;
        }

        public void Process(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output must have the same length.");

            int len = _buffer.Length;
            for (int i = 0; i < input.Length; i++)
            {
                // 先写入当前样本,再读取其后方 delay 个样本的历史值(delay=0 即直通)
                _buffer[_writeIndex] = input[i];

                int readIndex = (_writeIndex - _currentDelay + len) % len;
                float delayedSample = _buffer[readIndex];

                if (_transitionRemaining > 0)
                {
                    // 冷启动时缓冲未填满:直接跳到目标延迟,跳过平滑(避免混入未初始化的 0)。
                    // 用不绕回的 _totalWritten 判定 (流式中段 _writeIndex 绕回小值时旧条件会误触发)。
                    if (_totalWritten < _targetDelay)
                    {
                        _currentDelay = _targetDelay;
                        _transitionRemaining = 0;
                        readIndex = (_writeIndex - _currentDelay + len) % len;
                        delayedSample = _buffer[readIndex];
                    }
                    else
                    {
                        int targetReadIndex = (_writeIndex - _targetDelay + len) % len;
                        float targetSample = _buffer[targetReadIndex];

                        float blend = 1f - (float)_transitionRemaining / TransitionSamples;
                        delayedSample = delayedSample * (1f - blend) + targetSample * blend;

                        _transitionRemaining--;
                        if (_transitionRemaining == 0)
                            _currentDelay = _targetDelay;
                    }
                }

                output[i] = delayedSample;
                _writeIndex = (_writeIndex + 1) % len;
                _totalWritten++;
            }
        }

        public void Reset()
        {
            Array.Clear(_buffer);
            _writeIndex = 0;
            _totalWritten = 0;
            _transitionRemaining = 0;
            _targetDelay = _currentDelay;
        }
    }
}
