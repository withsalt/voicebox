namespace VoiceBox.DNR.DPDFNet.Internal
{
    /// <summary>
    /// 流式 STFT 分析器。对应参考实现 <c>STFTStreamingPreprocess</c>：
    /// 维护一个长度为 win 的滑动缓冲，每次推进一个 hop 的新样本，加窗后做 rfft。
    ///
    /// 内部缓冲布局：buffer 始终持有"最近 win 个样本"。每帧将旧样本左移一个 hop，
    /// 再把新 hop 拷到尾部，因此相邻帧重叠 (win - hop) 个样本。
    /// 非线程安全：由上层 <see cref="DPDFNetDNRService"/> 持锁串行调用。
    /// </summary>
    internal sealed class StftStreamingAnalyzer
    {
        private readonly int _winLen;
        private readonly int _hopSize;
        private readonly float[] _window;
        private readonly RealDft _dft;
        private readonly float[] _buffer;   // 长度 win
        private readonly float[] _frame;    // 复用：加窗后的帧

        public int BinCount => _dft.BinCount;
        public RealDft Dft => _dft;

        public StftStreamingAnalyzer(int winLen, int hopSize, float[] window, RealDft dft)
        {
            ArgumentNullException.ThrowIfNull(window);
            ArgumentNullException.ThrowIfNull(dft);
            if (window.Length != winLen)
                throw new ArgumentException($"window length {window.Length} != winLen {winLen}.", nameof(window));
            if (dft.FftSize != winLen)
                throw new ArgumentException($"dft.FftSize {dft.FftSize} != winLen {winLen}.", nameof(dft));
            if (hopSize <= 0 || hopSize > winLen)
                throw new ArgumentOutOfRangeException(nameof(hopSize), hopSize, "hopSize must be in (0, winLen].");

            _winLen = winLen;
            _hopSize = hopSize;
            _window = window;
            _dft = dft;
            _buffer = new float[winLen];
            _frame = new float[winLen];
        }

        /// <summary>
        /// 推进一帧 (<paramref name="hop"/> 长度须为 hopSize)，输出当前谱写入 <paramref name="re"/> / <paramref name="im"/>。
        /// </summary>
        public void Push(ReadOnlySpan<float> hop, Span<float> re, Span<float> im)
        {
            if (hop.Length != _hopSize)
                throw new ArgumentException($"hop length must be {_hopSize}, got {hop.Length}.", nameof(hop));

            // 左移 (win - hop) 个旧样本，尾部追加新 hop。
            int keep = _winLen - _hopSize;
            Array.Copy(_buffer, _hopSize, _buffer, 0, keep);
            hop.CopyTo(_buffer.AsSpan(keep, _hopSize));

            for (int i = 0; i < _winLen; i++)
                _frame[i] = _buffer[i] * _window[i];

            _dft.Forward(_frame, re, im);
        }

        /// <summary>清空滑动缓冲，回到初始状态。</summary>
        public void Reset() => Array.Clear(_buffer);
    }
}
