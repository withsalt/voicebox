namespace VoiceBox.DNR.DPDFNet.Internal
{
    /// <summary>
    /// 流式 ISTFT 合成器 (重叠-相加)。对应参考实现 <c>ISTFTStreamingPostprocess</c>：
    /// 对每帧谱做 irfft、加合成窗，与上一帧的重叠尾部相加，输出最前面的一个 hop。
    ///
    /// OLA 缓冲长度为 win，始终持有"尚未输出的合成尾部"。每帧：
    ///   shifted = [ola[hop:], 0...0]      // 左移一个 hop，尾部补零
    ///   new     = shifted + (irfft(spec) · window)
    ///   输出 new[:hop]，ola = new
    /// 配合 Vorbis 窗 + 50% 重叠，恒等输入下完美重建，固有延迟为一个 hop。
    /// 非线程安全：由上层持锁串行调用。
    /// </summary>
    internal sealed class IstftStreamingSynthesizer
    {
        private readonly int _winLen;
        private readonly int _hopSize;
        private readonly float[] _window;
        private readonly RealDft _dft;
        private readonly float[] _ola;     // 长度 win
        private readonly float[] _frame;   // 复用：irfft 输出

        public IstftStreamingSynthesizer(int winLen, int hopSize, float[] window, RealDft dft)
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
            _ola = new float[winLen];
            _frame = new float[winLen];
        }

        /// <summary>
        /// 输入一帧谱 (<paramref name="re"/> / <paramref name="im"/>)，输出合成的一个 hop 写入 <paramref name="output"/>。
        /// </summary>
        public void Push(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> output)
        {
            if (output.Length != _hopSize)
                throw new ArgumentException($"output length must be {_hopSize}, got {output.Length}.", nameof(output));

            _dft.Inverse(re, im, _frame);

            // shifted = [ola[hop:], 0]：将 ola 左移一个 hop。
            int keep = _winLen - _hopSize;
            Array.Copy(_ola, _hopSize, _ola, 0, keep);
            Array.Clear(_ola, keep, _hopSize);

            // new = shifted + frame·window；输出前 hop，其余留作下次重叠。
            for (int i = 0; i < _winLen; i++)
                _ola[i] += _frame[i] * _window[i];

            _ola.AsSpan(0, _hopSize).CopyTo(output);
        }

        /// <summary>清空 OLA 缓冲，回到初始状态。</summary>
        public void Reset() => Array.Clear(_ola);
    }
}
