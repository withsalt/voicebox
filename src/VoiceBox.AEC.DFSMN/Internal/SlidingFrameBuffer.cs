namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class SlidingFrameBuffer
    {
        private readonly float[] _buffer;
        private readonly int _frameLength;
        private readonly int _hopLength;
        private int _filled;
        private int _start;

        public SlidingFrameBuffer(int frameLength, int hopLength)
        {
            if (frameLength <= 0) throw new ArgumentOutOfRangeException(nameof(frameLength));
            if (hopLength <= 0 || hopLength > frameLength) throw new ArgumentOutOfRangeException(nameof(hopLength));
            _frameLength = frameLength;
            _hopLength = hopLength;
            _buffer = new float[frameLength * 4];
        }

        public bool HasFrame => _filled >= _frameLength;

        public ReadOnlySpan<float> Frame => HasFrame
            ? _buffer.AsSpan(_start, _frameLength)
            : throw new InvalidOperationException("A complete frame is not available yet.");

        public bool Push(ReadOnlySpan<float> hop)
        {
            if (hop.Length != _hopLength)
                throw new ArgumentException($"hop length must be {_hopLength}.", nameof(hop));

            if (_filled < _frameLength)
            {
                hop.CopyTo(_buffer.AsSpan(_filled));
                int total = _filled + hop.Length;
                if (total >= _frameLength)
                {
                    _start = total - _frameLength;
                    _filled = _frameLength;
                }
                else
                {
                    _filled = total;
                }
            }
            else if (_hopLength == _frameLength)
            {
                hop.CopyTo(_buffer.AsSpan(0, _frameLength));
                _start = 0;
            }
            else
            {
                int tail = _start + _frameLength;
                if (tail + _hopLength > _buffer.Length)
                {
                    _buffer.AsSpan(_start, _frameLength).CopyTo(_buffer);
                    _start = 0;
                    tail = _frameLength;
                }

                hop.CopyTo(_buffer.AsSpan(tail, _hopLength));
                _start += _hopLength;
            }

            return HasFrame;
        }

        public void Reset()
        {
            _buffer.AsSpan(0, _frameLength).Clear();
            _filled = 0;
            _start = 0;
        }
    }
}
