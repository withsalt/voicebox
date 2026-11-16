namespace VoiceBox.AEC.DTLN.Internal
{
    internal sealed class SampleDelayLine
    {
        private readonly float[] _buffer;
        private int _index;

        public SampleDelayLine(int delaySamples)
        {
            if (delaySamples < 0) throw new ArgumentOutOfRangeException(nameof(delaySamples));
            _buffer = new float[delaySamples];
        }

        public ReadOnlySpan<float> Process(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("input and output must have the same length.");

            if (_buffer.Length == 0)
                return input;

            int remaining = input.Length;
            int sourceOffset = 0;
            while (remaining > 0)
            {
                int count = Math.Min(remaining, _buffer.Length - _index);
                _buffer.AsSpan(_index, count).CopyTo(output.Slice(sourceOffset, count));
                input.Slice(sourceOffset, count).CopyTo(_buffer.AsSpan(_index, count));
                sourceOffset += count;
                remaining -= count;
                _index += count;
                if (_index == _buffer.Length) _index = 0;
            }

            return output;
        }

        public void Reset()
        {
            Array.Clear(_buffer);
            _index = 0;
        }
    }
}
