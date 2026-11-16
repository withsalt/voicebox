namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class SampleDelayLine
    {
        private readonly float[] _buffer;
        private int _index;

        public int DelaySamples => _buffer.Length;

        public SampleDelayLine(int delaySamples)
        {
            if (delaySamples < 0) throw new ArgumentOutOfRangeException(nameof(delaySamples));
            _buffer = new float[delaySamples];
        }

        public void Process(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("input and output must have the same length.");

            if (_buffer.Length == 0)
            {
                input.CopyTo(output);
                return;
            }

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = _buffer[_index];
                _buffer[_index] = input[i];
                _index++;
                if (_index == _buffer.Length) _index = 0;
            }
        }

        public void Reset()
        {
            Array.Clear(_buffer);
            _index = 0;
        }
    }
}
