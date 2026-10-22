namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Stateful mapper for processors that return one output sample per input sample after an
    /// initial algorithmic delay. The mapper is allocation-free after its source queue reaches
    /// the constructor's bounded callback capacity.
    /// </summary>
    public sealed class BufferedCaptureTimelineMapper
    {
        // WebRTC APM buffers one 10 ms block. 512 callback slots cover that delay even for
        // a pathological one-sample callback at the highest supported 48 kHz sample rate.
        private const int DefaultCallbackCapacity = 512;

        private readonly CaptureSourceRangeQueue _sourceRanges;
        private bool _hasConfiguration;
        private bool _buffered;
        private int _algorithmicDelaySamples;
        private int _remainingSyntheticSamples;
        private bool _pendingOutputDiscontinuity;

        public BufferedCaptureTimelineMapper(int initialCallbackCapacity = DefaultCallbackCapacity)
        {
            if (initialCallbackCapacity <= 0)
                throw new ArgumentOutOfRangeException(nameof(initialCallbackCapacity));

            _sourceRanges = new CaptureSourceRangeQueue(initialCallbackCapacity);
        }

        public BufferedCaptureOutput Map(
            in CaptureRange inputRange,
            int outputSampleCount,
            bool buffered,
            int algorithmicDelaySamples,
            bool hasDiscontinuity = false)
        {
            if (!inputRange.IsValid || inputRange.Length <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputRange));
            if (outputSampleCount <= 0 || outputSampleCount != inputRange.Length)
                throw new ArgumentOutOfRangeException(
                    nameof(outputSampleCount),
                    "Buffered capture output must contain one interleaved sample per input sample.");
            if (algorithmicDelaySamples < 0)
                throw new ArgumentOutOfRangeException(nameof(algorithmicDelaySamples));

            buffered &= algorithmicDelaySamples > 0;
            bool configurationChanged =
                _hasConfiguration &&
                (_buffered != buffered ||
                 _algorithmicDelaySamples != algorithmicDelaySamples);
            bool reset = hasDiscontinuity || configurationChanged;
            if (!_hasConfiguration || reset)
                BeginConfiguration(buffered, algorithmicDelaySamples);
            if (reset)
                _pendingOutputDiscontinuity = true;

            _hasConfiguration = true;

            if (!buffered)
            {
                _sourceRanges.Clear();
                bool outputDiscontinuity = ConsumeOutputDiscontinuity();
                return new BufferedCaptureOutput(
                    CaptureSignalTimeline.FromWholeBuffer(in inputRange, outputSampleCount),
                    outputDiscontinuity,
                    reset);
            }

            _sourceRanges.Enqueue(in inputRange);
            int syntheticPrefix = Math.Min(_remainingSyntheticSamples, outputSampleCount);
            _remainingSyntheticSamples -= syntheticPrefix;
            int validSamples = outputSampleCount - syntheticPrefix;
            if (validSamples <= 0)
            {
                return new BufferedCaptureOutput(
                    default,
                    reset,
                    reset);
            }

            if (!_sourceRanges.TryDequeue(validSamples, out CaptureRange outputRange))
            {
                Reset();
                _pendingOutputDiscontinuity = true;
                return new BufferedCaptureOutput(default, true, true);
            }

            bool hasOutputDiscontinuity = ConsumeOutputDiscontinuity();
            return new BufferedCaptureOutput(
                new CaptureSignalTimeline(outputRange, syntheticPrefix, validSamples),
                hasOutputDiscontinuity,
                reset);
        }

        public void Reset()
        {
            _sourceRanges.Clear();
            _hasConfiguration = false;
            _buffered = false;
            _algorithmicDelaySamples = 0;
            _remainingSyntheticSamples = 0;
            _pendingOutputDiscontinuity = false;
        }

        private void BeginConfiguration(bool buffered, int algorithmicDelaySamples)
        {
            _sourceRanges.Clear();
            _buffered = buffered;
            _algorithmicDelaySamples = algorithmicDelaySamples;
            _remainingSyntheticSamples = buffered ? algorithmicDelaySamples : 0;
        }

        private bool ConsumeOutputDiscontinuity()
        {
            bool value = _pendingOutputDiscontinuity;
            _pendingOutputDiscontinuity = false;
            return value;
        }
    }
}
