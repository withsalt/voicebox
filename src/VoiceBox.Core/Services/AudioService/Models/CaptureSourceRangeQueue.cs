namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Maps output from a buffered streaming processor to the oldest unconsumed source
    /// samples on the capture timeline.
    /// </summary>
    public sealed class CaptureSourceRangeQueue
    {
        private readonly Queue<CaptureRange> _segments;
        private long _pendingSamples;
        private long _headConsumedSamples;
        private CaptureRange _lastEnqueuedRange;

        public CaptureSourceRangeQueue(int initialCapacity = 0)
        {
            if (initialCapacity < 0)
                throw new ArgumentOutOfRangeException(nameof(initialCapacity));

            _segments = new Queue<CaptureRange>(initialCapacity);
        }

        public long PendingSamples => _pendingSamples;

        public void Enqueue(in CaptureRange range)
        {
            if (!range.IsValid || range.Length <= 0)
                throw new ArgumentOutOfRangeException(nameof(range), "Capture source range must be non-empty and valid.");
            if (_pendingSamples > 0 && !_lastEnqueuedRange.IsContiguousWith(in range))
            {
                throw new InvalidOperationException(
                    "Capture source ranges must be contiguous and belong to one stream epoch.");
            }

            long pendingSamples = checked(_pendingSamples + range.Length);
            _segments.Enqueue(range);
            _pendingSamples = pendingSamples;
            _lastEnqueuedRange = range;
        }

        public bool TryDequeue(int sampleCount, out CaptureRange range)
        {
            range = default;
            if (sampleCount <= 0 || _pendingSamples < sampleCount || _segments.Count == 0)
                return false;

            CaptureRange first = _segments.Peek();
            long startSample = first.StartSample + _headConsumedSamples;
            long startSequence = first.StartSequenceNumber;
            long endSample = startSample;
            long endSequence = startSequence;
            long remaining = sampleCount;

            while (remaining > 0)
            {
                CaptureRange segment = _segments.Peek();
                long available = segment.Length - _headConsumedSamples;
                long consumedNow = Math.Min(available, remaining);
                _headConsumedSamples += consumedNow;
                remaining -= consumedNow;
                endSample = segment.StartSample + _headConsumedSamples;
                endSequence = segment.EndSequenceNumber;

                if (_headConsumedSamples == segment.Length)
                {
                    _segments.Dequeue();
                    _headConsumedSamples = 0;
                }
            }

            _pendingSamples -= sampleCount;
            range = new CaptureRange(
                first.StreamEpoch,
                startSequence,
                endSequence,
                startSample,
                endSample,
                first.TimestampTicks);
            return true;
        }

        public void Clear()
        {
            _segments.Clear();
            _pendingSamples = 0;
            _headConsumedSamples = 0;
            _lastEnqueuedRange = default;
        }
    }
}
