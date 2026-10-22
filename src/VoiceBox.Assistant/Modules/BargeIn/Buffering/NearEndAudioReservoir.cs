using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Buffering
{
    /// <summary>
    /// Bounded capture-timeline reservoir. Snapshot requests are exact: data is never
    /// silently shifted to a newer retained tail, because that would authorize one
    /// speaker interval while replaying another interval to ASR.
    /// </summary>
    internal sealed class NearEndAudioReservoir
    {
        private readonly object _lock = new();
        private readonly float[] _buffer;
        private int _writeIndex;
        private int _count;
        private long _streamEpoch;
        private long _endSample;
        private bool _hasTimeline;

        public NearEndAudioReservoir(int capacitySamples)
            => _buffer = capacitySamples <= 0 ? Array.Empty<float>() : new float[capacitySamples];

        /// <summary>Returns true when an existing timeline had to be discarded.</summary>
        public bool Add(
            ReadOnlySpan<float> audio,
            in CaptureRange sourceRange,
            bool hasDiscontinuity = false)
        {
            if (audio.Length != sourceRange.Length || !sourceRange.IsValid)
                throw new ArgumentException("Audio and capture source range must describe the same valid interval.");
            if (audio.IsEmpty)
                return false;

            lock (_lock)
            {
                bool resetExisting = _hasTimeline &&
                    (hasDiscontinuity ||
                     _streamEpoch != sourceRange.StreamEpoch ||
                     _endSample != sourceRange.StartSample);
                if (!_hasTimeline || resetExisting)
                {
                    ClearLocked();
                    _hasTimeline = true;
                    _streamEpoch = sourceRange.StreamEpoch;
                    _endSample = sourceRange.StartSample;
                }

                _endSample = sourceRange.EndSampleExclusive;
                if (_buffer.Length == 0)
                    return resetExisting;

                if (audio.Length >= _buffer.Length)
                {
                    audio[^_buffer.Length..].CopyTo(_buffer);
                    _writeIndex = 0;
                    _count = _buffer.Length;
                    return resetExisting;
                }

                int first = Math.Min(audio.Length, _buffer.Length - _writeIndex);
                audio[..first].CopyTo(_buffer.AsSpan(_writeIndex, first));
                int remaining = audio.Length - first;
                if (remaining > 0)
                    audio[first..].CopyTo(_buffer.AsSpan(0, remaining));
                _writeIndex = (_writeIndex + audio.Length) % _buffer.Length;
                _count = Math.Min(_buffer.Length, _count + audio.Length);
                return resetExisting;
            }
        }

        public bool TryGetBounds(out long streamEpoch, out long retainedStartSample, out long endSample)
        {
            lock (_lock)
            {
                streamEpoch = _streamEpoch;
                endSample = _endSample;
                retainedStartSample = _endSample - _count;
                return _hasTimeline;
            }
        }

        /// <summary>
        /// Locates an unchanged VAD output chunk in the retained main stream. VAD can
        /// emit pre-roll or an internal frame before the end of the current Predict call,
        /// so assuming every notification is the current reservoir tail is not aligned.
        /// The newest exact occurrence wins when repeated samples are ambiguous.
        /// </summary>
        public bool TryLocate(
            ReadOnlySpan<float> audio,
            out long streamEpoch,
            out long startSample,
            out long endSample)
        {
            streamEpoch = 0;
            startSample = 0;
            endSample = 0;
            if (audio.IsEmpty)
                return false;

            lock (_lock)
            {
                if (!_hasTimeline || audio.Length > _count || _buffer.Length == 0)
                    return false;

                int oldestIndex = (_writeIndex - _count + _buffer.Length) % _buffer.Length;
                for (int offset = _count - audio.Length; offset >= 0; offset--)
                {
                    if (!SamplesEqual(GetSample(oldestIndex, offset), audio[0]) ||
                        !SamplesEqual(GetSample(oldestIndex, offset + audio.Length - 1), audio[^1]))
                    {
                        continue;
                    }

                    bool matches = true;
                    for (int i = 1; i < audio.Length - 1; i++)
                    {
                        if (!SamplesEqual(GetSample(oldestIndex, offset + i), audio[i]))
                        {
                            matches = false;
                            break;
                        }
                    }
                    if (!matches)
                        continue;

                    streamEpoch = _streamEpoch;
                    startSample = _endSample - _count + offset;
                    endSample = startSample + audio.Length;
                    return true;
                }
                return false;
            }
        }

        public IReadOnlyList<float[]> SnapshotLast(int maxSamples)
        {
            if (maxSamples <= 0)
                return [];
            lock (_lock)
                return SnapshotLocked(_endSample - Math.Min(maxSamples, _count));
        }

        public bool TrySnapshotFrom(
            long streamEpoch,
            long startSample,
            int maxSamples,
            out IReadOnlyList<float[]> audio)
        {
            audio = [];
            if (startSample < 0 || maxSamples <= 0)
                return false;

            lock (_lock)
            {
                long retainedStart = _endSample - _count;
                long length = _endSample - startSample;
                if (!_hasTimeline || streamEpoch != _streamEpoch || startSample < retainedStart ||
                    startSample > _endSample || length > maxSamples)
                {
                    return false;
                }

                audio = SnapshotLocked(startSample);
                return true;
            }
        }

        /// <summary>
        /// Copies one exact capture-timeline interval. Unlike <see cref="TrySnapshotFrom"/>,
        /// later retained samples do not make a bounded interval unavailable.
        /// </summary>
        public bool TrySnapshotRange(
            long streamEpoch,
            long startSample,
            long endSampleExclusive,
            out float[] audio)
        {
            audio = [];
            if (startSample < 0 || endSampleExclusive <= startSample)
                return false;

            lock (_lock)
            {
                long retainedStart = _endSample - _count;
                if (!_hasTimeline || streamEpoch != _streamEpoch ||
                    startSample < retainedStart || endSampleExclusive > _endSample)
                {
                    return false;
                }

                int take = checked((int)(endSampleExclusive - startSample));
                audio = new float[take];
                int oldestIndex = (_writeIndex - _count + _buffer.Length) % _buffer.Length;
                int offset = checked((int)(startSample - retainedStart));
                int start = (oldestIndex + offset) % _buffer.Length;
                int first = Math.Min(take, _buffer.Length - start);
                Array.Copy(_buffer, start, audio, 0, first);
                // first is bounded by take; a non-equal value is exactly the wrap case.
                if (take != first)
                    Array.Copy(_buffer, 0, audio, first, take - first);
                return true;
            }
        }

        public void Clear()
        {
            lock (_lock)
                ClearLocked();
        }

        private IReadOnlyList<float[]> SnapshotLocked(long startSample)
        {
            int take = checked((int)Math.Max(0, _endSample - startSample));
            if (take <= 0)
                return [];

            var result = new float[take];
            long retainedStart = _endSample - _count;
            int oldestIndex = (_writeIndex - _count + _buffer.Length) % _buffer.Length;
            int offset = checked((int)(startSample - retainedStart));
            int start = (oldestIndex + offset) % _buffer.Length;
            int first = Math.Min(take, _buffer.Length - start);
            Array.Copy(_buffer, start, result, 0, first);
            if (take > first)
                Array.Copy(_buffer, 0, result, first, take - first);
            return [result];
        }

        private void ClearLocked()
        {
            _writeIndex = 0;
            _count = 0;
            _streamEpoch = 0;
            _endSample = 0;
            _hasTimeline = false;
        }

        private float GetSample(int oldestIndex, int offset)
            => _buffer[(oldestIndex + offset) % _buffer.Length];

        private static bool SamplesEqual(float left, float right)
            => BitConverter.SingleToInt32Bits(left) == BitConverter.SingleToInt32Bits(right);
    }
}
