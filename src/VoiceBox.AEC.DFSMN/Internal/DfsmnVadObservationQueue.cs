namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class DfsmnVadObservationQueue
    {
        private readonly float[] _audio;
        private readonly float[] _probabilities;
        private readonly bool[] _hasAudio;
        private readonly VoiceBox.Core.Services.AudioService.Models.CaptureRange[] _ranges;
        private readonly int _capacityHops;
        private readonly int _hopSamples;
        private int _head;
        private int _count;

        public DfsmnVadObservationQueue(int capacityHops, int hopSamples)
        {
            if (capacityHops <= 0) throw new ArgumentOutOfRangeException(nameof(capacityHops));
            if (hopSamples <= 0) throw new ArgumentOutOfRangeException(nameof(hopSamples));

            _capacityHops = capacityHops;
            _hopSamples = hopSamples;
            _audio = new float[capacityHops * hopSamples];
            _probabilities = new float[capacityHops];
            _hasAudio = new bool[capacityHops];
            _ranges = new VoiceBox.Core.Services.AudioService.Models.CaptureRange[capacityHops];
        }

        public int Count => _count;

        public void Enqueue(
            float probability,
            ReadOnlySpan<float> audioHop,
            in VoiceBox.Core.Services.AudioService.Models.CaptureRange captureRange)
        {
            if (audioHop.Length != _hopSamples)
                throw new ArgumentException($"audioHop length must be {_hopSamples}.", nameof(audioHop));

            int slot;
            if (_count == _capacityHops)
            {
                slot = _head;
                _head = (_head + 1) % _capacityHops;
            }
            else
            {
                slot = (_head + _count) % _capacityHops;
                _count++;
            }

            _probabilities[slot] = probability;
            _hasAudio[slot] = true;
            _ranges[slot] = captureRange;
            audioHop.CopyTo(_audio.AsSpan(slot * _hopSamples, _hopSamples));
        }

        public void EnqueueProbability(
            float probability,
            in VoiceBox.Core.Services.AudioService.Models.CaptureRange captureRange)
        {
            int slot;
            if (_count == _capacityHops)
            {
                slot = _head;
                _head = (_head + 1) % _capacityHops;
            }
            else
            {
                slot = (_head + _count) % _capacityHops;
                _count++;
            }

            _probabilities[slot] = probability;
            _hasAudio[slot] = false;
            _ranges[slot] = captureRange;
        }

        public bool TryDequeue(
            out float probability,
            Span<float> audioHop,
            out bool hasAudio,
            out VoiceBox.Core.Services.AudioService.Models.CaptureRange captureRange)
        {
            if (audioHop.Length != _hopSamples)
                throw new ArgumentException($"audioHop length must be {_hopSamples}.", nameof(audioHop));

            if (_count == 0)
            {
                probability = 0f;
                hasAudio = false;
                captureRange = default;
                return false;
            }

            int slot = _head;
            probability = _probabilities[slot];
            hasAudio = _hasAudio[slot];
            captureRange = _ranges[slot];
            if (hasAudio)
                _audio.AsSpan(slot * _hopSamples, _hopSamples).CopyTo(audioHop);
            _head = (_head + 1) % _capacityHops;
            _count--;
            if (_count == 0) _head = 0;
            return true;
        }

        public void Clear()
        {
            _head = 0;
            _count = 0;
        }
    }
}
