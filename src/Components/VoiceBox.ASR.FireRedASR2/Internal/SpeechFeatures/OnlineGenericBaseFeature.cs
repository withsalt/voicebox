// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures
{
    /// <summary>
    /// Holds per-frame feature vectors with a sliding-window discard policy. The legacy
    /// implementation used <c>List&lt;List&lt;float&gt;&gt;</c> and <c>List.RemoveAt(0)</c>
    /// which made every <see cref="Pop"/> an O(n) shift. This version stores raw
    /// <c>float[]</c> arrays directly (no per-frame ToArray) and uses a head index to make
    /// pops O(1); the backing list is compacted lazily when the head consumes a meaningful
    /// fraction of it.
    /// </summary>
    class RecyclingVector
    {
        private readonly List<float[]> _items;
        private readonly int _itemsToHold;
        // Absolute frame counters use long so 24×7 services don't overflow after ~248 days
        // (100 frames/sec @ 10 ms shift). _head stays int because the in-memory list is
        // bounded by the compaction policy; only the cumulative counters can grow without
        // limit.
        private long _firstAvailableIndex;
        private int _head;              // index into _items of the first live element

        public RecyclingVector(int itemsToHold = -1)
        {
            _itemsToHold = itemsToHold == 0 ? -1 : itemsToHold;
            _firstAvailableIndex = 0;
            _head = 0;
            _items = new List<float[]>();
        }

        public float[] At(long index)
        {
            if (index < _firstAvailableIndex)
                throw new Exception($"Attempted to retrieve feature vector that was already removed by the RecyclingVector (index = {index}; first_available_index = {_firstAvailableIndex}; size = {Size()})");
            long local = (index - _firstAvailableIndex) + _head;
            return _items[checked((int)local)];
        }

        public void PushBack(float[] item)
        {
            int liveCount = _items.Count - _head;
            if (_itemsToHold > 0 && liveCount >= _itemsToHold)
            {
                // At capacity: drop the oldest live item by advancing head (O(1)).
                _head++;
                _firstAvailableIndex++;
            }
            _items.Add(item);
            MaybeCompact();
        }

        public long Size()
        {
            return _firstAvailableIndex + (_items.Count - _head);
        }

        public void Pop(int n)
        {
            int available = _items.Count - _head;
            int count = n < available ? n : available;
            _head += count;
            _firstAvailableIndex += count;
            MaybeCompact();
        }

        private void MaybeCompact()
        {
            // Keep memory bounded: when the dead prefix dominates, collapse it in one shot.
            if (_head > 256 && _head * 2 >= _items.Count)
            {
                _items.RemoveRange(0, _head);
                _head = 0;
            }
        }
    }

    public class OnlineGenericBaseFeature<T> where T : IFeatureComputer
    {
        private readonly T computer;
        private readonly FeatureWindowFunction windowFunction;
        private readonly RecyclingVector features;
        private bool inputFinished;
        private long waveformOffset;
        // Backing buffer + live count. Replaces the legacy List<float> which allocated and
        // copied on every AddRange / discard pass.
        private float[] _waveformRemainder = Array.Empty<float>();
        private int _remainderCount;
        // Padded-window scratch reused across frames; recreated only when frame size changes.
        private float[] _paddedFrame;

        public OnlineGenericBaseFeature(T opts)
        {
            computer = opts;
            windowFunction = new FeatureWindowFunction(computer.GetFrameOptions());
            inputFinished = false;
            waveformOffset = 0;
            features = new RecyclingVector();
            _paddedFrame = new float[computer.GetFrameOptions().PaddedWindowSize()];

            if (computer.GetFrameOptions().MaxFeatureVectors <= 200)
                throw new Exception("Online feature extraction requires more than 200 max feature vectors.");
        }

        public int Dim()
        {
            return computer.Dim();
        }

        public float FrameShiftInSeconds()
        {
            return computer.GetFrameOptions().FrameShiftMs / 1000.0f;
        }

        public long NumFramesReady()
        {
            return features.Size();
        }

        public bool IsLastFrame(long frame)
        {
            return inputFinished && frame == NumFramesReady() - 1;
        }

        public float[] GetFrame(long frame)
        {
            return features.At(frame);
        }

        public void AcceptWaveform(float samplingRate, float[] waveform, int n)
        {
            if (n == 0)
                return;

            if (inputFinished)
                throw new Exception("AcceptWaveform called after InputFinished() was called.");

            if (samplingRate != computer.GetFrameOptions().SampFreq)
                throw new Exception("Sampling rate mismatch.");

            EnsureRemainderCapacity(_remainderCount + n);
            Array.Copy(waveform, 0, _waveformRemainder, _remainderCount, n);
            _remainderCount += n;
            ComputeFeatures();
        }

        public void InputFinished()
        {
            inputFinished = true;
            ComputeFeatures();
        }

        public void Pop(int n)
        {
            features.Pop(n);
        }

        private void EnsureRemainderCapacity(int needed)
        {
            if (_waveformRemainder.Length >= needed) return;
            int newLen = Math.Max(needed, Math.Max(64, _waveformRemainder.Length * 2));
            var resized = new float[newLen];
            if (_remainderCount > 0)
                Array.Copy(_waveformRemainder, resized, _remainderCount);
            _waveformRemainder = resized;
        }

        private void ComputeFeatures()
        {
            var frameOpts = computer.GetFrameOptions();
            long numSamplesTotal = waveformOffset + _remainderCount;
            long numFramesOld = features.Size();
            long numFramesNew = FeatureWindow.NumFrames(numSamplesTotal, frameOpts, inputFinished);
            if (numFramesNew < numFramesOld)
                throw new Exception("Number of new frames cannot be less than old frames.");
            // Note: this online feature-extraction code does not support VTLN.
            float vtlnWarp = 1.0f;
            int paddedSize = frameOpts.PaddedWindowSize();
            if (_paddedFrame.Length != paddedSize)
                _paddedFrame = new float[paddedSize];
            bool needRawLogEnergy = computer.NeedRawLogEnergy();
            int dim = computer.Dim();
            for (long frame = numFramesOld; frame < numFramesNew; frame++)
            {
                // Clear the scratch so the FFT zero-pad region is reset between frames.
                Array.Clear(_paddedFrame, 0, _paddedFrame.Length);
                ReadOnlySpan<float> waveSpan = _waveformRemainder.AsSpan(0, _remainderCount);
                Span<float> windowSpan = _paddedFrame.AsSpan();
                float? rawLogEnergy = 0.0f;
                float? rawLogEnergyRef = null;
                if (needRawLogEnergy)
                {
                    FeatureWindow.ExtractWindow(waveformOffset, waveSpan, frame, frameOpts, windowFunction, windowSpan, ref rawLogEnergy);
                }
                else
                {
                    FeatureWindow.ExtractWindow(waveformOffset, waveSpan, frame, frameOpts, windowFunction, windowSpan, ref rawLogEnergyRef);
                }
                float[] thisFeature = new float[dim];
                computer.Compute(rawLogEnergy, vtlnWarp, windowSpan, thisFeature.AsSpan());
                features.PushBack(thisFeature);
            }
            // OK, we will now discard any portion of the signal that will not be
            // necessary to compute frames in the future.
            long firstSampleOfNextFrame = FirstSampleOfFrame(numFramesNew, frameOpts);
            int samplesToDiscard = (int)(firstSampleOfNextFrame - waveformOffset);
            if (samplesToDiscard > 0)
            {
                int newNumSamples = _remainderCount - samplesToDiscard;
                if (newNumSamples <= 0)
                {
                    waveformOffset += _remainderCount;
                    _remainderCount = 0;
                }
                else
                {
                    // In-place shift left within the same buffer.
                    Array.Copy(_waveformRemainder, samplesToDiscard, _waveformRemainder, 0, newNumSamples);
                    _remainderCount = newNumSamples;
                    waveformOffset += samplesToDiscard;
                }
            }
        }

        private long FirstSampleOfFrame(long numFramesNew, FrameExtractionOptions frameOpts)
        {
            // First-sample-of-frame is measured in audio samples, not milliseconds. The
            // legacy expression (numFramesNew * FrameShiftMs) under-counted by a factor of
            // SampFreq * 0.001 (≈ 160× at 16 kHz), so `samplesToDiscard` was almost always
            // non-positive and _waveformRemainder grew without bound — a long-run OOM /
            // O(N²) trap. WindowShift() is the correct sample-domain stride; mirrors the
            // identically-named static helper in FeatureWindow.
            return numFramesNew * frameOpts.WindowShift();
        }
    }

    public interface IFeatureComputer
    {
        FrameExtractionOptions GetFrameOptions();
        int Dim();
        bool NeedRawLogEnergy();
        void Compute(float? rawLogEnergy, float vtlnWarp, Span<float> signalFrame, Span<float> feature);
    }

    public class FbankFeature : OnlineGenericBaseFeature<FbankComputer> , IFeature
    {
        public FbankFeature(FbankComputer opts) : base(opts)
        {
        }
    }

    public class WhisperFeature : OnlineGenericBaseFeature<WhisperComputer>, IFeature
    {
        public WhisperFeature(WhisperComputer opts) : base(opts)
        {
        }
    }

    public interface IFeature
    {
        void AcceptWaveform(float sampleRate, float[] samples, int samplesSize);
        void InputFinished();
        long NumFramesReady();
        int Dim();
        float[] GetFrame(long index);
        void Pop(int n);
    }
}
