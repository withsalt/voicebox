using VoiceBox.ASR.FireRedASR2.Internal.Model;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Per-session audio + decoder state holder. Audio samples are accumulated as feature
    /// segments (one per <see cref="AddSamples"/> call) and coalesced into a single
    /// contiguous buffer only when the recognizer asks for it. Mutations are synchronised on
    /// an internal lock so <see cref="AddSamples"/> from the pump can run concurrently with
    /// the recognition pass.
    /// </summary>
    internal sealed class OfflineStream : IDisposable
    {
        private readonly object _lock = new();
        private readonly WavFrontend _wavFrontend;
        private readonly AsrModelMetadata _metadata;
        private readonly List<int[]> _timestamps;
        private readonly List<long> _tokens;
        private readonly List<float[]> _states;
        // Feature segments: each AddSamples call appends one element. O(1) append, no O(N²)
        // re-allocation/copy of the entire accumulated buffer on every push.
        private readonly List<float[]> _featureSegments = new();
        private int _totalFeatureLength;
        private float[]? _mergedSpeech;
        private bool _disposed;

        public OfflineStream(string mvnFilePath, AsrModelMetadata metadata)
        {
            ArgumentNullException.ThrowIfNull(metadata);

            _metadata = metadata;
            _wavFrontend = new WavFrontend(mvnFilePath, new FrontendConfEntity
            {
                SampleRate = metadata.SampleRate,
                NumMels = metadata.FeatureDim,
            });

            _tokens = new List<long> { metadata.SosId };
            // SOS owns slot 0 of both lists so callers can iterate them in lock-step. The
            // legacy code seeded only _tokens with SOS, which left _timestamps one slot short
            // and made FireRedTokenAssembler pair every emitted token with the next token's
            // timestamp (off-by-one).
            _timestamps = new List<int[]> { new int[2] };
            _states = CreateInitialDecoderCaches(metadata.DecoderCacheLayers);
        }

        internal List<long> Tokens => _tokens;
        internal List<int[]> Timestamps => _timestamps;
        internal IReadOnlyList<float[]> InitialDecoderCaches => _states;
        internal AsrModelMetadata Metadata => _metadata;

        /// <summary>
        /// Compute mel-CMVN features for the given samples and append to this session's
        /// feature buffer.
        /// </summary>
        public void AddSamples(ReadOnlySpan<float> samples)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (samples.IsEmpty) return;

            // Heavy compute (FBank + CMVN) is performed outside the lock — only the append
            // and cache invalidation are synchronised. Pass samples straight through;
            // OnlineFbank keeps a per-stream waveform remainder, so any tail samples that
            // can't be windowed yet roll into the next call. (Previously a 40-zero
            // trailing pad was appended here — since snip_edges=true requires a full
            // frame_length of real samples, the zeros never produced a new frame and
            // instead prefixed the next segment, polluting ~3 frames worth of windows
            // at every chunk boundary.)
            float[] features = _wavFrontend.GetFbank(samples);
            features = _wavFrontend.ApplyCmvn(features);

            lock (_lock)
            {
                _featureSegments.Add(features);
                _totalFeatureLength += features.Length;
                _mergedSpeech = null;
            }
        }

        /// <summary>
        /// Returns the current feature buffer or null when empty. The returned reference is
        /// the (cached) merged buffer — callers must not mutate.
        /// </summary>
        internal float[]? GetDecodeChunk()
        {
            lock (_lock)
            {
                if (_totalFeatureLength == 0) return null;
                if (_mergedSpeech is not null && _mergedSpeech.Length == _totalFeatureLength)
                    return _mergedSpeech;

                var merged = new float[_totalFeatureLength];
                int offset = 0;
                for (int i = 0; i < _featureSegments.Count; i++)
                {
                    float[] seg = _featureSegments[i];
                    Array.Copy(seg, 0, merged, offset, seg.Length);
                    offset += seg.Length;
                }
                _mergedSpeech = merged;
                return merged;
            }
        }

        /// <summary>
        /// Clears the buffer after a recognition pass. Guard is "more than just the SOS
        /// token": _tokens is seeded with SOS at index 0, so length > 1 means at least one
        /// real token was decoded and the input features have been consumed. The legacy
        /// "> 2" check was off by one and let _featureSegments accumulate indefinitely when
        /// the recognizer produced only a single token per finish (short utterances, silence),
        /// which is the main long-running OOM vector.
        /// </summary>
        internal void RemoveDecodedChunk()
        {
            lock (_lock)
            {
                if (_tokens.Count <= 1) return;
                _featureSegments.Clear();
                _totalFeatureLength = 0;
                _mergedSpeech = null;
            }
        }

        private static List<float[]> CreateInitialDecoderCaches(int layers)
        {
            var list = new List<float[]>(layers);
            for (int i = 0; i < layers; i++) list.Add(Array.Empty<float>());
            return list;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _wavFrontend.Dispose();
        }
    }
}
