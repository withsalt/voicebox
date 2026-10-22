using Microsoft.ML.OnnxRuntime;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core.Onnx;

namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class OnnxDfsmnMaskRunner : IDfsmnMaskRunner
    {
        private const int ChunkFrames = 1;

        private readonly InferenceSession _session;
        private readonly SessionOptions _sessionOptions;
        private readonly RunOptions _runOptions;
        private readonly int _chunkFrames;
        private readonly int _featureDim;
        private readonly int _maskBins;

        private readonly float[] _featureBuffer;
        private readonly float[]? _cache;
        private readonly float[]? _initialCache;

        // Long-lived OrtValue wrappers built once over the reusable buffers above. Feature/cache
        // bytes are mutated in place between calls; the wrappers see the changes immediately, so
        // there is no per-hop input marshalling and no managed copy of the (large) cache tensor.
        private readonly OrtValue _featureValue;
        private readonly OrtValue? _cacheValue;
        private readonly OrtValue[] _inputValues;
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;
        private readonly string[]? _outputNamesWithVad;

        public int FeatureDim => _featureDim;
        public int MaskBins => _maskBins;
        public bool HasVadProbabilities => _outputNamesWithVad is not null;

        public OnnxDfsmnMaskRunner(string modelPath, DfsmnAecOptions options)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"DFSMN AEC ONNX model not found: {modelPath}", modelPath);

            _chunkFrames = ChunkFrames;
            _featureDim = DfsmnAecOptions.DefaultFeatureDim;
            _maskBins = DfsmnAecOptions.DefaultMaskBins;
            _sessionOptions = OnnxSessionFactory.Create(options, SessionPurpose.StaticShapes);
            _session = new InferenceSession(modelPath, _sessionOptions);
            _runOptions = new RunOptions();

            string featureInputName = IdentifyFeatureInput(_session.InputMetadata, _featureDim);
            string? cacheInputName = IdentifyCacheTensor(_session.InputMetadata);
            string maskOutputName = IdentifyMaskOutput(_session.OutputMetadata, _maskBins);
            string? vadOutputName = TryIdentifyVadOutput(_session.OutputMetadata, maskOutputName);
            string? cacheOutputName = IdentifyCacheTensor(_session.OutputMetadata);

            // A cache input without a matching cache output cannot be streamed; fail fast at load
            // time (rather than per-hop) and guarantee `_cache is not null` ⟹ a cache_out result.
            if (cacheInputName is not null && cacheOutputName is null)
                throw new InvalidOperationException("DFSMN model has a cache input but no cache output; streaming cache cannot be maintained.");

            _featureBuffer = new float[_chunkFrames * _featureDim];
            _featureValue = OrtValue.CreateTensorValueFromMemory(
                _featureBuffer, new long[] { 1, _chunkFrames, _featureDim });

            if (cacheInputName is not null)
            {
                int[] dims = ResolvePositiveDims(_session.InputMetadata[cacheInputName].Dimensions);
                int cacheSize = dims.Aggregate(1, (a, b) => checked(a * b));
                long[] cacheShape = Array.ConvertAll(dims, static d => (long)d);
                _cache = new float[cacheSize];
                _initialCache = new float[cacheSize];
                _cacheValue = OrtValue.CreateTensorValueFromMemory(_cache, cacheShape);

                _inputNames = new[] { featureInputName, cacheInputName };
                _inputValues = new[] { _featureValue, _cacheValue };
            }
            else
            {
                _inputNames = new[] { featureInputName };
                _inputValues = new[] { _featureValue };
            }

            _outputNames = cacheOutputName is null
                ? new[] { maskOutputName }
                : new[] { maskOutputName, cacheOutputName };
            if (vadOutputName is not null)
            {
                _outputNamesWithVad = cacheOutputName is null
                    ? new[] { maskOutputName, vadOutputName }
                    : new[] { maskOutputName, vadOutputName, cacheOutputName };
            }
        }

        public void Run(ReadOnlySpan<float> features, int frames, Span<float> masks)
            => RunCore(features, frames, masks, vadProbabilities: default, requireVad: false);

        public void Run(ReadOnlySpan<float> features, int frames, Span<float> masks, Span<float> vadProbabilities)
        {
            if (vadProbabilities.Length != frames)
                throw new ArgumentException($"vadProbabilities length must be frames ({frames}).", nameof(vadProbabilities));
            if (!HasVadProbabilities)
                throw new InvalidOperationException("The DFSMN model does not expose a VAD probability output.");
            RunCore(features, frames, masks, vadProbabilities, requireVad: true);
        }

        private void RunCore(
            ReadOnlySpan<float> features,
            int frames,
            Span<float> masks,
            Span<float> vadProbabilities,
            bool requireVad)
        {
            if (frames <= 0 || frames > _chunkFrames)
                throw new ArgumentOutOfRangeException(nameof(frames), frames, $"frames must be in [1, {_chunkFrames}].");
            if (features.Length != frames * _featureDim)
                throw new ArgumentException($"features length must be frames*{_featureDim}.", nameof(features));
            if (masks.Length != frames * _maskBins)
                throw new ArgumentException($"masks length must be frames*{_maskBins}.", nameof(masks));

            features.CopyTo(_featureBuffer);

            string[] outputNames = requireVad
                ? _outputNamesWithVad ?? throw new InvalidOperationException("The DFSMN model does not expose a VAD probability output.")
                : _outputNames;
            using IDisposableReadOnlyCollection<OrtValue> results =
                _session.Run(_runOptions, _inputNames, _inputValues, outputNames);

            // results follow outputNames order: [mask, vad_prob?, cache_out?].
            ReadOnlySpan<float> maskData = results[0].GetTensorDataAsSpan<float>();
            if (maskData.Length < masks.Length)
                throw new InvalidOperationException($"DFSMN mask output too small: {maskData.Length} < {masks.Length}.");
            maskData[..masks.Length].CopyTo(masks);

            int resultIndex = 1;
            if (requireVad)
            {
                ReadOnlySpan<float> vadData = results[resultIndex++].GetTensorDataAsSpan<float>();
                if (vadData.Length < vadProbabilities.Length)
                    throw new InvalidOperationException($"DFSMN VAD output too small: {vadData.Length} < {vadProbabilities.Length}.");
                vadData[..vadProbabilities.Length].CopyTo(vadProbabilities);
            }

            if (_cache is not null)
            {
                // Feed the streaming cache back in place for the next hop.
                ReadOnlySpan<float> cacheData = results[resultIndex].GetTensorDataAsSpan<float>();
                if (cacheData.Length != _cache.Length)
                    throw new InvalidOperationException($"DFSMN cache_out size mismatch: {cacheData.Length} != {_cache.Length}.");
                cacheData.CopyTo(_cache);
            }
        }

        public void Reset()
        {
            if (_cache is not null && _initialCache is not null)
                Array.Copy(_initialCache, _cache, _cache.Length);
        }

        public void Dispose()
        {
            _featureValue.Dispose();
            _cacheValue?.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
            _sessionOptions.Dispose();
        }

        private static string IdentifyFeatureInput(IReadOnlyDictionary<string, NodeMetadata> metadata, int featureDim)
        {
            foreach (var kv in metadata)
            {
                int[] dims = kv.Value.Dimensions;
                if (dims.Length == 3 && (dims[2] == featureDim || dims[2] < 0))
                    return kv.Key;
            }
            throw new InvalidOperationException($"Could not identify feature input [1,T,{featureDim}] among inputs: {string.Join(", ", metadata.Keys)}.");
        }

        private static string? IdentifyCacheTensor(IReadOnlyDictionary<string, NodeMetadata> metadata)
        {
            foreach (var kv in metadata)
            {
                int rank = kv.Value.Dimensions.Length;
                if (rank >= 2 && rank != 3)
                    return kv.Key;
            }
            return null;
        }

        private static string IdentifyMaskOutput(IReadOnlyDictionary<string, NodeMetadata> metadata, int maskBins)
        {
            foreach (var kv in metadata)
            {
                int[] dims = kv.Value.Dimensions;
                if (dims.Length == 3 && (dims[2] == maskBins || dims[2] < 0))
                    return kv.Key;
            }
            throw new InvalidOperationException($"Could not identify mask output [1,T,{maskBins}] among outputs: {string.Join(", ", metadata.Keys)}.");
        }

        private static string? TryIdentifyVadOutput(IReadOnlyDictionary<string, NodeMetadata> metadata, string maskOutputName)
        {
            if (metadata.TryGetValue("vad_prob", out NodeMetadata? exact))
            {
                int[] dims = exact.Dimensions;
                if (dims.Length == 3 && (dims[2] == 1 || dims[2] < 0))
                    return "vad_prob";
            }

            foreach (var kv in metadata)
            {
                if (kv.Key == maskOutputName)
                    continue;
                int[] dims = kv.Value.Dimensions;
                if (dims.Length == 3 && (dims[2] == 1 || dims[2] < 0))
                    return kv.Key;
            }

            return null;
        }

        private static int[] ResolvePositiveDims(IReadOnlyList<int> dims)
        {
            var resolved = new int[dims.Count];
            for (int i = 0; i < dims.Count; i++)
            {
                int d = dims[i];
                if (d <= 0)
                    throw new InvalidOperationException("DFSMN cache tensors must have static positive dimensions.");
                resolved[i] = d;
            }
            return resolved;
        }
    }
}
