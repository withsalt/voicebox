using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Autoregressive decoder for Confucius4 ASR. Wraps the prefill + decode ONNX sessions
    /// and a persistent KV-cache buffer pool. Token selection is greedy argmax,
    /// matching the Python reference.
    /// </summary>
    /// <remarks>
    /// Hot-path discipline
    /// -------------------
    /// Per-step allocations are an enormous hidden tax at static_cache_len ~1.6 K
    /// (each layer cache is several MB; 28 × 2 of them per step). This decoder:
    ///
    ///   * Allocates KV-cache buffers ONCE (in the model's native dtype) and reuses
    ///     them across all <c>Decode</c> invocations. Prefill clears them in place.
    ///   * Builds <c>OrtValue</c> wrappers for those cache buffers ONCE (in the ctor)
    ///     and reuses them across every prefill + decode step — the underlying array
    ///     is pinned for the lifetime of the wrapper, and we mutate its contents in
    ///     place between steps via <see cref="CacheStore.CopyFromOrt"/> /
    ///     <see cref="CacheStore.ApplyDelta"/>, so ORT always sees the latest state.
    ///   * Pre-builds the small scratch inputs (input_ids buffer, cache_position
    ///     buffer, decode token buffers) and the input-dictionary so each step only
    ///     mutates element [0,0] and re-issues <c>Run</c>.
    ///   * Holds a long-lived <c>RunOptions</c> instance — its native handle is the
    ///     same regardless of call count.
    ///
    /// KV-cache output mode is selected by <c>decode_cache_mode</c>:
    ///   * <c>full</c>  — decoder returns the full updated cache each step; we
    ///     wholesale-copy back into the persistent buffer.
    ///   * <c>delta</c> — decoder returns just the [batch, heads, 1, head_dim] slice
    ///     and we scatter it into the right position of the persistent buffer.
    /// </remarks>
    internal sealed class Confucius4AsrDecoder : IDisposable
    {
        private readonly Confucius4AsrMetadata _metadata;
        private readonly Confucius4AsrSessions _sessions;
        private readonly bool _useFloat16;
        private readonly RunOptions _runOptions = new();
        private readonly HashSet<int> _eosTokenIds;

        private readonly string[] _prefillInputNames;
        private readonly string[] _prefillOutputNames;
        private readonly string[] _decodeInputNames;
        private readonly string[] _decodeOutputNames;

        private readonly bool _prefillHasCachePosition;
        private readonly int _prefillCacheStart;
        private readonly long[][] _cacheShapes;
        private readonly int _numCacheTensors;
        private readonly bool _deltaCacheMode;

        private readonly int _prefillInputIdsRank;
        private readonly int _prefillCachePositionRank;
        private readonly int _decodeInputIdsRank;
        private readonly int _decodeCachePositionRank;
        private readonly int _audioOutputDim;

        // Persistent KV-cache backing store. Allocated once, cleared in-place per prefill.
        private readonly CacheStore _cache;

        // Persistent scratch buffers for the decode step inputs. Mutated in-place each
        // step; the OrtValue wrappers below pin these and see the changes immediately.
        private readonly long[] _decodeTokenBuf = new long[1];
        private readonly long[] _decodeCachePosBuf = new long[1];
        private readonly long[] _decodeTokenShape;
        private readonly long[] _decodeCachePosShape;

        // Grow-on-demand scratch buffers for prefill's per-call input_ids (int→long
        // conversion target) and cache_position vectors. Persistent across calls so
        // we don't allocate two long[promptLen] per prefill — typical prompts are
        // 60–120 tokens, so steady-state allocation is zero after the first call.
        // The recognition gate serialises Decode invocations, so no locking needed.
        private long[] _prefillIdsScratch = Array.Empty<long>();
        private long[] _prefillPosScratch = Array.Empty<long>();

        // Long-lived OrtValue wrappers (built once in ctor, disposed in Dispose).
        // Each one pins its backing array; ORT reads from the pinned memory at Run
        // time, so we can rewrite the buffer between Run calls without rewrapping.
        private readonly OrtValue _persistentDecodeTokenValue;
        private readonly OrtValue _persistentDecodeCachePosValue;
        private readonly OrtValue? _persistentAudioPlaceholder;
        private readonly Float16[]? _audioPlaceholderFp16;
        private readonly float[]? _audioPlaceholderFp32;
        private readonly OrtValue[] _persistentCacheValues;

        // Reusable input dictionaries — prefill is partially primed in ctor (cache
        // entries only); decode is fully primed in ctor since every input there is
        // a persistent wrapper.
        private readonly Dictionary<string, OrtValue> _prefillInputs;
        private readonly Dictionary<string, OrtValue> _decodeInputs;

        // -------- GPU-resident KV cache via IOBinding (full cache mode + CUDA only) --------
        // When _useGpuCache is true, prefill + decode go through the IOBinding path:
        //   * KV cache inputs and outputs live entirely on CUDA memory, so the ~190 MB
        //     per-step H2D / D2H of the cache that the Dictionary path triggers is
        //     eliminated.
        //   * The model in full cache mode emits an output cache of the same shape
        //     as the input cache; the previous Run's cache outputs become the next
        //     Run's cache inputs (single-step swap, no manual scatter).
        //   * Init buffers hold uninitialised GPU memory — fine because the model's
        //     attention mask zeros out positions >= prompt_len during prefill (no
        //     mathematical dependency on the garbage values there).
        //   * The Dictionary path's fields above (_cache, _persistentCacheValues, …)
        //     remain allocated as a fallback in case IOBinding init throws.
        private readonly ILogger _logger;
        private readonly bool _useGpuCache;
        private readonly OrtMemoryInfo? _cudaMemInfoArena;
        private readonly OrtMemoryInfo? _cpuMemInfoCpuOutput;
        private readonly OrtAllocator? _cudaAllocator;
        private readonly OrtIoBinding? _decoderIoBinding;
        private readonly OrtValue[]? _gpuCacheInitInputs;   // throwaway prefill-input buffers
        private readonly TensorElementType _cacheElementType;

        private int _disposed;

        public IReadOnlySet<int> EosTokenIds => _eosTokenIds;

        public Confucius4AsrDecoder(Confucius4AsrMetadata metadata, Confucius4AsrSessions sessions, bool useGpuCache = false, int gpuDeviceId = 0, ILogger? logger = null)
        {
            try
            {
                _metadata = metadata;
                _sessions = sessions;
                _useFloat16 = Confucius4AsrDtypeParser.Parse(metadata.Dtype) == Confucius4AsrIoDtype.Float16;
                _cacheElementType = _useFloat16 ? TensorElementType.Float16 : TensorElementType.Float;
                _logger = logger ?? NullLogger.Instance;
                _eosTokenIds = new HashSet<int>(metadata.EosTokenIds);
                _deltaCacheMode = string.Equals(metadata.DecodeCacheMode, "delta", StringComparison.OrdinalIgnoreCase);
                _audioOutputDim = metadata.AudioOutputDim;

                _prefillInputNames = sessions.Prefill.InputNames.ToArray();
                _prefillOutputNames = sessions.Prefill.OutputNames.ToArray();
                _decodeInputNames = sessions.Decode.InputNames.ToArray();
                _decodeOutputNames = sessions.Decode.OutputNames.ToArray();

                _prefillHasCachePosition =
                    _prefillInputNames.Length >= 3 && _prefillInputNames[2] == "cache_position";
                _prefillCacheStart = _prefillHasCachePosition ? 3 : 2;

                var prefillInputs = sessions.Prefill.InputMetadata;
                var decodeInputs = sessions.Decode.InputMetadata;
                _prefillInputIdsRank = prefillInputs[_prefillInputNames[0]].Dimensions.Length;
                _decodeInputIdsRank = decodeInputs[_decodeInputNames[0]].Dimensions.Length;
                _prefillCachePositionRank = _prefillHasCachePosition
                    ? prefillInputs[_prefillInputNames[2]].Dimensions.Length
                    : 0;
                int decodeCachePositionIndex = sessions.SharedDecoder ? 2 : 1;
                _decodeCachePositionRank = decodeInputs[_decodeInputNames[decodeCachePositionIndex]].Dimensions.Length;

                int cacheCount = _prefillInputNames.Length - _prefillCacheStart;
                _cacheShapes = new long[cacheCount][];
                for (int i = 0; i < cacheCount; i++)
                {
                    var meta = prefillInputs[_prefillInputNames[_prefillCacheStart + i]];
                    long[] dims = meta.Dimensions.Select(d => (long)d).ToArray();
                    if (dims.Length == 0)
                        throw new InvalidOperationException("KV cache input has empty shape.");
                    if (dims[0] <= 0) dims[0] = 1;
                    for (int d = 1; d < dims.Length; d++)
                        if (dims[d] <= 0)
                            throw new InvalidOperationException(
                                $"KV cache input {_prefillInputNames[_prefillCacheStart + i]} has dynamic non-batch dim — re-export with static_cache_len.");
                    _cacheShapes[i] = dims;
                }
                _numCacheTensors = cacheCount;

                _cache = _useFloat16
                    ? CacheStore.AllocateFloat16(_cacheShapes)
                    : CacheStore.AllocateFloat32(_cacheShapes);

                _decodeTokenShape = _decodeInputIdsRank == 1 ? new long[] { 1L } : new long[] { 1L, 1L };
                _decodeCachePosShape = _decodeCachePositionRank == 1 ? new long[] { 1L } : new long[] { 1L, 1L };

                // ---- Build all long-lived OrtValue wrappers ----
                // These pin their backing arrays for the decoder's entire lifetime. We
                // mutate the underlying buffers in place between Run calls; ORT sees the
                // updated bytes each call without ever needing a fresh OrtValue.
                _persistentDecodeTokenValue =
                    OrtValue.CreateTensorValueFromMemory<long>(_decodeTokenBuf, _decodeTokenShape);
                _persistentDecodeCachePosValue =
                    OrtValue.CreateTensorValueFromMemory<long>(_decodeCachePosBuf, _decodeCachePosShape);

                if (_sessions.SharedDecoder)
                {
                    long[] phShape = { 1L, _audioOutputDim };
                    if (_useFloat16)
                    {
                        _audioPlaceholderFp16 = new Float16[_audioOutputDim]; // zero-init
                        _persistentAudioPlaceholder =
                            OrtValue.CreateTensorValueFromMemory<Float16>(_audioPlaceholderFp16, phShape);
                    }
                    else
                    {
                        _audioPlaceholderFp32 = new float[_audioOutputDim]; // zero-init
                        _persistentAudioPlaceholder =
                            OrtValue.CreateTensorValueFromMemory<float>(_audioPlaceholderFp32, phShape);
                    }
                }

                _persistentCacheValues = new OrtValue[_numCacheTensors];
                for (int i = 0; i < _numCacheTensors; i++)
                    _persistentCacheValues[i] = _cache.WrapAsOrt(i, _cacheShapes[i]);

                // ---- Prime the reusable input dictionaries ----
                _prefillInputs = new Dictionary<string, OrtValue>(_prefillInputNames.Length);
                for (int i = 0; i < _numCacheTensors; i++)
                    _prefillInputs[_prefillInputNames[_prefillCacheStart + i]] = _persistentCacheValues[i];

                _decodeInputs = new Dictionary<string, OrtValue>(_decodeInputNames.Length);
                _decodeInputs[_decodeInputNames[0]] = _persistentDecodeTokenValue;
                int decodeCacheStart;
                if (_sessions.SharedDecoder)
                {
                    _decodeInputs[_decodeInputNames[1]] = _persistentAudioPlaceholder!;
                    _decodeInputs[_decodeInputNames[2]] = _persistentDecodeCachePosValue;
                    decodeCacheStart = 3;
                }
                else
                {
                    _decodeInputs[_decodeInputNames[1]] = _persistentDecodeCachePosValue;
                    decodeCacheStart = 2;
                }
                for (int i = 0; i < _numCacheTensors; i++)
                    _decodeInputs[_decodeInputNames[decodeCacheStart + i]] = _persistentCacheValues[i];

                // -------- Optional IOBinding GPU cache path --------
                // Only attempted when caller opted in AND the model uses full cache mode
                // (output cache shape == input cache shape) AND the decoder is the shared
                // prefill+decode session (so we have a single IOBinding to manage). If any
                // of the binding API calls fail (driver mismatch, ORT version skew, etc.)
                // we log a warning and fall through to the Dictionary path — the user
                // never sees a hard failure from this best-effort fallback.
                bool tryGpu = useGpuCache && !_deltaCacheMode && _sessions.SharedDecoder;
                if (tryGpu)
                {
                    try
                    {
                        _cudaMemInfoArena = new OrtMemoryInfo(
                            OrtMemoryInfo.allocatorCUDA,
                            OrtAllocatorType.ArenaAllocator,
                            gpuDeviceId,
                            OrtMemType.Default);
                        _cpuMemInfoCpuOutput = new OrtMemoryInfo(
                            OrtMemoryInfo.allocatorCPU,
                            OrtAllocatorType.DeviceAllocator,
                            0,
                            OrtMemType.CpuOutput);
                        // Prefer the env-registered shared allocator (registered in
                        // Confucius4 runtime allocator initialisation) — falling back to
                        // a per-session CUDA allocator if none was registered.
                        OrtAllocator? envAllocator = null;
                        try { envAllocator = OrtEnv.Instance().GetSharedAllocator(_cudaMemInfoArena); }
                        catch { /* not available; fall through */ }
                        _cudaAllocator = envAllocator ?? new OrtAllocator(_sessions.Decode, _cudaMemInfoArena);

                        _gpuCacheInitInputs = new OrtValue[_numCacheTensors];
                        for (int i = 0; i < _numCacheTensors; i++)
                        {
                            _gpuCacheInitInputs[i] = OrtValue.CreateAllocatedTensorValue(
                                _cudaAllocator, _cacheElementType, _cacheShapes[i]);
                        }

                        _decoderIoBinding = _sessions.Decode.CreateIoBinding();
                        _useGpuCache = true;
                        _logger.LogInformation(
                            "Confucius4 decoder: GPU-resident KV cache via IOBinding enabled (device={DeviceId}, layers={Layers}, allocator={Allocator}).",
                            gpuDeviceId,
                            _numCacheTensors,
                            envAllocator is null ? "session-local" : "env-shared");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex,
                            "Confucius4 decoder: failed to initialise GPU-resident KV cache; falling back to Dictionary path.");
                        DisposeGpuResources();
                        _useGpuCache = false;
                    }
                }
                else if (useGpuCache)
                {
                    // Caller asked for GPU cache but the model can't use it — explain why.
                    _logger.LogInformation(
                        "Confucius4 decoder: GPU-resident KV cache requested but not applicable (cache_mode={Mode}, shared_decoder={Shared}). Using Dictionary path.",
                        _deltaCacheMode ? "delta" : "full",
                        _sessions.SharedDecoder);
                }
            }
            catch
            {
                try { Dispose(); } catch { }
                throw;
            }
        }

        /// <summary>
        /// Run greedy decoding over the given prompt + audio features. Returns the
        /// generated token id list (excluding EOS) and inference wall-clock time.
        /// </summary>
        public DecodeResult Decode(
            int[] inputIds,
            AudioFeatures audioFeatures,
            int maxNewTokens,
            CancellationToken ct)
        {
            EnsureNotDisposed();
            int promptLen = inputIds.Length;
            int staticCacheLen = _metadata.StaticCacheLen;
            if (promptLen + maxNewTokens > staticCacheLen)
                throw new InvalidOperationException(
                    $"Prompt + max_new_tokens ({promptLen}+{maxNewTokens}) exceeds static_cache_len ({staticCacheLen}).");

            if (_useGpuCache)
                return DecodeWithIoBinding(inputIds, audioFeatures, maxNewTokens, ct);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Reuse the persistent cache; zero it before prefill so cross-session
            // state never leaks.
            _cache.Clear();

            int firstToken = RunPrefill(inputIds, audioFeatures);
            double prefillSeconds = stopwatch.Elapsed.TotalSeconds;
            if (_eosTokenIds.Contains(firstToken))
                return new DecodeResult(Array.Empty<int>(), prefillSeconds, 0.0, prefillSeconds);

            var generated = new List<int>(maxNewTokens) { firstToken };
            int nextToken = firstToken;
            int cachePos = promptLen;

            for (int step = 1; step < maxNewTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                int produced = RunDecodeStep(nextToken, cachePos);
                if (_eosTokenIds.Contains(produced)) break;
                generated.Add(produced);
                nextToken = produced;
                cachePos += 1;
            }

            double totalSeconds = stopwatch.Elapsed.TotalSeconds;
            double decodeStepSeconds = Math.Max(0.0, totalSeconds - prefillSeconds);
            return new DecodeResult(generated.ToArray(), prefillSeconds, decodeStepSeconds, totalSeconds);
        }

        /// <summary>
        /// GPU-resident KV-cache decode path. Uses an <see cref="OrtIoBinding"/> bound
        /// to the shared prefill+decode session with cache inputs/outputs living on
        /// CUDA memory. Each Run's cache outputs become the next Run's cache inputs;
        /// the previous Run's collection is disposed after the next Run completes so
        /// the BFCArena recycles the memory.
        /// </summary>
        /// <remarks>
        /// Invariant: at most two cache-output collections are alive at any time —
        /// <c>prevOutputs</c> (last Run's outputs, currently bound as next Run's
        /// inputs) and the freshly-returned <c>curOutputs</c>. <c>prevOutputs</c> is
        /// disposed only after the Run that consumed it has returned, so its GPU
        /// memory stays valid for the kernel reads.
        /// </remarks>
        private DecodeResult DecodeWithIoBinding(
            int[] inputIds,
            AudioFeatures audioFeatures,
            int maxNewTokens,
            CancellationToken ct)
        {
            int promptLen = inputIds.Length;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Reuse the same grow-on-demand host buffers as the dictionary path. GPU
            // I/O binding still needs short-lived OrtValue wrappers, but no longer
            // allocates two long[promptLen] arrays for every streaming update.
            if (_prefillIdsScratch.Length < promptLen)
                _prefillIdsScratch = new long[Math.Max(promptLen, _prefillIdsScratch.Length * 2)];
            if (_prefillPosScratch.Length < promptLen)
                _prefillPosScratch = new long[Math.Max(promptLen, _prefillPosScratch.Length * 2)];
            for (int i = 0; i < promptLen; i++)
            {
                _prefillIdsScratch[i] = inputIds[i];
                _prefillPosScratch[i] = i;
            }

            long[] idsShape = _prefillInputIdsRank == 1
                ? new long[] { promptLen }
                : new long[] { 1L, promptLen };
            long[] posShape = _prefillCachePositionRank == 1
                ? new long[] { promptLen }
                : new long[] { 1L, promptLen };

            IDisposableReadOnlyCollection<OrtValue>? prevOutputs = null;
            OrtValue? prefillIdsValue = null;
            OrtValue? prefillPosValue = null;
            OrtValue? audioValue = null;
            int firstToken;
            double prefillSeconds;

            try
            {
                prefillIdsValue = OrtValue.CreateTensorValueFromMemory<long>(
                    OrtMemoryInfo.DefaultInstance, _prefillIdsScratch.AsMemory(0, promptLen), idsShape);
                prefillPosValue = OrtValue.CreateTensorValueFromMemory<long>(
                    OrtMemoryInfo.DefaultInstance, _prefillPosScratch.AsMemory(0, promptLen), posShape);
                audioValue = audioFeatures.CreateOrtValue();

                BindPrefillInputs(prefillIdsValue, audioValue, prefillPosValue);
                BindAllOutputsToDevice();

                // Capture the collection directly into prevOutputs (no `using` here:
                // the outer finally is the single owner). The collection's OrtValues
                // must stay alive past the prefill Run to be bound as inputs of the
                // first decode step.
                prevOutputs = _sessions.Decode.RunWithBoundResults(_runOptions, _decoderIoBinding!);
                firstToken = ArgmaxLastToken(prevOutputs[0]);
                prefillSeconds = stopwatch.Elapsed.TotalSeconds;
                if (_eosTokenIds.Contains(firstToken))
                    return new DecodeResult(Array.Empty<int>(), prefillSeconds, 0.0, prefillSeconds);

                var generated = new List<int>(maxNewTokens) { firstToken };
                int nextToken = firstToken;
                int cachePos = promptLen;

                for (int step = 1; step < maxNewTokens; step++)
                {
                    ct.ThrowIfCancellationRequested();

                    _decodeTokenBuf[0] = nextToken;
                    _decodeCachePosBuf[0] = cachePos;

                    BindDecodeStepInputs(prevOutputs);
                    BindAllOutputsToDevice();

                    var stepResults = _sessions.Decode.RunWithBoundResults(_runOptions, _decoderIoBinding!);
                    int produced;
                    try
                    {
                        produced = ArgmaxLastToken(stepResults[0]);
                    }
                    catch
                    {
                        stepResults.Dispose();
                        throw;
                    }

                    // It is safe to dispose prevOutputs only AFTER the Run that read
                    // from them has returned — which has happened by now. Once they
                    // are disposed, their GPU memory returns to the arena and may be
                    // reused for `stepResults`'s tensors on a subsequent step.
                    prevOutputs.Dispose();
                    prevOutputs = stepResults;

                    if (_eosTokenIds.Contains(produced)) break;
                    generated.Add(produced);
                    nextToken = produced;
                    cachePos += 1;
                }

                double totalSeconds = stopwatch.Elapsed.TotalSeconds;
                double decodeStepSeconds = Math.Max(0.0, totalSeconds - prefillSeconds);
                return new DecodeResult(generated.ToArray(), prefillSeconds, decodeStepSeconds, totalSeconds);
            }
            finally
            {
                // Always release the most recent outputs collection; once we leave
                // this scope no further Runs depend on its tensors.
                prevOutputs?.Dispose();
                prefillIdsValue?.Dispose();
                prefillPosValue?.Dispose();
                audioValue?.Dispose();
            }
        }

        private void BindPrefillInputs(OrtValue idsValue, OrtValue audioValue, OrtValue posValue)
        {
            var b = _decoderIoBinding!;
            b.BindInput(_prefillInputNames[0], idsValue);
            // Shared-decoder layout (full cache mode): [input_ids, audio_features, cache_position, …cache…]
            b.BindInput(_prefillInputNames[1], audioValue);
            if (_prefillHasCachePosition)
                b.BindInput(_prefillInputNames[2], posValue);
            for (int i = 0; i < _numCacheTensors; i++)
                b.BindInput(_prefillInputNames[_prefillCacheStart + i], _gpuCacheInitInputs![i]);
        }

        private void BindDecodeStepInputs(IDisposableReadOnlyCollection<OrtValue> prevOutputs)
        {
            var b = _decoderIoBinding!;
            b.BindInput(_decodeInputNames[0], _persistentDecodeTokenValue);
            int decodeCacheStart;
            if (_sessions.SharedDecoder)
            {
                b.BindInput(_decodeInputNames[1], _persistentAudioPlaceholder!);
                b.BindInput(_decodeInputNames[2], _persistentDecodeCachePosValue);
                decodeCacheStart = 3;
            }
            else
            {
                // Non-shared decoder is excluded from the GPU cache path in the ctor
                // (we only opt in when SharedDecoder==true). This branch exists for
                // structural symmetry only and is unreachable at runtime.
                b.BindInput(_decodeInputNames[1], _persistentDecodeCachePosValue);
                decodeCacheStart = 2;
            }
            // prevOutputs[0] = logits, prevOutputs[1..N] = cache outputs.
            for (int i = 0; i < _numCacheTensors; i++)
                b.BindInput(_decodeInputNames[decodeCacheStart + i], prevOutputs[1 + i]);
        }

        private void BindAllOutputsToDevice()
        {
            var b = _decoderIoBinding!;
            // Logits → CPU so ArgmaxLastToken's GetTensorDataAsSpan works without an
            // explicit readback step (ORT does the D2H as part of Run when the bound
            // output device is CPU).
            b.BindOutputToDevice(_decodeOutputNames[0], _cpuMemInfoCpuOutput!);
            // Cache outputs → CUDA so they stay GPU-resident and can be fed back as
            // inputs on the next step with zero copies.
            for (int i = 0; i < _numCacheTensors; i++)
                b.BindOutputToDevice(_decodeOutputNames[1 + i], _cudaMemInfoArena!);
        }

        private int RunPrefill(int[] inputIds, AudioFeatures audioFeatures)
        {
            int promptLen = inputIds.Length;
            long[] inputIdsShape = _prefillInputIdsRank == 1
                ? new long[] { promptLen }
                : new long[] { 1L, promptLen };

            // Grow-on-demand: rare event (only when this prefill's prompt exceeds the
            // largest prior prompt). Capacity grows ×2 to amortise across similar sizes.
            if (_prefillIdsScratch.Length < promptLen)
                _prefillIdsScratch = new long[Math.Max(promptLen, _prefillIdsScratch.Length * 2)];
            for (int i = 0; i < promptLen; i++) _prefillIdsScratch[i] = inputIds[i];

            // Memory<long> overload lets us pass a slice of the (possibly oversized)
            // scratch buffer with the exact shape ORT expects. The OrtValue pins the
            // slice region for its lifetime; disposal at the using-scope's end
            // releases the pin and we can mutate the scratch buffer again next call.
            using OrtValue idsValue = OrtValue.CreateTensorValueFromMemory<long>(
                OrtMemoryInfo.DefaultInstance,
                _prefillIdsScratch.AsMemory(0, promptLen),
                inputIdsShape);

            using OrtValue audioValue = audioFeatures.CreateOrtValue();

            OrtValue? positionValue = null;
            try
            {
                if (_prefillHasCachePosition)
                {
                    if (_prefillPosScratch.Length < promptLen)
                        _prefillPosScratch = new long[Math.Max(promptLen, _prefillPosScratch.Length * 2)];
                    for (int i = 0; i < promptLen; i++) _prefillPosScratch[i] = i;
                    long[] posShape = _prefillCachePositionRank == 1
                        ? new long[] { promptLen }
                        : new long[] { 1L, promptLen };
                    positionValue = OrtValue.CreateTensorValueFromMemory<long>(
                        OrtMemoryInfo.DefaultInstance,
                        _prefillPosScratch.AsMemory(0, promptLen),
                        posShape);
                }

                // Per-call inputs are slotted into the long-lived dictionary; the cache
                // entries were pre-populated in the ctor and still point to our
                // (just-cleared) persistent buffers.
                _prefillInputs[_prefillInputNames[0]] = idsValue;
                _prefillInputs[_prefillInputNames[1]] = audioValue;
                if (_prefillHasCachePosition) _prefillInputs[_prefillInputNames[2]] = positionValue!;

                using var outputs = _sessions.Prefill.Run(_runOptions, _prefillInputs, _prefillOutputNames);
                int firstToken = ArgmaxLastToken(outputs[0]);
                for (int i = 0; i < _numCacheTensors; i++)
                    _cache.CopyFromOrt(i, outputs[1 + i]);
                return firstToken;
            }
            finally
            {
                positionValue?.Dispose();
            }
        }

        private int RunDecodeStep(int tokenId, int cachePos)
        {
            // Mutate the small input buffers in place. The persistent OrtValue
            // wrappers pin these arrays, so ORT sees the new values on the next Run.
            _decodeTokenBuf[0] = tokenId;
            _decodeCachePosBuf[0] = cachePos;

            using var outputs = _sessions.Decode.Run(_runOptions, _decodeInputs, _decodeOutputNames);
            int next = ArgmaxLastToken(outputs[0]);

            if (_deltaCacheMode)
            {
                for (int i = 0; i < _numCacheTensors; i++)
                    _cache.ApplyDelta(i, outputs[1 + i], _cacheShapes[i], cachePos);
            }
            else
            {
                for (int i = 0; i < _numCacheTensors; i++)
                    _cache.CopyFromOrt(i, outputs[1 + i]);
            }
            return next;
        }

        private int ArgmaxLastToken(OrtValue logits)
        {
            var shape = logits.GetTensorTypeAndShape().Shape;
            int vocab = (int)shape[^1];
            if (_useFloat16)
            {
                ReadOnlySpan<Float16> data = logits.GetTensorDataAsSpan<Float16>();
                int offset = data.Length - vocab;
                int best = 0;
                ushort bestBits = data[offset].value;
                for (int i = 1; i < vocab; i++)
                {
                    ushort bits = data[offset + i].value;
                    if (CompareFp16(bits, bestBits) > 0)
                    {
                        bestBits = bits;
                        best = i;
                    }
                }
                return best;
            }
            else
            {
                ReadOnlySpan<float> data = logits.GetTensorDataAsSpan<float>();
                int offset = data.Length - vocab;
                return TensorPrimitives.IndexOfMax<float>(data.Slice(offset, vocab));
            }
        }

        /// <summary>
        /// Order-preserving comparison of two IEEE-754 binary16 (Float16) raw bit
        /// patterns. Faster than decoding to float for the argmax hot loop.
        /// Returns &gt;0 if a&gt;b, &lt;0 if a&lt;b, 0 if equal.
        /// </summary>
        private static int CompareFp16(ushort a, ushort b)
        {
            // Flip sign bit so the unsigned ordering matches numeric ordering for
            // positives; for negatives flip the whole value so larger magnitude
            // sorts lower.
            int aKey = (a & 0x8000) != 0 ? ~a & 0xFFFF : a | 0x8000;
            int bKey = (b & 0x8000) != 0 ? ~b & 0xFFFF : b | 0x8000;
            return aKey - bKey;
        }

        private void EnsureNotDisposed()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            DisposeGpuResources();
            if (_persistentCacheValues is not null)
            {
                for (int i = 0; i < _persistentCacheValues.Length; i++)
                {
                    try { _persistentCacheValues[i]?.Dispose(); } catch { }
                }
            }
            try { _persistentAudioPlaceholder?.Dispose(); } catch { }
            try { _persistentDecodeCachePosValue?.Dispose(); } catch { }
            try { _persistentDecodeTokenValue?.Dispose(); } catch { }
            _runOptions.Dispose();
        }

        private void DisposeGpuResources()
        {
            // Dispose order: IOBinding first (it holds internal references to bound
            // OrtValues), then the OrtValues themselves, then the allocator and
            // memory-info objects they were allocated against.
            try { _decoderIoBinding?.Dispose(); } catch { }
            if (_gpuCacheInitInputs is not null)
            {
                for (int i = 0; i < _gpuCacheInitInputs.Length; i++)
                {
                    try { _gpuCacheInitInputs[i]?.Dispose(); } catch { }
                }
            }
            // _cudaAllocator may be a session-owned handle (we created it via
            // `new OrtAllocator(session, memInfo)`) or an env-shared handle
            // (GetSharedAllocator). Both are SafeHandles and the dispose path
            // selects the right native release, so unconditional dispose is safe.
            try { _cudaAllocator?.Dispose(); } catch { }
            try { _cudaMemInfoArena?.Dispose(); } catch { }
            try { _cpuMemInfoCpuOutput?.Dispose(); } catch { }
        }
    }

    /// <summary>
    /// Native-dtype KV-cache backing store. Holds one buffer per layer tensor in
    /// either Float16 or float, exposes wrap-as-OrtValue for inputs and a
    /// copy-back-from-OrtValue for outputs — never converts dtype on the hot path.
    /// Buffers are reused across <c>Decode</c> invocations; <see cref="Clear"/>
    /// zeros them before each prefill.
    /// </summary>
    internal sealed class CacheStore
    {
        private readonly Float16[][]? _fp16;
        private readonly float[][]? _fp32;

        private CacheStore(Float16[][]? fp16, float[][]? fp32)
        {
            _fp16 = fp16;
            _fp32 = fp32;
        }

        public static CacheStore AllocateFloat16(long[][] shapes)
        {
            var arr = new Float16[shapes.Length][];
            for (int i = 0; i < shapes.Length; i++)
            {
                long n = 1;
                foreach (var d in shapes[i]) n *= d;
                arr[i] = new Float16[n];
            }
            return new CacheStore(arr, null);
        }

        public static CacheStore AllocateFloat32(long[][] shapes)
        {
            var arr = new float[shapes.Length][];
            for (int i = 0; i < shapes.Length; i++)
            {
                long n = 1;
                foreach (var d in shapes[i]) n *= d;
                arr[i] = new float[n];
            }
            return new CacheStore(null, arr);
        }

        public void Clear()
        {
            if (_fp16 is not null)
            {
                for (int i = 0; i < _fp16.Length; i++) Array.Clear(_fp16[i]);
            }
            else if (_fp32 is not null)
            {
                for (int i = 0; i < _fp32.Length; i++) Array.Clear(_fp32[i]);
            }
        }

        public OrtValue WrapAsOrt(int index, long[] shape)
        {
            return _fp16 is not null
                ? OrtValue.CreateTensorValueFromMemory<Float16>(_fp16[index], shape)
                : OrtValue.CreateTensorValueFromMemory<float>(_fp32![index], shape);
        }

        public void CopyFromOrt(int index, OrtValue value)
        {
            if (_fp16 is not null)
            {
                var src = value.GetTensorDataAsSpan<Float16>();
                var dst = _fp16[index];
                src.Slice(0, Math.Min(src.Length, dst.Length)).CopyTo(dst);
            }
            else
            {
                var src = value.GetTensorDataAsSpan<float>();
                var dst = _fp32![index];
                src.Slice(0, Math.Min(src.Length, dst.Length)).CopyTo(dst);
            }
        }

        public void ApplyDelta(int index, OrtValue delta, long[] shape, int cachePos)
        {
            if (shape.Length != 4)
                throw new InvalidOperationException("Delta cache mode requires 4-D cache tensors.");
            int heads = (int)shape[1];
            int cacheLen = (int)shape[2];
            int headDim = (int)shape[3];
            int planeStride = cacheLen * headDim;

            if (_fp16 is not null)
            {
                ReadOnlySpan<Float16> src = delta.GetTensorDataAsSpan<Float16>();
                Span<Float16> dst = _fp16[index];
                for (int h = 0; h < heads; h++)
                {
                    int dstOffset = h * planeStride + cachePos * headDim;
                    int srcOffset = h * headDim;
                    src.Slice(srcOffset, headDim).CopyTo(dst.Slice(dstOffset, headDim));
                }
            }
            else
            {
                ReadOnlySpan<float> src = delta.GetTensorDataAsSpan<float>();
                Span<float> dst = _fp32![index];
                for (int h = 0; h < heads; h++)
                {
                    int dstOffset = h * planeStride + cachePos * headDim;
                    int srcOffset = h * headDim;
                    src.Slice(srcOffset, headDim).CopyTo(dst.Slice(dstOffset, headDim));
                }
            }
        }
    }

    internal readonly struct DecodeResult
    {
        public int[] Tokens { get; }

        /// <summary>Time spent in the prefill step (== time-to-first-token).</summary>
        public double PrefillSeconds { get; }

        /// <summary>Time spent in the autoregressive decode loop after the first token.</summary>
        public double DecodeStepSeconds { get; }

        /// <summary>Total wall-clock seconds for the full decode (prefill + decode loop).</summary>
        public double InferenceSeconds { get; }

        public DecodeResult(int[] tokens, double prefillSeconds, double decodeStepSeconds, double inferenceSeconds)
        {
            Tokens = tokens;
            PrefillSeconds = prefillSeconds;
            DecodeStepSeconds = decodeStepSeconds;
            InferenceSeconds = inferenceSeconds;
        }
    }
}
