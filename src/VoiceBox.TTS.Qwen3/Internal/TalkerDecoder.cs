using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Talker (main autoregressive transformer) over the unified talker.onnx graph.
    /// Prefill = Step with empty past (P=0); decode = Step with the growing KV cache.
    ///
    /// Two KV paths (chosen at construction):
    ///   * <b>IO-binding</b> (CUDA, default): present_k/present_v stay in VRAM and are bound directly
    ///     as the next step's past_k/past_v — the growing KV is never copied to managed memory. Only
    ///     logits + last hidden (small) come back to CPU (needed for sampling and the code-predictor).
    ///     Per-Generate binding state (Reset() rebuilds it).
    ///   * <b>Dictionary</b> (CPU / fallback): present.* copied to managed arrays and re-fed each step.
    ///
    /// (A static-KV CUDA-graph decode path was prototyped — talker_static.onnx exists and is byte-exact
    /// — but ORT C# cannot host-seed a device KV buffer / drive an in-place device cache under cuda-graph,
    /// so it is not wired here. See the memory note for details.)
    /// </summary>
    internal sealed class TalkerDecoder : IDisposable
    {
        private static readonly string[] InNames = ["inputs_embeds", "attention_mask", "position_ids", "past_k", "past_v"];
        private static readonly string[] OutNames = ["logits", "hidden", "present_k", "present_v"];

        private readonly InferenceSession _session;
        private readonly RunOptions _run = new();
        private readonly OrtDtype _dtype;
        private readonly int _L, _kv, _D, _H, _V;

        // ---- Dictionary path state ----
        private Array _pastK = Array.Empty<float>();
        private Array _pastV = Array.Empty<float>();
        private int _pastLen;

        // ---- IO-binding path state (CUDA) ----
        private readonly bool _ioBinding;
        private readonly OrtMemoryInfo? _cudaMem;     // present.* (device-resident)
        private readonly OrtMemoryInfo? _cpuMem;      // logits/hidden (CPU, read for sampling/cp)
        private OrtIoBinding? _binding;
        private IDisposableReadOnlyCollection<OrtValue>? _prevOutputs;  // [logits, hidden, present_k, present_v]
        private OrtValue[]? _emptyPast;               // [present_k, present_v] empty (P=0) for prefill

        public int HiddenSize => _H;
        public int VocabSize => _V;

        public TalkerDecoder(string modelPath, SessionOptions so, Qwen3ModelConfig cfg, bool useIoBinding, int gpuDeviceId)
        {
            _session = new InferenceSession(modelPath, so);
            // Detect from the graph I/O type; fall back to config when metadata is stale.
            _dtype = OnnxIo.Detect(_session, "inputs_embeds", cfg.Dtype);
            var t = cfg.Talker;
            _L = t.NumLayers; _kv = t.NumKvHeads; _D = t.HeadDim; _H = t.HiddenSize; _V = t.CodecVocabSize;

            _ioBinding = useIoBinding;
            if (_ioBinding)
            {
                _cudaMem = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.ArenaAllocator, gpuDeviceId, OrtMemType.Default);
                _cpuMem = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput);
                using (var _ = _session.CreateIoBinding()) { }   // probe API once at construction
            }
        }

        public void Reset()
        {
            if (_ioBinding)
            {
                _prevOutputs?.Dispose();
                _prevOutputs = null;
                _binding?.Dispose();
                _binding = _session.CreateIoBinding();
                DisposeEmptyPast();
                long[] empty = [_L, 1, _kv, 0, _D];
                _emptyPast =
                [
                    OnnxIo.MakeKvOwned(_dtype, OnnxIo.EmptyKv(_dtype), empty),
                    OnnxIo.MakeKvOwned(_dtype, OnnxIo.EmptyKv(_dtype), empty),
                ];
            }
            else
            {
                _pastK = OnnxIo.EmptyKv(_dtype);
                _pastV = OnnxIo.EmptyKv(_dtype);
            }
            _pastLen = 0;
        }

        /// <summary>
        /// Release the per-generation KV state once a synthesis finishes, so the device-resident KV
        /// cache (IO-binding path) and managed KV (dictionary path) are not held during idle. The next
        /// <see cref="Reset"/> re-creates the binding/empty-past before the next generation; callers
        /// MUST call <see cref="Reset"/> before the next <see cref="Step"/>.
        /// </summary>
        public void EndGeneration()
        {
            if (_ioBinding)
            {
                _prevOutputs?.Dispose();   // frees the full-sequence present_k/present_v in VRAM
                _prevOutputs = null;
                DisposeEmptyPast();
                _binding?.Dispose();
                _binding = null;
            }
            else
            {
                _pastK = OnnxIo.EmptyKv(_dtype);
                _pastV = OnnxIo.EmptyKv(_dtype);
            }
            _pastLen = 0;
        }

        /// <summary>
        /// Run one talker step. <paramref name="inputsEmbeds"/> is [S*H] row-major; positions are
        /// posStart..posStart+S-1 (all 3 mrope rows equal). Returns last-row logits[V] and hidden[H];
        /// the KV cache is advanced by S.
        /// </summary>
        public (float[] logits, float[] hidden) Step(float[] inputsEmbeds, int attnLen, int posStart, int s)
            => _ioBinding ? StepIoBinding(inputsEmbeds, attnLen, posStart, s)
                          : StepDictionary(inputsEmbeds, attnLen, posStart, s);

        private (long[] am, long[] pos) BuildPositions(int attnLen, int posStart, int s)
        {
            var am = new long[attnLen];
            Array.Fill(am, 1L);
            var pos = new long[3 * s];
            for (int r = 0; r < 3; r++)
                for (int i = 0; i < s; i++) pos[r * s + i] = posStart + i;
            return (am, pos);
        }

        private (float[] logits, float[] hidden) StepIoBinding(float[] inputsEmbeds, int attnLen, int posStart, int s)
        {
            var (am, pos) = BuildPositions(attnLen, posStart, s);
            var own = new List<OrtValue>(3);
            IDisposableReadOnlyCollection<OrtValue> outputs;
            try
            {
                var b = _binding!;
                b.BindInput("inputs_embeds", OnnxIo.MakeFloat(_dtype, inputsEmbeds, [1, s, _H], own));
                b.BindInput("attention_mask", OnnxIo.MakeLong(am, [1, attnLen], own));
                b.BindInput("position_ids", OnnxIo.MakeLong(pos, [3, 1, s], own));
                b.BindInput("past_k", _prevOutputs is null ? _emptyPast![0] : _prevOutputs[2]);
                b.BindInput("past_v", _prevOutputs is null ? _emptyPast![1] : _prevOutputs[3]);
                // logits/hidden -> CPU (consumed by sampler + code-predictor); present.* -> VRAM.
                b.BindOutputToDevice("logits", _cpuMem!);
                b.BindOutputToDevice("hidden", _cpuMem!);
                b.BindOutputToDevice("present_k", _cudaMem!);
                b.BindOutputToDevice("present_v", _cudaMem!);
                outputs = _session.RunWithBoundResults(_run, b);
            }
            finally { OnnxIo.Dispose(own); }   // inputs consumed by Run; present outputs are separate

            float[] logits = OnnxIo.ReadFloat(outputs[0], _dtype);   // [1,S,V] on CPU
            float[] hidden = OnnxIo.ReadFloat(outputs[1], _dtype);   // [1,S,H] on CPU
            var logitsLast = logits.AsSpan((s - 1) * _V, _V).ToArray();
            var hiddenLast = hidden.AsSpan((s - 1) * _H, _H).ToArray();

            _prevOutputs?.Dispose();   // its present.* was consumed as THIS step's past -> safe to free
            _prevOutputs = outputs;    // this step's present.* becomes next step's past
            _pastLen += s;
            return (logitsLast, hiddenLast);
        }

        private (float[] logits, float[] hidden) StepDictionary(float[] inputsEmbeds, int attnLen, int posStart, int s)
        {
            var (am, pos) = BuildPositions(attnLen, posStart, s);
            var own = new List<OrtValue>(5);
            try
            {
                var inputs = new[]
                {
                    OnnxIo.MakeFloat(_dtype, inputsEmbeds, [1, s, _H], own),
                    OnnxIo.MakeLong(am, [1, attnLen], own),
                    OnnxIo.MakeLong(pos, [3, 1, s], own),
                    OnnxIo.MakeKv(_dtype, _pastK, [_L, 1, _kv, _pastLen, _D], own),
                    OnnxIo.MakeKv(_dtype, _pastV, [_L, 1, _kv, _pastLen, _D], own),
                };

                using var outputs = _session.Run(_run, InNames, inputs, OutNames);
                float[] logits = OnnxIo.ReadFloat(outputs[0], _dtype);   // [1,S,V]
                float[] hidden = OnnxIo.ReadFloat(outputs[1], _dtype);   // [1,S,H]
                _pastK = OnnxIo.ReadKv(_dtype, outputs[2]);              // [L,1,kv,P+S,D] native dtype
                _pastV = OnnxIo.ReadKv(_dtype, outputs[3]);
                _pastLen += s;

                var logitsLast = logits.AsSpan((s - 1) * _V, _V).ToArray();
                var hiddenLast = hidden.AsSpan((s - 1) * _H, _H).ToArray();
                return (logitsLast, hiddenLast);
            }
            finally { OnnxIo.Dispose(own); }
        }

        private void DisposeEmptyPast()
        {
            if (_emptyPast is null) return;
            foreach (var v in _emptyPast) v.Dispose();
            _emptyPast = null;
        }

        public void Dispose()
        {
            _prevOutputs?.Dispose();
            DisposeEmptyPast();
            _binding?.Dispose();
            _cudaMem?.Dispose();
            _cpuMem?.Dispose();
            _run.Dispose();
            _session.Dispose();
        }
    }
}
