using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Code-predictor (sub-talker) over the FUSED code_predictor.onnx graph. The whole frame's
    /// num_code_groups-1 fine codes are produced in a SINGLE Run: the 15 autoregressive steps,
    /// the per-step input-embedding lookup, and the sampling all run inside the graph. Greedy
    /// (do_sample=0) is byte-exact vs PyTorch; sampling uses temperature + baked top_k + inverse-CDF.
    ///
    /// NOTE: CUDA-Graph was attempted here (the fused graph is fully static + 0-memcpy) but the cp's
    /// TopK/CumSum kernels allocate workspace dynamically, which ORT's CUDA-graph capture/replay does
    /// not support — capture succeeds but replay hangs at scale. So the cp uses a normal per-frame Run.
    /// </summary>
    internal sealed class CodePredictor : IDisposable
    {
        private static readonly string[] InNames = ["past_hidden", "emb0", "do_sample", "temperature", "randoms"];
        private static readonly string[] OutNames = ["fine_codes", "next_sum"];

        private readonly InferenceSession _sampleSession;
        private readonly SessionOptions _sessionOptions;
        private readonly string _greedyPath;
        private readonly bool _greedyUsesSampleSession;
        private readonly object _greedyLock = new();
        private InferenceSession? _greedySession;
        private readonly RunOptions _run = new();
        private readonly EmbeddingStore _store;
        private readonly OrtDtype _dtype;
        private readonly int _Hin, _nSteps;
        private readonly float[] _emb0Scratch;          // fp32 emb0 gather target (reused)

        public CodePredictor(string modelDir, SessionOptions so, Qwen3ModelConfig cfg, EmbeddingStore store)
        {
            string legacy = Path.Combine(modelDir, "code_predictor.onnx");
            string sampleOnly = Path.Combine(modelDir, "code_predictor_sample.onnx");
            string greedyOnly = Path.Combine(modelDir, "code_predictor_greedy.onnx");
            string sample = File.Exists(legacy) ? legacy : ExistingOrFallback(sampleOnly, legacy);
            string greedy = ExistingOrFallback(greedyOnly, legacy);

            _sampleSession = new InferenceSession(sample, so);
            _sessionOptions = so;
            _greedyPath = greedy;
            _greedyUsesSampleSession = string.Equals(sample, greedy, StringComparison.OrdinalIgnoreCase);
            _store = store;
            _dtype = OnnxIo.Detect(_sampleSession, "past_hidden", cfg.Dtype);
            _Hin = cfg.Talker.HiddenSize;
            _nSteps = cfg.NumCodeGroups - 1;            // 15 fine groups (group0 comes from the talker)
            _emb0Scratch = new float[_Hin];
        }

        private static string ExistingOrFallback(string preferred, string fallback)
            => File.Exists(preferred) ? preferred : fallback;

        /// <summary>Number of uniforms the caller must supply per frame when sampling (= G-1).</summary>
        public int RandomCount => _nSteps;

        /// <summary>
        /// Generate the frame's fine codec groups 1..G-1 in one Run. <paramref name="pastHidden"/> is
        /// the talker hidden[H] that produced <paramref name="group0"/>. When <paramref name="doSample"/>
        /// is true, <paramref name="randoms"/> (length ≥ G-1) drives sampling. Returns the 15 codes and
        /// <c>nextSum</c> = codec(group0) + Σ cp_emb[gs][code_gs] (the talker next-input embedding sum;
        /// the caller adds tts_pad).
        /// </summary>
        public (int[] fineCodes, float[] nextSum) GenerateFrame(
            float[] pastHidden, int group0, bool doSample, float temperature, float[] randoms)
        {
            _store.Codec(group0, _emb0Scratch);
            var own = new List<OrtValue>(5);
            try
            {
                var inputs = new[]
                {
                    OnnxIo.MakeFloat(_dtype, pastHidden, [1, 1, _Hin], own),
                    OnnxIo.MakeFloat(_dtype, _emb0Scratch, [1, 1, _Hin], own),
                    OnnxIo.MakeScalarLong(doSample ? 1L : 0L, own),
                    OnnxIo.MakeFloat(OrtDtype.Float32, [temperature], [], own),   // always fp32 scalar
                    OnnxIo.MakeFloat(OrtDtype.Float32, randoms, [_nSteps], own),  // always fp32 [G-1]
                };
                InferenceSession session = doSample ? _sampleSession : GetGreedySession();
                using var outputs = session.Run(_run, InNames, inputs, OutNames);

                var fc = outputs[0].GetTensorDataAsSpan<long>();   // fine_codes [G-1] int64
                var fine = new int[_nSteps];
                for (int i = 0; i < _nSteps; i++) fine[i] = (int)fc[i];
                float[] nextSum = OnnxIo.ReadFloat(outputs[1], _dtype);   // [1,1,H] native dtype
                return (fine, nextSum);
            }
            finally { OnnxIo.Dispose(own); }
        }

        private InferenceSession GetGreedySession()
        {
            if (_greedyUsesSampleSession) return _sampleSession;
            if (_greedySession is not null) return _greedySession;
            lock (_greedyLock)
                return _greedySession ??= new InferenceSession(_greedyPath, _sessionOptions);
        }

        public void Dispose()
        {
            _run.Dispose();
            _greedySession?.Dispose();
            _sampleSession.Dispose();
        }
    }
}
