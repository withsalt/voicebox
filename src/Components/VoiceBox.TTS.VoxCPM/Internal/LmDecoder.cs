using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>lm.onnx 单步输出 (dit_hidden 供 CFM, stop_logits 供判停)。</summary>
    internal readonly record struct LmStepResult(float[] DitHidden, float[] StopLogits);

    /// <summary>
    /// 统一 prefill/decode 的 VoxCPM LM 图 (base_lm 24L + FSQ + residual_lm 6L + 双头)。
    /// prefill = 空 past (P=0) 一次喂入整个文本(+prompt)序列; decode = S=1 增长 KV。
    ///
    /// 两条路径共享同一 KV 生命周期: 上一步 Run 输出的 4 张 present OrtValue 直接作为
    /// 本步的 past 输入 (零拷贝, 不落托管数组 —— base KV 在长序列下每步可达数十 MB)。
    ///   * IO-binding (CUDA 默认): present 绑定到显存常驻, 小张量回 CPU。
    ///   * Dictionary (CPU / 回退): present 留在 ORT 原生 CPU 内存, 逐步回喂。
    /// </summary>
    internal sealed class LmDecoder : ILmDecoder
    {
        private static readonly string[] InNames =
        [
            "inputs_embeds", "feat_embed_masked", "fsq_mask", "attention_mask", "position_ids",
            "base_past_k", "base_past_v", "res_past_k", "res_past_v",
        ];
        private static readonly string[] OutNames =
        [
            "dit_hidden", "stop_logits", "lm_hidden", "residual_hidden",
            "base_present_k", "base_present_v", "res_present_k", "res_present_v",
        ];

        private readonly InferenceSession _session;
        private readonly RunOptions _run = new();
        private readonly OrtDtype _dtype;
        private readonly int _Lb, _Lr, _kv, _D, _H;
        private readonly float[] _singleFsqMask = [1f];
        private readonly long[] _decodePosition = new long[1];
        private readonly long[] _embedsShape = new long[3];
        private readonly long[] _fsqMaskShape = new long[2];
        private readonly long[] _attentionMaskShape = new long[2];
        private readonly long[] _positionShape = new long[2];

        // ---- 共享 KV 状态: 上一步 outputs (OutNames 顺序) 与 P=0 空 past ----
        private IDisposableReadOnlyCollection<OrtValue>? _prevOutputs;
        private OrtValue[]? _emptyPast;   // [bk, bv, rk, rv]
        private int _pastLen;

        // ---- IO-binding 路径 (CUDA) ----
        private readonly bool _ioBinding;
        private readonly OrtMemoryInfo? _cudaMem;
        private readonly OrtMemoryInfo? _cpuMem;
        private OrtIoBinding? _binding;

        public OrtDtype Dtype => _dtype;
        public int PastLength => _pastLen;

        public LmDecoder(string modelPath, SessionOptions so, VoxCpmModelConfig cfg, bool useIoBinding, int gpuDeviceId)
        {
            _session = new InferenceSession(modelPath, so);
            _dtype = OnnxIo.Detect(_session, "inputs_embeds");
            _Lb = cfg.BaseLmLayers;
            _Lr = cfg.ResidualLmLayers;
            _kv = cfg.NumKeyValueHeads;
            _D = cfg.HeadDim;
            _H = cfg.HiddenSize;

            _ioBinding = useIoBinding;
            if (_ioBinding)
            {
                _cudaMem = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.ArenaAllocator, gpuDeviceId, OrtMemType.Default);
                _cpuMem = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput);
                using (var _ = _session.CreateIoBinding()) { }   // 构造期探测 API 一次
            }
        }

        /// <summary>开始一次新的合成: 清空 KV, 重建 empty-past (与 CUDA 下的 binding)。</summary>
        public void Reset()
        {
            _prevOutputs?.Dispose();
            _prevOutputs = null;
            DisposeEmptyPast();
            _emptyPast =
            [
                MakeOwnedEmpty(_Lb), MakeOwnedEmpty(_Lb),
                MakeOwnedEmpty(_Lr), MakeOwnedEmpty(_Lr),
            ];
            if (_ioBinding)
            {
                _binding?.Dispose();
                _binding = _session.CreateIoBinding();
            }
            _pastLen = 0;
        }

        /// <summary>合成结束后释放整段 KV (显存/原生内存), 空闲期不占内存。下次使用前必须 Reset。</summary>
        public void EndGeneration()
        {
            _prevOutputs?.Dispose();
            _prevOutputs = null;
            DisposeEmptyPast();
            _binding?.Dispose();
            _binding = null;
            _pastLen = 0;
        }

        /// <summary>
        /// prefill: 整段序列 (S=T, P=0)。inputs/fem 为 [S*H] 行主序, fsqMask [S]。
        /// </summary>
        public LmStepResult Prefill(float[] inputsEmbeds, float[] featEmbedMasked, float[] fsqMask, int s)
            => Run(inputsEmbeds, featEmbedMasked, fsqMask, s);

        /// <summary>
        /// decode 单步: currEmbed [H] 同时充当 inputs_embeds 与 feat_embed_masked, fsq_mask=1。
        /// </summary>
        public LmStepResult Step(float[] currEmbed)
            => Run(currEmbed, currEmbed, _singleFsqMask, 1);

        private LmStepResult Run(float[] embeds, float[] fem, float[] fsqMask, int s)
            => _ioBinding ? RunIoBinding(embeds, fem, fsqMask, s) : RunDictionary(embeds, fem, fsqMask, s);

        private (long[] am, long[] pos) BuildAux(int s)
        {
            var am = new long[_pastLen + s];
            Array.Fill(am, 1L);
            long[] pos;
            if (s == 1)
            {
                _decodePosition[0] = _pastLen;
                pos = _decodePosition;
            }
            else
            {
                pos = new long[s];
                for (int i = 0; i < s; i++) pos[i] = _pastLen + i;
            }
            return (am, pos);
        }

        private void SetInputShapes(int s)
        {
            _embedsShape[0] = 1;
            _embedsShape[1] = s;
            _embedsShape[2] = _H;
            _fsqMaskShape[0] = 1;
            _fsqMaskShape[1] = s;
            _attentionMaskShape[0] = 1;
            _attentionMaskShape[1] = _pastLen + s;
            _positionShape[0] = 1;
            _positionShape[1] = s;
        }

        private OrtValue PastInput(int i) => _prevOutputs is null ? _emptyPast![i] : _prevOutputs[4 + i];

        private LmStepResult RunIoBinding(float[] embeds, float[] fem, float[] fsqMask, int s)
        {
            var (am, pos) = BuildAux(s);
            SetInputShapes(s);
            var own = new List<OrtValue>(5);
            IDisposableReadOnlyCollection<OrtValue> outputs;
            try
            {
                var b = _binding!;
                b.BindInput("inputs_embeds", OnnxIo.MakeFloat(_dtype, embeds, _embedsShape, own));
                b.BindInput("feat_embed_masked", OnnxIo.MakeFloat(_dtype, fem, _embedsShape, own));
                b.BindInput("fsq_mask", OnnxIo.MakeFloat(_dtype, fsqMask, _fsqMaskShape, own));
                b.BindInput("attention_mask", OnnxIo.MakeLong(am, _attentionMaskShape, own));
                b.BindInput("position_ids", OnnxIo.MakeLong(pos, _positionShape, own));
                b.BindInput("base_past_k", PastInput(0));
                b.BindInput("base_past_v", PastInput(1));
                b.BindInput("res_past_k", PastInput(2));
                b.BindInput("res_past_v", PastInput(3));
                // 小张量回 CPU; KV 常驻显存
                b.BindOutputToDevice("dit_hidden", _cpuMem!);
                b.BindOutputToDevice("stop_logits", _cpuMem!);
                b.BindOutputToDevice("lm_hidden", _cpuMem!);
                b.BindOutputToDevice("residual_hidden", _cpuMem!);
                b.BindOutputToDevice("base_present_k", _cudaMem!);
                b.BindOutputToDevice("base_present_v", _cudaMem!);
                b.BindOutputToDevice("res_present_k", _cudaMem!);
                b.BindOutputToDevice("res_present_v", _cudaMem!);
                outputs = _session.RunWithBoundResults(_run, b);
            }
            finally { OnnxIo.Dispose(own); }

            return Advance(outputs, s);
        }

        private LmStepResult RunDictionary(float[] embeds, float[] fem, float[] fsqMask, int s)
        {
            var (am, pos) = BuildAux(s);
            SetInputShapes(s);
            var own = new List<OrtValue>(5);
            IDisposableReadOnlyCollection<OrtValue> outputs;
            try
            {
                var inputs = new[]
                {
                    OnnxIo.MakeFloat(_dtype, embeds, _embedsShape, own),
                    OnnxIo.MakeFloat(_dtype, fem, _embedsShape, own),
                    OnnxIo.MakeFloat(_dtype, fsqMask, _fsqMaskShape, own),
                    OnnxIo.MakeLong(am, _attentionMaskShape, own),
                    OnnxIo.MakeLong(pos, _positionShape, own),
                    PastInput(0), PastInput(1), PastInput(2), PastInput(3),
                };
                outputs = _session.Run(_run, InNames, inputs, OutNames);
            }
            finally { OnnxIo.Dispose(own); }

            return Advance(outputs, s);
        }

        private LmStepResult Advance(IDisposableReadOnlyCollection<OrtValue> outputs, int s)
        {
            float[] dit = OnnxIo.ReadFloat(outputs[0], _dtype);
            float[] stop = OnnxIo.ReadFloat(outputs[1], _dtype);
            _prevOutputs?.Dispose();   // 其 present 已被本步消费为 past, 可安全释放
            _prevOutputs = outputs;
            _pastLen += s;
            return new LmStepResult(dit, stop);
        }

        private OrtValue MakeOwnedEmpty(int layers)
        {
            long[] shape = [layers, 1, _kv, 0, _D];
            return _dtype switch
            {
                OrtDtype.Float16 => OrtValue.CreateTensorValueFromMemory(Array.Empty<Float16>(), shape),
                _ => OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), shape),
            };
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
