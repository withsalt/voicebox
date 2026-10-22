using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>统一的 LM 解码器抽象: 标准图 (lm.onnx) 与 GQA 图 (lm_gqa.onnx) 共用。</summary>
    internal interface ILmDecoder : IDisposable
    {
        /// <summary>开始一次新合成: 清空 KV 状态。</summary>
        void Reset();

        /// <summary>合成结束: 释放整段 KV, 空闲期不占内存。下次使用前必须 Reset。</summary>
        void EndGeneration();

        /// <summary>prefill: 整段序列 (S=T, P=0)。inputs/fem 为 [S*H] 行主序, fsqMask [S]。</summary>
        LmStepResult Prefill(float[] inputsEmbeds, float[] featEmbedMasked, float[] fsqMask, int s);

        /// <summary>decode 单步: currEmbed [H] 同时充当 inputs_embeds 与 feat_embed_masked。</summary>
        LmStepResult Step(float[] currEmbed);
    }

    /// <summary>
    /// lm_gqa.onnx 解码器: 注意力为 com.microsoft::GroupQueryAttention 融合 kernel
    /// (rope + KV 追加 + 因果 attention 一个节点), KV 逐层输入/输出 —— 消除标准图中
    /// 堆叠 KV 每步的 Split/Concat 全量拷贝与注意力 Transpose 物化 (CPU 上
    /// P=100 时省 ~4ms/步, P=550 时省 ~5.5ms/步, 见 WORKLOG)。
    ///
    /// 与 lm.onnx 同权重 (initializer 去重共享 lm.onnx.data), 数学等价但非比特一致
    /// (融合 kernel 归约顺序不同) —— CPU EP 且包内存在 lm_gqa.onnx 时默认使用;
    /// CUDA 的 GQA kernel 不支持 fp32, 保持标准图。
    ///
    /// KV 生命周期与 LmDecoder 相同: 上一步的 present OrtValue 直接回喂为 past (零拷贝)。
    /// </summary>
    internal sealed class LmGqaDecoder : ILmDecoder
    {
        private readonly InferenceSession _session;
        private readonly RunOptions _run = new();
        private readonly OrtDtype _dtype;
        private readonly string[] _inNames;    // 5 常规输入 + 逐层 past
        private readonly string[] _outNames;   // 4 头输出 + 逐层 present
        private readonly int _H, _kv, _D, _nKv;
        private readonly float[] _singleFsqMask = [1f];
        private readonly int[] _seqlensKInput = new int[1];
        private readonly int[] _totalSequenceLengthInput = new int[1];
        private readonly long[] _embedsShape = new long[3];
        private readonly long[] _fsqMaskShape = new long[2];
        private static readonly long[] Vector1Shape = [1];
        private static readonly long[] ScalarShape = [];

        private IDisposableReadOnlyCollection<OrtValue>? _prevOutputs;
        private OrtValue? _emptyPast;          // [1,kv,0,D], 所有层共享同一只读实例
        private int _pastLen;

        public LmGqaDecoder(string modelPath, SessionOptions so, VoxCpmModelConfig cfg)
        {
            _session = new InferenceSession(modelPath, so);
            _dtype = OnnxIo.Detect(_session, "inputs_embeds");
            _H = cfg.HiddenSize;
            _kv = cfg.NumKeyValueHeads;
            _D = cfg.HeadDim;
            _nKv = (cfg.BaseLmLayers + cfg.ResidualLmLayers) * 2;

            var past = new List<string>(_nKv);
            past.AddRange(Enumerable.Range(0, cfg.BaseLmLayers).Select(i => $"base_past_k_{i}"));
            past.AddRange(Enumerable.Range(0, cfg.BaseLmLayers).Select(i => $"base_past_v_{i}"));
            past.AddRange(Enumerable.Range(0, cfg.ResidualLmLayers).Select(i => $"res_past_k_{i}"));
            past.AddRange(Enumerable.Range(0, cfg.ResidualLmLayers).Select(i => $"res_past_v_{i}"));
            _inNames = ["inputs_embeds", "feat_embed_masked", "fsq_mask", "seqlens_k", "total_sequence_length", .. past];
            _outNames = ["dit_hidden", "stop_logits", "lm_hidden", "residual_hidden",
                         .. past.Select(n => n.Replace("past", "present"))];
        }

        public void Reset()
        {
            _prevOutputs?.Dispose();
            _prevOutputs = null;
            _emptyPast?.Dispose();
            _emptyPast = _dtype == OrtDtype.Float16
                ? OrtValue.CreateTensorValueFromMemory(Array.Empty<Float16>(), [1, _kv, 0, _D])
                : OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), [1, _kv, 0, _D]);
            _pastLen = 0;
        }

        public void EndGeneration()
        {
            _prevOutputs?.Dispose();
            _prevOutputs = null;
            _emptyPast?.Dispose();
            _emptyPast = null;
            _pastLen = 0;
        }

        public LmStepResult Prefill(float[] inputsEmbeds, float[] featEmbedMasked, float[] fsqMask, int s)
            => Run(inputsEmbeds, featEmbedMasked, fsqMask, s);

        public LmStepResult Step(float[] currEmbed)
            => Run(currEmbed, currEmbed, _singleFsqMask, 1);

        private LmStepResult Run(float[] embeds, float[] fem, float[] fsqMask, int s)
        {
            var own = new List<OrtValue>(5);
            IDisposableReadOnlyCollection<OrtValue> outputs;
            try
            {
                int total = _pastLen + s;
                _seqlensKInput[0] = total - 1;
                _totalSequenceLengthInput[0] = total;
                _embedsShape[0] = 1;
                _embedsShape[1] = s;
                _embedsShape[2] = _H;
                _fsqMaskShape[0] = 1;
                _fsqMaskShape[1] = s;
                var inputs = new OrtValue[5 + _nKv];
                inputs[0] = OnnxIo.MakeFloat(_dtype, embeds, _embedsShape, own);
                inputs[1] = OnnxIo.MakeFloat(_dtype, fem, _embedsShape, own);
                inputs[2] = OnnxIo.MakeFloat(_dtype, fsqMask, _fsqMaskShape, own);
                inputs[3] = OnnxIo.MakeInt(_seqlensKInput, Vector1Shape, own);
                inputs[4] = OnnxIo.MakeInt(_totalSequenceLengthInput, ScalarShape, own);
                for (int i = 0; i < _nKv; i++)
                    inputs[5 + i] = _prevOutputs is null ? _emptyPast! : _prevOutputs[4 + i];
                outputs = _session.Run(_run, _inNames, inputs, _outNames);
            }
            finally { OnnxIo.Dispose(own); }

            float[] dit = OnnxIo.ReadFloat(outputs[0], _dtype);
            float[] stop = OnnxIo.ReadFloat(outputs[1], _dtype);
            _prevOutputs?.Dispose();   // 其 present 已被本步消费为 past
            _prevOutputs = outputs;
            _pastLen += s;
            return new LmStepResult(dit, stop);
        }

        public void Dispose()
        {
            _prevOutputs?.Dispose();
            _emptyPast?.Dispose();
            _run.Dispose();
            _session.Dispose();
        }
    }
}
