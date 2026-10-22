using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 封装 encoder/decoder/joiner 三个 ONNX session, 暴露 transducer 流式推理所需的全部操作。
    /// 整个生命周期与服务一致, 单流式 (非线程安全)。
    /// </summary>
    internal sealed class TransducerModel : IDisposable
    {
        private readonly InferenceSession _encoder;
        private readonly InferenceSession _decoder;
        private readonly InferenceSession _joiner;
        private readonly SessionOptions _encoderOpts;
        private readonly SessionOptions _decoderOpts;
        private readonly SessionOptions _joinerOpts;
        private readonly RunOptions _runOptions = new();

        private readonly string[] _encoderInputNames;
        private readonly string[] _encoderOutputNames;
        private readonly string[] _decoderInputNames;
        private readonly string[] _decoderOutputNames;
        private readonly string[] _joinerInputNames;
        private readonly string[] _joinerOutputNames;

        public EncoderMetadata EncoderMeta { get; }
        public int ContextSize { get; }
        public int VocabSize { get; }
        public int SubsamplingFactor => 4;
        public int ChunkSize => EncoderMeta.T;
        public int ChunkShift => EncoderMeta.DecodeChunkLen;
        public int FeatureDim => EncoderMeta.FeatureDim;

        public TransducerModel(SherpaOnnxKwsOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);

            void RequireFile(string path, string name)
            {
                if (string.IsNullOrWhiteSpace(path))
                    throw new ArgumentException($"{name} path must be provided.", nameof(options));
                if (!File.Exists(path))
                    throw new FileNotFoundException($"{name} not found: {path}", path);
            }
            RequireFile(options.EncoderModelPath, "Encoder model");
            RequireFile(options.DecoderModelPath, "Decoder model");
            RequireFile(options.JoinerModelPath, "Joiner model");

            // 三个 session 形状都会随调用变化 (batch / hyp 数), 因此都用 Dynamic 模式。
            _encoderOpts = OnnxSessionFactory.Create(options, SessionPurpose.Dynamic);
            _decoderOpts = OnnxSessionFactory.Create(options, SessionPurpose.Dynamic);
            _joinerOpts = OnnxSessionFactory.Create(options, SessionPurpose.Dynamic);

            _encoder = new InferenceSession(options.EncoderModelPath, _encoderOpts);
            _decoder = new InferenceSession(options.DecoderModelPath, _decoderOpts);
            _joiner = new InferenceSession(options.JoinerModelPath, _joinerOpts);

            _encoderInputNames = _encoder.InputMetadata.Keys.ToArray();
            _encoderOutputNames = _encoder.OutputMetadata.Keys.ToArray();
            _decoderInputNames = _decoder.InputMetadata.Keys.ToArray();
            _decoderOutputNames = _decoder.OutputMetadata.Keys.ToArray();
            _joinerInputNames = _joiner.InputMetadata.Keys.ToArray();
            _joinerOutputNames = _joiner.OutputMetadata.Keys.ToArray();

            EncoderMeta = ReadEncoderMetadata(_encoder);
            var decMeta = _decoder.ModelMetadata.CustomMetadataMap;
            ContextSize = int.Parse(decMeta["context_size"]);
            VocabSize = int.Parse(decMeta["vocab_size"]);

            // 校验状态数量与 metadata 是否一致 (encoder 输入数 = 1 + state count)
            int expectedStateCount = EncoderMeta.NumStateTensors;
            int actualStateCount = _encoderInputNames.Length - 1;
            if (expectedStateCount != actualStateCount)
            {
                throw new InvalidOperationException(
                    $"Encoder state count mismatch: metadata says {expectedStateCount}, model has {actualStateCount} state inputs.");
            }
        }

        // ---------------- Encoder ----------------

        /// <summary>
        /// 构建全零的 encoder 初始状态 (与 sherpa-onnx C++ GetEncoderInitStates 等价)。
        /// </summary>
        public EncoderState CreateInitialEncoderState()
        {
            int feat = EncoderMeta.FeatureDim;
            int embedDim = EncoderMeta.EmbedDim;
            var state = new EncoderState();

            var floatBufs = new List<float[]>();
            var floatShapes = new List<long[]>();

            for (int groupIdx = 0; groupIdx < EncoderMeta.EncoderDims.Length; groupIdx++)
            {
                int numLayers = EncoderMeta.NumEncoderLayers[groupIdx];
                int keyDim = EncoderMeta.QueryHeadDims[groupIdx] * EncoderMeta.NumHeads[groupIdx];
                int valueDim = EncoderMeta.ValueHeadDims[groupIdx] * EncoderMeta.NumHeads[groupIdx];
                int nonlinAttnHeadDim = 3 * EncoderMeta.EncoderDims[groupIdx] / 4;
                int leftCtx = EncoderMeta.LeftContextLen[groupIdx];
                int cnnHalf = EncoderMeta.CnnModuleKernels[groupIdx] / 2;
                int encDim = EncoderMeta.EncoderDims[groupIdx];

                for (int li = 0; li < numLayers; li++)
                {
                    AddZerosFloat(floatBufs, floatShapes, leftCtx, 1, keyDim);                  // cached_key
                    AddZerosFloat(floatBufs, floatShapes, 1, 1, leftCtx, nonlinAttnHeadDim);    // cached_nonlin_attn
                    AddZerosFloat(floatBufs, floatShapes, leftCtx, 1, valueDim);                // cached_val1
                    AddZerosFloat(floatBufs, floatShapes, leftCtx, 1, valueDim);                // cached_val2
                    AddZerosFloat(floatBufs, floatShapes, 1, encDim, cnnHalf);                  // cached_conv1
                    AddZerosFloat(floatBufs, floatShapes, 1, encDim, cnnHalf);                  // cached_conv2
                }
            }
            // embed_states
            AddZerosFloat(floatBufs, floatShapes, 1, 128, 3, embedDim);
            // processed_lens (int64, 0)
            state.ProcessedLens = new long[1] { 0 };
            state.ProcessedLensShape = new long[] { 1 };

            state.FloatBuffers = floatBufs.ToArray();
            state.FloatShapes = floatShapes.ToArray();
            return state;
        }

        private static void AddZerosFloat(List<float[]> bufs, List<long[]> shapes, params long[] shape)
        {
            long total = 1;
            for (int i = 0; i < shape.Length; i++) total *= shape[i];
            bufs.Add(new float[total]);
            shapes.Add(shape);
        }

        /// <summary>
        /// 运行 encoder。features 形状 [1, ChunkSize, FeatureDim]。
        /// 输出 encoder_out (转为 float[] + shape), 并把新状态写回 <paramref name="state"/>。
        /// </summary>
        public (float[] EncoderOut, long[] EncoderOutShape) RunEncoder(float[] features, EncoderState state)
        {
            // 构造 OrtValue: 1 个 features + N 个 state
            int totalInputs = 1 + state.NumFloatStates + 1;
            if (totalInputs != _encoderInputNames.Length)
                throw new InvalidOperationException(
                    $"Encoder input count mismatch: expected {_encoderInputNames.Length}, prepared {totalInputs}.");

            var ortInputs = new OrtValue[totalInputs];
            try
            {
                ortInputs[0] = OrtValue.CreateTensorValueFromMemory<float>(features,
                    new long[] { 1, ChunkSize, FeatureDim });

                for (int i = 0; i < state.NumFloatStates; i++)
                {
                    ortInputs[i + 1] = OrtValue.CreateTensorValueFromMemory<float>(
                        state.FloatBuffers[i], state.FloatShapes[i]);
                }
                ortInputs[totalInputs - 1] = OrtValue.CreateTensorValueFromMemory<long>(
                    state.ProcessedLens, state.ProcessedLensShape);

                using var outputs = _encoder.Run(_runOptions,
                    _encoderInputNames, ortInputs, _encoderOutputNames);

                // outputs[0] = encoder_out; outputs[1..] = new states (顺序同 inputs[1..])
                var encOut = outputs[0];
                var encOutShape = encOut.GetTensorTypeAndShape().Shape.ToArray();
                var encOutData = encOut.GetTensorDataAsSpan<float>().ToArray();

                // 把新状态拷回托管缓冲 (复用旧 buffer 时若 shape 变化需要新分配)
                for (int i = 0; i < state.NumFloatStates; i++)
                {
                    var newState = outputs[i + 1];
                    var newShape = newState.GetTensorTypeAndShape().Shape.ToArray();
                    var newSpan = newState.GetTensorDataAsSpan<float>();

                    state.FloatShapes[i] = newShape;
                    if (state.FloatBuffers[i].Length != newSpan.Length)
                        state.FloatBuffers[i] = new float[newSpan.Length];
                    newSpan.CopyTo(state.FloatBuffers[i]);
                }
                // processed_lens
                var pl = outputs[outputs.Count - 1];
                var plSpan = pl.GetTensorDataAsSpan<long>();
                state.ProcessedLensShape = pl.GetTensorTypeAndShape().Shape.ToArray();
                if (state.ProcessedLens.Length != plSpan.Length)
                    state.ProcessedLens = new long[plSpan.Length];
                plSpan.CopyTo(state.ProcessedLens);

                return (encOutData, encOutShape);
            }
            finally
            {
                for (int i = 0; i < ortInputs.Length; i++) ortInputs[i]?.Dispose();
            }
        }

        // ---------------- Decoder ----------------

        /// <summary>
        /// 取最近 context_size 个 token 组成 decoder 输入 [N, context_size] int64。
        /// </summary>
        public (long[] Data, long[] Shape) BuildDecoderInput(IReadOnlyList<Hypothesis> hyps)
        {
            int n = hyps.Count;
            int cs = ContextSize;
            var data = new long[n * cs];
            for (int i = 0; i < n; i++)
            {
                var ys = hyps[i].Ys;
                int len = ys.Count;
                for (int k = 0; k < cs; k++)
                {
                    long token = ys[len - cs + k];
                    if (token < 0) token = 0; // 初始 hyp 用 -1 占位, 喂给 decoder 时折成 blank (0)
                    data[i * cs + k] = token;
                }
            }
            return (data, new long[] { n, cs });
        }

        /// <summary>运行 decoder。返回 [N, decoder_dim] 的 float[]。</summary>
        public float[] RunDecoder(long[] decoderInputData, long[] decoderInputShape, out long[] decoderOutShape)
        {
            using var input = OrtValue.CreateTensorValueFromMemory<long>(decoderInputData, decoderInputShape);
            using var outputs = _decoder.Run(_runOptions, _decoderInputNames, new[] { input }, _decoderOutputNames);
            var span = outputs[0].GetTensorDataAsSpan<float>();
            decoderOutShape = outputs[0].GetTensorTypeAndShape().Shape.ToArray();
            return span.ToArray();
        }

        // ---------------- Joiner ----------------

        /// <summary>
        /// 运行 joiner。 encOut[N, joiner_dim] + decOut[N, joiner_dim] -> logit[N, vocab_size]。
        /// </summary>
        public float[] RunJoiner(float[] encOut, long[] encOutShape, float[] decOut, long[] decOutShape)
        {
            using var encOrt = OrtValue.CreateTensorValueFromMemory<float>(encOut, encOutShape);
            using var decOrt = OrtValue.CreateTensorValueFromMemory<float>(decOut, decOutShape);
            using var outputs = _joiner.Run(_runOptions, _joinerInputNames, new[] { encOrt, decOrt }, _joinerOutputNames);
            return outputs[0].GetTensorDataAsSpan<float>().ToArray();
        }

        // ---------------- helpers ----------------

        private static EncoderMetadata ReadEncoderMetadata(InferenceSession session)
        {
            var meta = session.ModelMetadata.CustomMetadataMap;
            EncoderMetadata m = new();

            if (!meta.TryGetValue("model_type", out string? modelType) || modelType != "zipformer2")
            {
                throw new NotSupportedException(
                    $"Only sherpa-onnx zipformer2 streaming KWS models are supported; got model_type='{modelType ?? "<missing>"}'.");
            }

            m.T = int.Parse(meta["T"]);
            m.DecodeChunkLen = int.Parse(meta["decode_chunk_len"]);
            m.NumEncoderLayers = ParseIntList(meta["num_encoder_layers"]);
            m.EncoderDims = ParseIntList(meta["encoder_dims"]);
            m.QueryHeadDims = ParseIntList(meta["query_head_dims"]);
            m.ValueHeadDims = ParseIntList(meta["value_head_dims"]);
            m.NumHeads = ParseIntList(meta["num_heads"]);
            m.CnnModuleKernels = ParseIntList(meta["cnn_module_kernels"]);
            m.LeftContextLen = ParseIntList(meta["left_context_len"]);
            m.FeatureDim = 80;
            return m;
        }

        private static int[] ParseIntList(string s)
        {
            string[] parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries);
            int[] result = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++) result[i] = int.Parse(parts[i].Trim());
            return result;
        }

        public void Dispose()
        {
            _runOptions.Dispose();
            _encoder.Dispose();
            _decoder.Dispose();
            _joiner.Dispose();
            _encoderOpts.Dispose();
            _decoderOpts.Dispose();
            _joinerOpts.Dispose();
        }
    }
}
