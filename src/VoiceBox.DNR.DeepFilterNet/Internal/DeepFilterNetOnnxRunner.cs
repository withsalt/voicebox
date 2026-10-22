using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;

namespace VoiceBox.DNR.DeepFilterNet.Internal
{
    /// <summary>
    /// DeepFilterNet 有状态流式 ONNX 推理器，封装 <c>combined_streaming.onnx</c>（dfn3_ll 等）。
    ///
    /// 模型把三个 GRU 隐状态外置为 I/O：
    ///   输入  feat_erb[1,1,T,nbErb] + feat_spec[1,2,T,nbDf] + h0[1,1,H] + erb_h0[L_e,1,H_e] + df_h0[L_d,1,H_d]
    ///   输出  lsnr + m[.,1,1,nbErb] + coefs[1,1,nbDf,dfOrder*2] + h1 + erb_h1 + df_h1
    /// 编码器卷积看 T 帧上下文，模型内部把时间轴切到 T[-1:] 再进 GRU，故输出恒为 1 帧 (take index 0)。
    /// 三个 GRU 态在帧间循环传递；<see cref="Reset"/> 清零。非线程安全。
    /// </summary>
    internal sealed class DeepFilterNetOnnxRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly SessionOptions _sessionOptions;
        private readonly RunOptions _runOptions;

        private readonly int _encWindow; // T：编码器卷积上下文帧数
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;

        // 复用输入缓冲与 OrtValue（固定形状，StaticShapes）。
        private readonly float[] _featErb;   // [T*nbErb]
        private readonly float[] _featSpec;  // [2*T*nbDf]
        private readonly float[] _h0;        // [H]
        private readonly float[] _erbH0;     // [L_e*H_e]
        private readonly float[] _dfH0;      // [L_d*H_d]
        private readonly OrtValue _vFeatErb;
        private readonly OrtValue _vFeatSpec;
        private readonly OrtValue _vH0;
        private readonly OrtValue _vErbH0;
        private readonly OrtValue _vDfH0;
        private readonly OrtValue[] _inputValues;

        public int NbErb { get; }
        public int NbDf { get; }
        public int DfOrder { get; }
        public int EncWindow => _encWindow;
        public int EncHiddenDim { get; }

        public DeepFilterNetOnnxRunner(string modelPath, OnnxExecutionOptions onnxOptions, int encWindow)
        {
            ArgumentNullException.ThrowIfNull(onnxOptions);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"DeepFilterNet model file not found: {modelPath}", modelPath);
            if (encWindow <= 0)
                throw new ArgumentOutOfRangeException(nameof(encWindow), encWindow, "encWindow must be > 0.");

            _encWindow = encWindow;
            SessionOptions? sessionOptions = null;
            InferenceSession? session = null;
            RunOptions? runOptions = null;
            OrtValue? vFeatErb = null;
            OrtValue? vFeatSpec = null;
            OrtValue? vH0 = null;
            OrtValue? vErbH0 = null;
            OrtValue? vDfH0 = null;
            bool initialized = false;

            try
            {
                sessionOptions = OnnxSessionFactory.Create(onnxOptions, SessionPurpose.StaticShapes);
                session = new InferenceSession(modelPath, sessionOptions);
                runOptions = new RunOptions();

                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;
                RequireKeys(inputs, "input", "feat_erb", "feat_spec", "h0", "erb_h0", "df_h0");
                RequireKeys(outputs, "output", "lsnr", "m", "coefs", "h1", "erb_h1", "df_h1");

                // 从模型张量维度推导各尺寸（动态维以 1 计）。
                NbErb = Dim(inputs["feat_erb"], 3, 32);
                NbDf = Dim(inputs["feat_spec"], 3, 96);
                EncHiddenDim = Dim(inputs["h0"], 2, 512);
                int erbLayers = Dim(inputs["erb_h0"], 0, 2);
                int erbHidden = Dim(inputs["erb_h0"], 2, EncHiddenDim);
                int dfLayers = Dim(inputs["df_h0"], 0, 3);
                int dfHidden = Dim(inputs["df_h0"], 2, EncHiddenDim);
                DfOrder = Math.Max(1, Dim(outputs["coefs"], 3, 10) / 2);

                _inputNames = new[] { "feat_erb", "feat_spec", "h0", "erb_h0", "df_h0" };
                _outputNames = new[] { "lsnr", "m", "coefs", "h1", "erb_h1", "df_h1" };

                _featErb = new float[_encWindow * NbErb];
                _featSpec = new float[2 * _encWindow * NbDf];
                _h0 = new float[EncHiddenDim];
                _erbH0 = new float[erbLayers * erbHidden];
                _dfH0 = new float[dfLayers * dfHidden];

                vFeatErb = OrtValue.CreateTensorValueFromMemory(_featErb, new long[] { 1, 1, _encWindow, NbErb });
                vFeatSpec = OrtValue.CreateTensorValueFromMemory(_featSpec, new long[] { 1, 2, _encWindow, NbDf });
                vH0 = OrtValue.CreateTensorValueFromMemory(_h0, new long[] { 1, 1, EncHiddenDim });
                vErbH0 = OrtValue.CreateTensorValueFromMemory(_erbH0, new long[] { erbLayers, 1, erbHidden });
                vDfH0 = OrtValue.CreateTensorValueFromMemory(_dfH0, new long[] { dfLayers, 1, dfHidden });

                _sessionOptions = sessionOptions;
                _session = session;
                _runOptions = runOptions;
                _vFeatErb = vFeatErb;
                _vFeatSpec = vFeatSpec;
                _vH0 = vH0;
                _vErbH0 = vErbH0;
                _vDfH0 = vDfH0;
                _inputValues = new[] { _vFeatErb, _vFeatSpec, _vH0, _vErbH0, _vDfH0 };
                initialized = true;
            }
            finally
            {
                if (!initialized)
                {
                    vDfH0?.Dispose();
                    vErbH0?.Dispose();
                    vH0?.Dispose();
                    vFeatSpec?.Dispose();
                    vFeatErb?.Dispose();
                    runOptions?.Dispose();
                    session?.Dispose();
                    sessionOptions?.Dispose();
                }
            }
        }

        /// <summary>
        /// 运行一帧推理。
        /// </summary>
        /// <param name="featErb">ERB 特征 [T*nbErb]，时间-频率行主序 (oldest-first)。</param>
        /// <param name="featSpec">复数谱特征 [2*T*nbDf]，planar：先全部实部 [T*nbDf] 再全部虚部。</param>
        /// <param name="mask">输出 ERB 掩码 [nbErb]。</param>
        /// <param name="coefs">输出深度滤波系数 [nbDf*dfOrder*2]。</param>
        /// <returns>当前帧 local SNR (dB)。</returns>
        public float Run(ReadOnlySpan<float> featErb, ReadOnlySpan<float> featSpec, Span<float> mask, Span<float> coefs)
        {
            if (featErb.Length != _featErb.Length)
                throw new ArgumentException($"featErb length must be {_featErb.Length}, got {featErb.Length}.");
            if (featSpec.Length != _featSpec.Length)
                throw new ArgumentException($"featSpec length must be {_featSpec.Length}, got {featSpec.Length}.");
            if (mask.Length != NbErb)
                throw new ArgumentException($"mask length must be {NbErb}.");
            int coefLen = NbDf * DfOrder * 2;
            if (coefs.Length != coefLen)
                throw new ArgumentException($"coefs length must be {coefLen}.");

            featErb.CopyTo(_featErb);
            featSpec.CopyTo(_featSpec);

            using IDisposableReadOnlyCollection<OrtValue> results =
                _session.Run(_runOptions, _inputNames, _inputValues, _outputNames);

            // outputs: [0]=lsnr, [1]=m, [2]=coefs, [3]=h1, [4]=erb_h1, [5]=df_h1
            ReadOnlySpan<float> lsnrSpan = results[0].GetTensorDataAsSpan<float>();
            float lsnr = lsnrSpan.Length > 0 ? lsnrSpan[0] : 0f;

            ReadOnlySpan<float> maskSpan = results[1].GetTensorDataAsSpan<float>();
            maskSpan.Slice(0, NbErb).CopyTo(mask);

            ReadOnlySpan<float> coefSpan = results[2].GetTensorDataAsSpan<float>();
            coefSpan.Slice(0, coefLen).CopyTo(coefs);

            CopyState(results[3], _h0);
            CopyState(results[4], _erbH0);
            CopyState(results[5], _dfH0);

            return lsnr;
        }

        /// <summary>清零三个 GRU 隐状态，开始新的独立流。</summary>
        public void Reset()
        {
            Array.Clear(_h0);
            Array.Clear(_erbH0);
            Array.Clear(_dfH0);
        }

        private static void CopyState(OrtValue output, float[] dst)
        {
            ReadOnlySpan<float> src = output.GetTensorDataAsSpan<float>();
            if (src.Length != dst.Length)
                throw new InvalidOperationException(
                    $"DeepFilterNet state output length mismatch: expected {dst.Length}, got {src.Length}.");
            src.CopyTo(dst);
        }

        private static int Dim(NodeMetadata meta, int axis, int fallback)
        {
            int[] dims = meta.Dimensions;
            if (dims is null || axis >= dims.Length) return fallback;
            return dims[axis] > 0 ? dims[axis] : fallback;
        }

        private static void RequireKeys(IReadOnlyDictionary<string, NodeMetadata> meta, string kind, params string[] keys)
        {
            foreach (var k in keys)
            {
                if (!meta.ContainsKey(k))
                    throw new InvalidOperationException(
                        $"DeepFilterNet streaming model missing {kind} '{k}'. " +
                        $"Available {kind}s: [{string.Join(", ", meta.Keys)}]. " +
                        "Expected a combined_streaming.onnx with externalized GRU states (h0/erb_h0/df_h0).");
            }
        }

        public void Dispose()
        {
            _vFeatErb.Dispose();
            _vFeatSpec.Dispose();
            _vH0.Dispose();
            _vErbH0.Dispose();
            _vDfH0.Dispose();
            _runOptions.Dispose();
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}
