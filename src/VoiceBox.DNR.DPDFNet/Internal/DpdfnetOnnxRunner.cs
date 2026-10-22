using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;

namespace VoiceBox.DNR.DPDFNet.Internal
{
    /// <summary>
    /// DPDFNet 流式 ONNX 推理器。封装单个 <see cref="InferenceSession"/> 及其循环状态。
    ///
    /// 模型为单帧 (streaming) 结构：
    ///   输入  spec[1,1,F,2] (实/虚部) + state_in[S]
    ///   输出  spec_e[1,1,F,2]         + state_out[S]
    /// 状态在帧间循环传递；初始状态由模型 metadata (erb/spec 归一化初值) 构造。
    /// 非线程安全：由上层 <see cref="DPDFNetDNRService"/> 持锁串行调用。
    /// </summary>
    internal sealed class DpdfnetOnnxRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly SessionOptions _sessionOptions;
        private readonly RunOptions _runOptions;

        private readonly string _inSpecName;
        private readonly string _inStateName;
        private readonly string _outSpecName;
        private readonly string _outStateName;

        private readonly int _freqBins;
        private readonly int _stateSize;
        private readonly float[] _initialState;

        // 循环状态与复用张量缓冲。
        private float[] _state;
        private readonly float[] _specBuffer;          // [F*2]，按 (f, ri) 行主序
        private readonly string[] _outputNames;
        private readonly string[] _inputNames;
        private readonly OrtValue _specValue;
        private readonly OrtValue _stateValue;
        private readonly OrtValue[] _inputValues;

        public int FreqBins => _freqBins;
        public int FftSize { get; }
        public int HopSize { get; }
        public int SampleRate { get; }
        public string WindowType { get; }

        public DpdfnetOnnxRunner(string modelPath, OnnxExecutionOptions onnxOptions)
        {
            ArgumentNullException.ThrowIfNull(onnxOptions);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"DPDFNet model file not found: {modelPath}", modelPath);

            _sessionOptions = OnnxSessionFactory.Create(onnxOptions, SessionPurpose.StaticShapes);
            _session = new InferenceSession(modelPath, _sessionOptions);
            _runOptions = new RunOptions();

            var inputs = _session.InputMetadata;
            var outputs = _session.OutputMetadata;
            if (inputs.Count < 2 || outputs.Count < 2)
                throw new InvalidOperationException(
                    $"Expected a streaming DPDFNet model with 2 inputs (spec, state) and 2 outputs, " +
                    $"got inputs=[{string.Join(",", inputs.Keys)}], outputs=[{string.Join(",", outputs.Keys)}].");

            // 按张量秩区分：谱为 4 维 [1,1,F,2]，状态为 1 维 [S]。
            // 不依赖名称/顺序，避免不同导出工具的命名差异导致错配。
            (_inSpecName, _inStateName) = IdentifySpecAndState(inputs, "input");
            (_outSpecName, _outStateName) = IdentifySpecAndState(outputs, "output");

            var meta = _session.ModelMetadata.CustomMetadataMap;
            FftSize = ReadIntMeta(meta, "n_fft", 320);
            HopSize = ReadIntMeta(meta, "hop_length", 160);
            SampleRate = ReadIntMeta(meta, "sample_rate", 16000);
            _freqBins = ReadIntMeta(meta, "freq_bins", FftSize / 2 + 1);
            WindowType = meta.TryGetValue("window_type", out var wt) ? wt : "vorbis";

            // state_in 形状给出权威状态长度。
            var stateDims = _session.InputMetadata[_inStateName].Dimensions;
            _stateSize = stateDims.Length > 0 && stateDims[0] > 0
                ? stateDims[0]
                : ReadIntMeta(meta, "state_size", 0);
            if (_stateSize <= 0)
                throw new InvalidOperationException("Unable to determine DPDFNet state size from model.");

            _initialState = BuildInitialState(meta, _stateSize);
            _state = (float[])_initialState.Clone();

            _specBuffer = new float[_freqBins * 2];
            _outputNames = new[] { _outSpecName, _outStateName };
            _inputNames = new[] { _inSpecName, _inStateName };
            _specValue = OrtValue.CreateTensorValueFromMemory(
                _specBuffer, new long[] { 1, 1, _freqBins, 2 });
            _stateValue = OrtValue.CreateTensorValueFromMemory(
                _state, new long[] { _stateSize });
            _inputValues = new[] { _specValue, _stateValue };
        }

        /// <summary>
        /// 运行一帧。输入谱 <paramref name="re"/> / <paramref name="im"/> (各长度 <see cref="FreqBins"/>)，
        /// 增强谱写回 <paramref name="outRe"/> / <paramref name="outIm"/>。循环状态在内部更新。
        /// </summary>
        public void Run(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> outRe, Span<float> outIm)
        {
            if (re.Length != _freqBins || im.Length != _freqBins)
                throw new ArgumentException($"input spec length must be {_freqBins}.");
            if (outRe.Length != _freqBins || outIm.Length != _freqBins)
                throw new ArgumentException($"output spec length must be {_freqBins}.");

            // 打包 spec[1,1,F,2]：(f,0)=re, (f,1)=im。
            for (int f = 0; f < _freqBins; f++)
            {
                _specBuffer[f * 2] = re[f];
                _specBuffer[f * 2 + 1] = im[f];
            }

            using IDisposableReadOnlyCollection<OrtValue> results =
                _session.Run(_runOptions, _inputNames, _inputValues, _outputNames);

            ReadOnlySpan<float> specSpan = results[0].GetTensorDataAsSpan<float>();
            if (specSpan.Length < _freqBins * 2)
                throw new InvalidOperationException($"DPDFNet spec output too small: {specSpan.Length} < {_freqBins * 2}.");
            for (int f = 0; f < _freqBins; f++)
            {
                outRe[f] = specSpan[f * 2];
                outIm[f] = specSpan[f * 2 + 1];
            }

            // 回填循环状态到下一帧输入张量。
            ReadOnlySpan<float> stateSpan = results[1].GetTensorDataAsSpan<float>();
            if (stateSpan.Length != _stateSize)
                throw new InvalidOperationException($"DPDFNet state output size mismatch: {stateSpan.Length} != {_stateSize}.");
            stateSpan.CopyTo(_state);
        }

        /// <summary>重置循环状态到初始值。</summary>
        public void Reset() => Array.Copy(_initialState, _state, _stateSize);

        /// <summary>
        /// 在两个张量中按秩区分谱 (4 维) 与状态 (1 维)。返回 (specName, stateName)。
        /// </summary>
        private static (string spec, string state) IdentifySpecAndState(
            IReadOnlyDictionary<string, NodeMetadata> metadata, string kind)
        {
            string? spec = null;
            string? state = null;
            foreach (var kv in metadata)
            {
                int rank = kv.Value.Dimensions.Length;
                if (rank == 4 && spec is null) spec = kv.Key;
                else if (rank == 1 && state is null) state = kv.Key;
            }
            if (spec is null || state is null)
                throw new InvalidOperationException(
                    $"Could not identify spec (rank-4) and state (rank-1) tensors among {kind}s: " +
                    $"[{string.Join(",", metadata.Select(m => $"{m.Key}:{m.Value.Dimensions.Length}d"))}].");
            return (spec, state);
        }

        private static float[] BuildInitialState(IReadOnlyDictionary<string, string> meta, int stateSize)
        {
            // 与参考实现一致：state 前段写入 erb 归一化初值，紧接着写入 spec 归一化初值，其余为 0。
            var state = new float[stateSize];
            int erbSize = ReadIntMeta(meta, "erb_norm_state_size", 0);
            int specSize = ReadIntMeta(meta, "spec_norm_state_size", 0);
            float[] erbInit = ParseFloatArray(meta, "erb_norm_init");
            float[] specInit = ParseFloatArray(meta, "spec_norm_init");

            if (erbSize > 0)
            {
                if (erbInit.Length < erbSize)
                    throw new InvalidOperationException($"erb_norm_init has {erbInit.Length} values, expected {erbSize}.");
                Array.Copy(erbInit, 0, state, 0, erbSize);
            }
            if (specSize > 0)
            {
                if (specInit.Length < specSize)
                    throw new InvalidOperationException($"spec_norm_init has {specInit.Length} values, expected {specSize}.");
                if (erbSize + specSize > stateSize)
                    throw new InvalidOperationException("erb+spec norm sizes exceed state size.");
                Array.Copy(specInit, 0, state, erbSize, specSize);
            }
            return state;
        }

        private static int ReadIntMeta(IReadOnlyDictionary<string, string> meta, string key, int fallback)
            => meta.TryGetValue(key, out var v) && int.TryParse(v, out var i) ? i : fallback;

        private static float[] ParseFloatArray(IReadOnlyDictionary<string, string> meta, string key)
        {
            if (!meta.TryGetValue(key, out var v) || string.IsNullOrWhiteSpace(v))
                return Array.Empty<float>();
            var parts = v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var result = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                result[i] = float.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);
            return result;
        }

        public void Dispose()
        {
            _specValue.Dispose();
            _stateValue.Dispose();
            _runOptions.Dispose();
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}
