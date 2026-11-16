using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Wakeup.FsmnKws.Models;

namespace VoiceBox.Wakeup.FsmnKws.Internal
{
    /// <summary>
    /// 流式 FSMN-KWS ONNX 推理封装。
    /// <para>
    /// 模型契约 (scripts/kws-xiaoyun-onnx-export 导出):
    /// 输入 feats [1, T, 400] + cache0..3 [1, 128, 11, 1]; 输出 probs [1, T, 2599] (softmax 后验)
    /// + new_cache0..3。cache 为精确 lookahead 缓存: 输出帧 j 对应输入帧 j-8 (4 层 × rorder 2),
    /// 稳态输出与整段推理一致, 与分块大小无关。
    /// </para>
    /// </summary>
    internal sealed class StreamingFsmnModel : IDisposable
    {
        public const int InputDim = 400;
        public const int OutputDim = 2599;
        public const int NumLayers = 4;
        public const int ProjDim = 128;
        public const int CacheLen = 11;
        /// <summary>输出相对输入的固定延迟 (LFR 帧, 每帧 30ms)。</summary>
        public const int OutputDelayFrames = 8;

        private static readonly long[] CacheShape = { 1, ProjDim, CacheLen, 1 };
        private const int TensorBindingCacheSize = 4;

        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions = new();

        private readonly string[] _inputNames;   // feats, cache0..3
        private readonly string[] _outputNames;  // probs, new_cache0..3

        // cache 双缓冲: bank[_curBank] 作输入, bank[1-_curBank] 收输出, 跑完交换
        private readonly float[][][] _cacheBufs = new float[2][][];
        private readonly OrtValue[][] _cacheVals = new OrtValue[2][];
        private int _curBank;

        private readonly List<TensorBinding> _bindings = new(TensorBindingCacheSize);
        private long _bindingUseCounter;

        private readonly OrtValue[] _inputVals = new OrtValue[NumLayers + 1];
        private readonly OrtValue[] _outputVals = new OrtValue[NumLayers + 1];

        public StreamingFsmnModel(FsmnKwsOptions options)
        {
            if (!File.Exists(options.ModelPath))
                throw new FileNotFoundException($"FsmnKws model not found: {options.ModelPath}", options.ModelPath);

            var sessionOptions = OnnxSessionFactory.Create(options);
            _session = new InferenceSession(options.ModelPath, sessionOptions);

            _inputNames = new string[NumLayers + 1];
            _outputNames = new string[NumLayers + 1];
            _inputNames[0] = "feats";
            _outputNames[0] = "probs";
            for (int i = 0; i < NumLayers; i++)
            {
                _inputNames[i + 1] = $"cache{i}";
                _outputNames[i + 1] = $"new_cache{i}";
            }

            foreach (string required in _inputNames)
            {
                if (!_session.InputMetadata.ContainsKey(required))
                    throw new InvalidDataException(
                        $"Model input '{required}' missing; not a kws_streaming.onnx? " +
                        $"actual inputs: [{string.Join(", ", _session.InputMetadata.Keys)}]");
            }

            for (int bank = 0; bank < 2; bank++)
            {
                _cacheBufs[bank] = new float[NumLayers][];
                _cacheVals[bank] = new OrtValue[NumLayers];
                for (int i = 0; i < NumLayers; i++)
                {
                    var buf = new float[ProjDim * CacheLen];
                    _cacheBufs[bank][i] = buf;
                    _cacheVals[bank][i] = OrtValue.CreateTensorValueFromMemory(buf, CacheShape);
                }
            }

            // 预热: 触发 ORT 内核初始化/内存池分配, 避免首帧音频卡顿
            GetFeatsBuffer(1).Clear();
            Run(1);
            ResetCache();
        }

        public void ResetCache()
        {
            for (int bank = 0; bank < 2; bank++)
                for (int i = 0; i < NumLayers; i++)
                    Array.Clear(_cacheBufs[bank][i]);
            _curBank = 0;
        }

        /// <summary>
        /// 取容纳 frameCount 个 LFR 帧的输入缓冲 (行优先 [frameCount, 400]), 由调用方填充后调
        /// <see cref="Run"/>。
        /// </summary>
        public Span<float> GetFeatsBuffer(int frameCount)
        {
            if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
            TensorBinding binding = GetBinding(frameCount);
            return binding.Feats.AsSpan(0, frameCount * InputDim);
        }

        /// <summary>
        /// 推理 frameCount 个 LFR 帧 (输入须已写入 <see cref="GetFeatsBuffer"/> 返回的缓冲),
        /// 返回 [frameCount, 2599] 后验概率 (行优先, 指向内部缓冲, 下次调用前有效), 并更新 cache。
        /// </summary>
        public ReadOnlySpan<float> Run(int frameCount)
        {
            if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
            TensorBinding binding = GetBinding(frameCount);

            int inBank = _curBank;
            int outBank = 1 - _curBank;
            _inputVals[0] = binding.FeatsValue;
            _outputVals[0] = binding.ProbsValue;
            for (int i = 0; i < NumLayers; i++)
            {
                _inputVals[i + 1] = _cacheVals[inBank][i];
                _outputVals[i + 1] = _cacheVals[outBank][i];
            }

            _session.Run(_runOptions, _inputNames, _inputVals, _outputNames, _outputVals);
            _curBank = outBank;
            return binding.Probs.AsSpan(0, frameCount * OutputDim);
        }

        private TensorBinding GetBinding(int frameCount)
        {
            for (int i = 0; i < _bindings.Count; i++)
            {
                TensorBinding existing = _bindings[i];
                if (existing.FrameCount == frameCount)
                {
                    existing.LastUsed = ++_bindingUseCounter;
                    return existing;
                }
            }

            var binding = new TensorBinding(frameCount, ++_bindingUseCounter);
            if (_bindings.Count < TensorBindingCacheSize)
            {
                _bindings.Add(binding);
                return binding;
            }

            int oldestIndex = 0;
            long oldestUse = _bindings[0].LastUsed;
            for (int i = 1; i < _bindings.Count; i++)
            {
                if (_bindings[i].LastUsed >= oldestUse) continue;
                oldestIndex = i;
                oldestUse = _bindings[i].LastUsed;
            }

            _bindings[oldestIndex].Dispose();
            _bindings[oldestIndex] = binding;
            return binding;
        }

        public void Dispose()
        {
            foreach (TensorBinding binding in _bindings)
                binding.Dispose();
            for (int bank = 0; bank < 2; bank++)
                if (_cacheVals[bank] != null)
                    foreach (var v in _cacheVals[bank]) v?.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
        }

        private sealed class TensorBinding : IDisposable
        {
            public TensorBinding(int frameCount, long lastUsed)
            {
                FrameCount = frameCount;
                LastUsed = lastUsed;
                Feats = new float[frameCount * InputDim];
                Probs = new float[frameCount * OutputDim];
                FeatsValue = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance,
                    Feats.AsMemory(0, Feats.Length),
                    new long[] { 1, frameCount, InputDim });
                ProbsValue = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance,
                    Probs.AsMemory(0, Probs.Length),
                    new long[] { 1, frameCount, OutputDim });
            }

            public int FrameCount { get; }

            public long LastUsed { get; set; }

            public float[] Feats { get; }

            public float[] Probs { get; }

            public OrtValue FeatsValue { get; }

            public OrtValue ProbsValue { get; }

            public void Dispose()
            {
                FeatsValue.Dispose();
                ProbsValue.Dispose();
            }
        }
    }
}
