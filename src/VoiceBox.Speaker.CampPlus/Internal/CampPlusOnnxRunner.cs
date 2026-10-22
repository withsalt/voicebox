using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Speaker.CampPlus.Models;

namespace VoiceBox.Speaker.CampPlus.Internal
{
    /// <summary>
    /// campplus.onnx 的推理封装: feats [1, T, 80] (fbank+CMN) → embedding [1, 192]。
    /// 时间轴动态, 同一会话服务任意窗口/整段长度。非线程安全, 由服务层加锁。
    /// </summary>
    internal sealed class CampPlusOnnxRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions = new();
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;
        private readonly OrtValue[] _inputValues = new OrtValue[1];
        private readonly OrtValue[]? _outputValues;
        private readonly long[] _inputShape;
        private readonly float[]? _outputBuffer;
        private readonly OrtValue? _outputValue;
        private readonly int _featureDim;
        private readonly int _embeddingSize;
        private readonly bool _useReusableIoBinding;
        private OrtIoBinding? _reusableIoBinding;
        private OrtValue? _reusableInputValue;
        private float[]? _reusableInputOwner;
        private int _reusableInputFrames;
        private bool _disposed;

        public CampPlusOnnxRunner(string modelPath, CampPlusOptions options, int featureDim, int embeddingSize)
        {
            _featureDim = featureDim;
            _embeddingSize = embeddingSize;
            _useReusableIoBinding =
                options.ExecutionProvider == OnnxExecutionProvider.Cpu &&
                options.SessionOptionsFactory is null;
            using var sessionOptions = OnnxSessionFactory.Create(options, SessionPurpose.Dynamic);
            _session = new InferenceSession(modelPath, sessionOptions);
            try
            {
                if (_session.InputMetadata.Count != 1 || _session.OutputMetadata.Count != 1)
                    throw new InvalidOperationException(
                        $"Unexpected CampPlus model signature: {_session.InputMetadata.Count} inputs / {_session.OutputMetadata.Count} outputs (expect 1/1). [{modelPath}]");

                _inputNames = new[] { _session.InputMetadata.Keys.First() };
                _outputNames = new[] { _session.OutputMetadata.Keys.First() };

                int[] inputDims = _session.InputMetadata[_inputNames[0]].Dimensions;
                if (inputDims.Length != 3 || inputDims[2] != featureDim)
                    throw new InvalidOperationException(
                        $"Unexpected CampPlus model input shape [{string.Join(",", inputDims)}] (expect [batch, time, {featureDim}]). [{modelPath}]");

                _inputShape = new long[] { 1, 0, featureDim };
                // 能从 metadata 唯一推导输出形状时复用固定输出张量；自定义模型若有多个动态输出维，
                // 保留原有的 ORT 动态分配路径，不缩窄此前可接受的模型签名。
                if (TryResolveOutputShape(
                        _session.OutputMetadata[_outputNames[0]].Dimensions,
                        embeddingSize,
                        out long[] outputShape))
                {
                    _outputBuffer = GC.AllocateUninitializedArray<float>(embeddingSize, pinned: true);
                    _outputValue = OrtValue.CreateTensorValueFromMemory(_outputBuffer, outputShape);
                    _outputValues = new[] { _outputValue };
                }
            }
            catch
            {
                // 校验失败时释放已创建的原生会话, 避免宿主重试初始化造成累积泄漏
                _outputValue?.Dispose();
                _runOptions.Dispose();
                _session.Dispose();
                throw;
            }
        }

        /// <summary>推理: feats 前 frames*featureDim 个元素为行优先 [frames, featureDim] 特征。</summary>
        public float[] Run(float[] feats, int frames)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _inputShape[1] = frames;
            using var input = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                feats.AsMemory(0, frames * _featureDim),
                _inputShape);
            _inputValues[0] = input;
            return RunCore();
        }

        /// <summary>
        /// 推理并复用由调用方长期持有的固定输入缓冲对应的 OrtValue。
        /// 调用方必须保证 <paramref name="feats"/> 在后续调用前保持存活且不更换数组实例。
        /// </summary>
        public float[] RunReusable(float[] feats, int frames)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!ReferenceEquals(_reusableInputOwner, feats) || _reusableInputFrames != frames)
            {
                _inputShape[1] = frames;
                OrtValue nextInputValue = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance,
                    feats.AsMemory(0, frames * _featureDim),
                    _inputShape);
                OrtIoBinding? nextBinding = null;
                try
                {
                    if (_useReusableIoBinding && _outputValue is not null)
                    {
                        nextBinding = _session.CreateIoBinding();
                        nextBinding.BindInput(_inputNames[0], nextInputValue);
                        nextBinding.BindOutput(_outputNames[0], _outputValue);
                    }
                }
                catch
                {
                    nextBinding?.Dispose();
                    nextInputValue.Dispose();
                    throw;
                }

                _reusableIoBinding?.Dispose();
                _reusableInputValue?.Dispose();
                _reusableIoBinding = nextBinding;
                _reusableInputValue = nextInputValue;
                _reusableInputOwner = feats;
                _reusableInputFrames = frames;
            }

            _inputValues[0] = _reusableInputValue!;
            return RunCore(_reusableIoBinding);
        }

        private float[] RunCore(OrtIoBinding? binding = null)
        {
            if (binding is not null)
            {
                _session.RunWithBinding(_runOptions, binding);
                return _outputBuffer!.AsSpan().ToArray();
            }

            if (_outputValue is not null)
            {
                _session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues!);

                // 调用方拥有并可能长期保存返回值；只复制最终 192 个 float，内部固定输出缓冲可立即供下次推理复用。
                return _outputBuffer!.AsSpan().ToArray();
            }

            using var outputs = _session.Run(_runOptions, _inputNames, _inputValues, _outputNames);
            var span = outputs[0].GetTensorDataAsSpan<float>();
            if (span.Length != _embeddingSize)
                throw new InvalidOperationException(
                    $"Unexpected embedding size {span.Length} (expect {_embeddingSize}).");
            return span.ToArray();
        }

        private static bool TryResolveOutputShape(int[] dimensions, int elementCount, out long[] shape)
        {
            shape = new long[dimensions.Length];
            if (dimensions.Length == 0) return false;

            long knownProduct = 1;
            int dynamicIndex = -1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                int dimension = dimensions[i];
                if (dimension <= 0)
                {
                    if (dynamicIndex >= 0) return false;
                    dynamicIndex = i;
                    continue;
                }
                shape[i] = dimension;
                knownProduct = checked(knownProduct * dimension);
            }

            if (dynamicIndex >= 0)
            {
                if (knownProduct == 0 || elementCount % knownProduct != 0) return false;
                shape[dynamicIndex] = elementCount / knownProduct;
                knownProduct *= shape[dynamicIndex];
            }
            return knownProduct == elementCount;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _reusableIoBinding?.Dispose();
            _reusableInputValue?.Dispose();
            _outputValue?.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
        }
    }
}
