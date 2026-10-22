using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Speaker.WeSpeaker.Models;

namespace VoiceBox.Speaker.WeSpeaker.Internal
{
    /// <summary>
    /// cnceleb_resnet34.onnx 推理封装: feats [1,T,80] (fbank+CMN) → embs [1,256]。
    /// 时间轴动态, 同一会话服务任意窗口/整段长度。非线程安全, 由服务层加锁。
    /// </summary>
    internal sealed class WeSpeakerOnnxRunner : IDisposable
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
        private OrtValue? _reusableInputValue;
        private float[]? _reusableInputOwner;
        private int _reusableInputFrames;
        private bool _disposed;

        public WeSpeakerOnnxRunner(
            string modelPath,
            WeSpeakerOptions options,
            WeSpeakerModelConfig config)
        {
            _featureDim = config.FeatureDim;
            _embeddingSize = config.EmbeddingSize;
            using var sessionOptions = OnnxSessionFactory.Create(options, SessionPurpose.Dynamic);
            _session = new InferenceSession(modelPath, sessionOptions);
            try
            {
                if (_session.InputMetadata.Count != 1 || _session.OutputMetadata.Count != 1)
                    throw new InvalidOperationException(
                        $"Unexpected WeSpeaker model signature: {_session.InputMetadata.Count} inputs / {_session.OutputMetadata.Count} outputs (expect 1/1). [{modelPath}]");

                _inputNames = new[] { config.InputName };
                _outputNames = new[] { config.OutputName };
                if (!_session.InputMetadata.ContainsKey(config.InputName) ||
                    !_session.OutputMetadata.ContainsKey(config.OutputName))
                {
                    throw new InvalidOperationException(
                        $"Unexpected WeSpeaker tensor names (expect input '{config.InputName}' and output '{config.OutputName}'). [{modelPath}]");
                }

                NodeMetadata inputMetadata = _session.InputMetadata[_inputNames[0]];
                NodeMetadata outputMetadata = _session.OutputMetadata[_outputNames[0]];
                if (inputMetadata.ElementType != typeof(float) ||
                    outputMetadata.ElementType != typeof(float))
                {
                    throw new InvalidOperationException(
                        $"Unexpected WeSpeaker tensor types {inputMetadata.ElementType.Name}/{outputMetadata.ElementType.Name} " +
                        $"(expect Float32/Float32). [{modelPath}]");
                }

                int[] inputDims = inputMetadata.Dimensions;
                if (inputDims.Length != 3 || inputDims[2] != _featureDim ||
                    inputDims[0] > 1 || inputDims[1] > 0)
                    throw new InvalidOperationException(
                        $"Unexpected WeSpeaker model input shape [{string.Join(",", inputDims)}] " +
                        $"(expect [1/dynamic batch, dynamic time, {_featureDim}]). [{modelPath}]");

                int[] outputDims = outputMetadata.Dimensions;
                if (outputDims.Length != 2 || outputDims[0] > 1 || outputDims[1] != _embeddingSize)
                    throw new InvalidOperationException(
                        $"Unexpected WeSpeaker model output shape [{string.Join(",", outputDims)}] " +
                        $"(expect [batch, {_embeddingSize}]). [{modelPath}]");

                _inputShape = new long[] { 1, 0, _featureDim };
                // 能从 metadata 唯一推导输出形状时复用固定输出张量；自定义模型若有多个动态输出维，
                // 保留原有的 ORT 动态分配路径，不缩窄此前可接受的模型签名。
                if (TryResolveOutputShape(
                        outputDims,
                        _embeddingSize,
                        out long[] outputShape))
                {
                    _outputBuffer = GC.AllocateUninitializedArray<float>(_embeddingSize, pinned: true);
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
                _reusableInputValue?.Dispose();
                _reusableInputValue = nextInputValue;
                _reusableInputOwner = feats;
                _reusableInputFrames = frames;
            }

            _inputValues[0] = _reusableInputValue!;
            return RunCore();
        }

        private float[] RunCore()
        {
            if (_outputValue is not null)
            {
                _session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues!);

                // 调用方拥有并可能长期保存返回值；内部固定输出缓冲立即供下次推理复用。
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
            _reusableInputValue?.Dispose();
            _outputValue?.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
        }
    }
}
