using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using VoiceBox.Core.Onnx;
using VoiceBox.Wakeup.OpenWakeWord.Models;

namespace VoiceBox.Wakeup.OpenWakeWord.Internal
{
    /// <summary>
    /// 单个唤醒词的 ONNX 推理 + 状态机。
    /// 负责对一帧 (80ms) 新特征做预测、维护最近 30 帧的分数缓冲、并对外报告是否命中。
    /// 命中后会按 <see cref="WakeWordModelDefinition.DebounceMs"/> 抑制重复触发。
    /// </summary>
    internal sealed class WakeWordDetector : IDisposable
    {
        private const int EmbeddingDim = 96;
        private const int PredictionBufferSize = 30;
        private const int InitialZeroFrames = 5; // 与 Python 版一致, 前 5 帧强制 0

        private readonly InferenceSession _session;
        private readonly SessionOptions _sessionOptions;
        private readonly string _inputName;
        private readonly string _outputName;

        private readonly int _featureFrames;
        private readonly float[] _predictionBuffer = new float[PredictionBufferSize];
        private int _predictionCount;
        private int _predictionHead; // ring buffer head

        // 防抖
        private long _lastTriggerSampleIndex = long.MinValue;
        private readonly long _debounceSampleSpan;

        public WakeWordModelDefinition Definition { get; }

        public int FeatureFrames => _featureFrames;

        public WakeWordDetector(
            WakeWordModelDefinition definition,
            uint sampleRate,
            OnnxExecutionOptions onnxOptions)
        {
            Definition = definition ?? throw new ArgumentNullException(nameof(definition));
            if (string.IsNullOrWhiteSpace(definition.Keyword))
                throw new ArgumentException("Keyword cannot be empty.", nameof(definition));
            if (string.IsNullOrWhiteSpace(definition.ModelPath))
                throw new ArgumentException("ModelPath cannot be empty.", nameof(definition));
            if (!File.Exists(definition.ModelPath))
                throw new FileNotFoundException($"Wakeword model not found: {definition.ModelPath}", definition.ModelPath);

            _sessionOptions = OnnxSessionFactory.Create(onnxOptions);
            _session = new InferenceSession(definition.ModelPath, _sessionOptions);

            var inputMeta = _session.InputMetadata.First();
            _inputName = inputMeta.Key;
            // 模型输入形状: [batch=1, n_frames, 96]; n_frames 由模型决定
            int frames = 0;
            var dims = inputMeta.Value.Dimensions;
            if (dims.Length >= 2)
            {
                frames = dims[1];
            }
            if (frames <= 0)
                throw new InvalidOperationException(
                    $"Cannot determine feature frame count for wakeword model '{definition.Keyword}'.");
            _featureFrames = frames;

            _outputName = _session.OutputMetadata.Keys.First();

            _debounceSampleSpan = definition.DebounceMs > 0
                ? (long)Math.Ceiling(definition.DebounceMs / 1000.0 * sampleRate)
                : 0;
        }

        /// <summary>
        /// 复位状态 (清空历史预测、解除防抖)。
        /// </summary>
        public void Reset()
        {
            _predictionCount = 0;
            _predictionHead = 0;
            Array.Clear(_predictionBuffer, 0, _predictionBuffer.Length);
            _lastTriggerSampleIndex = long.MinValue;
        }

        /// <summary>
        /// 对当前最新一帧特征做推理。
        /// </summary>
        /// <param name="features">由 <see cref="AudioFeatureExtractor"/> 拷贝出的 [featureFrames, 96] 行优先 buffer。</param>
        /// <param name="processedSampleIndex">截至当前已经处理过的样本总数 (用于防抖与时间戳)。</param>
        /// <param name="score">本帧实际预测分数 (0..1, 未应用初始静默)。</param>
        /// <returns>是否触发唤醒事件。</returns>
        public bool Predict(ReadOnlySpan<float> features, long processedSampleIndex, out float score)
        {
            if (features.Length < _featureFrames * EmbeddingDim)
                throw new ArgumentException("Feature buffer too small.", nameof(features));

            var tensor = new DenseTensor<float>(new[] { 1, _featureFrames, EmbeddingDim });
            features.Slice(0, _featureFrames * EmbeddingDim).CopyTo(tensor.Buffer.Span);

            using var results = _session.Run(
                new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });

            var output = results.First(r => r.Name == _outputName).AsTensor<float>();
            // 输出可能为 [1, 1, 1] / [1, 1] / [1] — 都取第一个元素即可
            float raw = ReadFirstScalar(output);

            score = raw;

            // 与 Python 一致: 前 5 次预测强制视为 0 (但仍记录到 buffer)
            float effective = _predictionCount < InitialZeroFrames ? 0f : raw;

            AppendPrediction(effective);

            if (effective < Definition.Threshold)
                return false;

            // patience: 需要近 N 帧均 >= threshold
            if (Definition.Patience > 0)
            {
                int n = Definition.Patience;
                if (CountRecentAboveThreshold(n) < n)
                    return false;
            }

            // debounce: 距上次触发 < debounce 间隔则抑制
            if (_lastTriggerSampleIndex != long.MinValue &&
                processedSampleIndex - _lastTriggerSampleIndex < _debounceSampleSpan)
            {
                return false;
            }

            _lastTriggerSampleIndex = processedSampleIndex;
            return true;
        }

        private static float ReadFirstScalar(Tensor<float> output)
        {
            if (output is DenseTensor<float> dense)
                return dense.Buffer.Span[0];
            var arr = output.ToArray();
            return arr.Length > 0 ? arr[0] : 0f;
        }

        private void AppendPrediction(float value)
        {
            if (_predictionCount < PredictionBufferSize)
            {
                int idx = (_predictionHead + _predictionCount) % PredictionBufferSize;
                _predictionBuffer[idx] = value;
                _predictionCount++;
            }
            else
            {
                _predictionBuffer[_predictionHead] = value;
                _predictionHead = (_predictionHead + 1) % PredictionBufferSize;
            }
        }

        private int CountRecentAboveThreshold(int n)
        {
            int take = Math.Min(n, _predictionCount);
            int count = 0;
            for (int i = 0; i < take; i++)
            {
                int physical = (_predictionHead + _predictionCount - 1 - i) % PredictionBufferSize;
                if (physical < 0) physical += PredictionBufferSize;
                if (_predictionBuffer[physical] >= Definition.Threshold)
                    count++;
            }
            return count;
        }

        public void Dispose()
        {
            _session.Dispose();
            _sessionOptions.Dispose();
        }
    }
}
