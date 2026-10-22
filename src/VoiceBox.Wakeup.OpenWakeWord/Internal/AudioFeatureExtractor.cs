using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using VoiceBox.Core.Onnx;

namespace VoiceBox.Wakeup.OpenWakeWord.Internal
{
    /// <summary>
    /// 复刻自 Python 版 <c>openwakeword.utils.AudioFeatures</c> 的流式音频特征提取器。
    /// 维护原始音频缓冲、mel 频谱缓冲、以及由 Google speech_embedding 模型计算得到的
    /// 96 维 embedding 特征缓冲。所有唤醒词模型共享同一份 embedding 序列, 因此特征只需
    /// 计算一次。
    /// </summary>
    internal sealed class AudioFeatureExtractor : IDisposable
    {
        // 与原版一致的固定常量, 不暴露为配置。
        private const int ChunkSamples = 1280;          // 80ms @ 16kHz
        private const int MelHopOverlapSamples = 480;   // 160 * 3, 流式 melspec 的窗口重叠样本
        private const int MelWindowFrames = 76;         // embedding 模型输入的 mel 帧窗口大小
        private const int MelFrameStride = 8;           // embedding 模型每次推进 8 帧 (~80ms)
        private const int MelBins = 32;                 // melspec 输出特征维度
        private const int EmbeddingDim = 96;            // embedding 模型输出维度
        private const int MaxMelBufferFrames = 10 * 97; // 约 10 秒 mel 历史
        private const int MaxFeatureBufferFrames = 120; // 约 10 秒 embedding 历史 (远超任何模型需要)
        private const int RawDataMaxSamples = 16000 * 10; // 10 秒 raw 历史 (容错)

        private readonly InferenceSession _melSession;
        private readonly InferenceSession _embeddingSession;
        private readonly SessionOptions _melSessionOptions;
        private readonly SessionOptions _embeddingSessionOptions;

        private readonly string _melInputName;
        private readonly string _embeddingInputName;
        private readonly string _melOutputName;
        private readonly string _embeddingOutputName;

        // raw audio buffer (int16 振幅但存储为 float32, 与原版一致)
        private readonly float[] _rawDataBuffer;
        private int _rawDataCount;

        // melspec buffer: 行优先, 每行 32 个 float, 共 _melBufferFrames 行有效
        private readonly float[] _melBuffer;
        private int _melBufferFrames;

        // embedding feature buffer: 每行 96 个 float
        private readonly float[] _featureBuffer;
        private int _featureBufferFrames;

        // 未凑齐 ChunkSamples 的余量
        private readonly float[] _accumulator;
        private int _accumulatorCount;

        public int EmbeddingFrames => _featureBufferFrames;

        public AudioFeatureExtractor(
            string melModelPath,
            string embeddingModelPath,
            OnnxExecutionOptions onnxOptions)
        {
            if (!File.Exists(melModelPath))
                throw new FileNotFoundException($"Melspectrogram model not found: {melModelPath}", melModelPath);
            if (!File.Exists(embeddingModelPath))
                throw new FileNotFoundException($"Embedding model not found: {embeddingModelPath}", embeddingModelPath);

            _melSessionOptions = OnnxSessionFactory.Create(onnxOptions);
            _embeddingSessionOptions = OnnxSessionFactory.Create(onnxOptions);
            _melSession = new InferenceSession(melModelPath, _melSessionOptions);
            _embeddingSession = new InferenceSession(embeddingModelPath, _embeddingSessionOptions);

            _melInputName = _melSession.InputMetadata.Keys.First();
            _melOutputName = _melSession.OutputMetadata.Keys.First();
            _embeddingInputName = _embeddingSession.InputMetadata.Keys.First();
            _embeddingOutputName = _embeddingSession.OutputMetadata.Keys.First();

            _rawDataBuffer = new float[RawDataMaxSamples];
            _melBuffer = new float[MaxMelBufferFrames * MelBins];
            _featureBuffer = new float[MaxFeatureBufferFrames * EmbeddingDim];
            _accumulator = new float[ChunkSamples];

            Reset();
        }

        /// <summary>
        /// 复位所有内部状态。等价于 Python 版 <c>reset()</c>:
        /// melspec buffer 初始化为全 1, feature buffer 通过随机噪声预热到容量上限。
        /// </summary>
        public void Reset()
        {
            _rawDataCount = 0;
            _accumulatorCount = 0;

            Array.Clear(_melBuffer, 0, _melBuffer.Length);
            for (int i = 0; i < MelWindowFrames * MelBins; i++)
                _melBuffer[i] = 1f;
            _melBufferFrames = MelWindowFrames;

            // 用噪声预热 feature buffer (与 Python 版完全等价: 4 秒随机 int16 噪声生成的 embeddings)
            Array.Clear(_featureBuffer, 0, _featureBuffer.Length);
            _featureBufferFrames = 0;
            WarmUpFeatureBuffer();
        }

        private void WarmUpFeatureBuffer()
        {
            const int warmSamples = 16000 * 4;
            float[] noise = new float[warmSamples];
            var rng = new Random(0);
            for (int i = 0; i < warmSamples; i++)
            {
                // 与 Python np.random.randint(-1000, 1000) 等价的区间, 数值是 int16 振幅尺度
                noise[i] = rng.Next(-1000, 1000);
            }

            float[] spec = ComputeMelspectrogram(noise, 0, warmSamples);
            int frames = spec.Length / MelBins;
            // 滑动窗口生成 embedding
            int windowCount = (frames - MelWindowFrames) / MelFrameStride + 1;
            if (windowCount <= 0) return;

            var batchInput = new float[windowCount * MelWindowFrames * MelBins];
            for (int w = 0; w < windowCount; w++)
            {
                int start = w * MelFrameStride;
                Array.Copy(spec, start * MelBins, batchInput, w * MelWindowFrames * MelBins, MelWindowFrames * MelBins);
            }

            float[] embeddings = RunEmbeddingBatch(batchInput, windowCount);

            // 写入 feature buffer (尾对齐, 截断到容量)
            int rowsToWrite = Math.Min(windowCount, MaxFeatureBufferFrames);
            int srcStart = (windowCount - rowsToWrite) * EmbeddingDim;
            Array.Copy(embeddings, srcStart, _featureBuffer, 0, rowsToWrite * EmbeddingDim);
            _featureBufferFrames = rowsToWrite;
        }

        /// <summary>
        /// 流式输入新音频, 返回本次新增的 embedding 帧数 (每帧对应 80ms)。
        /// </summary>
        public int Process(ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty) return 0;

            int newFrames = 0;
            int i = 0;
            while (i < audio.Length)
            {
                int remainingInChunk = ChunkSamples - _accumulatorCount;
                int take = Math.Min(remainingInChunk, audio.Length - i);
                audio.Slice(i, take).CopyTo(_accumulator.AsSpan(_accumulatorCount));
                _accumulatorCount += take;
                i += take;

                if (_accumulatorCount == ChunkSamples)
                {
                    AppendRaw(_accumulator, 0, ChunkSamples);
                    _accumulatorCount = 0;
                    if (UpdateMelspectrogram())
                    {
                        if (AppendEmbeddingForLatestChunk())
                            newFrames++;
                    }
                }
            }
            return newFrames;
        }

        /// <summary>
        /// 拷贝最近 N 帧 embedding 到目标缓冲 (形状 [N, 96], 行优先)。
        /// 若可用帧数不足返回 false。
        /// </summary>
        public bool TryCopyLatestFeatures(int n, Span<float> destination)
        {
            if (n <= 0 || destination.Length < n * EmbeddingDim) return false;
            if (_featureBufferFrames < n) return false;

            int srcStart = (_featureBufferFrames - n) * EmbeddingDim;
            _featureBuffer.AsSpan(srcStart, n * EmbeddingDim).CopyTo(destination);
            return true;
        }

        private void AppendRaw(float[] src, int srcOffset, int count)
        {
            if (_rawDataCount + count <= _rawDataBuffer.Length)
            {
                Array.Copy(src, srcOffset, _rawDataBuffer, _rawDataCount, count);
                _rawDataCount += count;
            }
            else
            {
                int keep = _rawDataBuffer.Length - count;
                if (keep > 0)
                {
                    Array.Copy(_rawDataBuffer, _rawDataCount - keep, _rawDataBuffer, 0, keep);
                    _rawDataCount = keep;
                }
                else
                {
                    _rawDataCount = 0;
                }
                Array.Copy(src, srcOffset, _rawDataBuffer, _rawDataCount, count);
                _rawDataCount += count;
            }
        }

        /// <summary>
        /// 计算最近一段 raw audio 的 mel 频谱并追加到 melspec buffer。
        /// 完全对齐 Python 流式版本: 取 _rawDataBuffer[-ChunkSamples-480:] 计算 melspec。
        /// 首次调用时只使用可用样本 (raw_data 不足 1760)。
        /// </summary>
        private bool UpdateMelspectrogram()
        {
            int wanted = ChunkSamples + MelHopOverlapSamples; // 1760
            int available = _rawDataCount;
            int useLen = Math.Min(wanted, available);
            if (useLen < 400) return false; // melspec 至少需要 25ms (400 样本)

            int srcStart = _rawDataCount - useLen;

            float[] newFrames = ComputeMelspectrogram(_rawDataBuffer, srcStart, useLen);
            int newFrameCount = newFrames.Length / MelBins;
            if (newFrameCount == 0) return false;

            AppendMelFrames(newFrames, newFrameCount);
            return true;
        }

        /// <summary>
        /// 运行 melspectrogram ONNX 模型, 对应 Python 中的 melspec_transform: <c>x / 10 + 2</c>。
        /// </summary>
        private float[] ComputeMelspectrogram(float[] src, int srcOffset, int count)
        {
            var tensor = new DenseTensor<float>(new[] { 1, count });
            var buffer = tensor.Buffer.Span;
            for (int j = 0; j < count; j++)
                buffer[j] = src[srcOffset + j];

            using var results = _melSession.Run(
                new[] { NamedOnnxValue.CreateFromTensor(_melInputName, tensor) });

            var outTensor = results.First(r => r.Name == _melOutputName).AsTensor<float>();
            float[] flat = outTensor.ToArray();
            // melspec 模型输出形状可能为 [1, 1, n_frames, 32] 或 [1, n_frames, 32], 总数应为 n_frames * 32
            int frames = flat.Length / MelBins;
            for (int j = 0; j < flat.Length; j++)
                flat[j] = flat[j] / 10f + 2f;
            return frames * MelBins == flat.Length ? flat : flat.AsSpan(0, frames * MelBins).ToArray();
        }

        private void AppendMelFrames(float[] frames, int frameCount)
        {
            // 容量满则把旧帧向左移动 (尾对齐)
            int totalAfter = _melBufferFrames + frameCount;
            int trim = totalAfter - MaxMelBufferFrames;
            if (trim > 0)
            {
                int keep = _melBufferFrames - trim;
                if (keep < 0) keep = 0;
                if (keep > 0)
                    Array.Copy(_melBuffer, trim * MelBins, _melBuffer, 0, keep * MelBins);
                _melBufferFrames = keep;
                totalAfter = _melBufferFrames + frameCount;
            }

            Array.Copy(frames, 0, _melBuffer, _melBufferFrames * MelBins, frameCount * MelBins);
            _melBufferFrames = totalAfter;
        }

        /// <summary>
        /// 取 mel buffer 末尾 76 帧, 计算一次 embedding 并写入 feature buffer。
        /// 若 mel buffer 帧数不足 76 则跳过。
        /// </summary>
        private bool AppendEmbeddingForLatestChunk()
        {
            if (_melBufferFrames < MelWindowFrames) return false;

            // 输入形状: [1, 76, 32, 1]
            int srcStart = (_melBufferFrames - MelWindowFrames) * MelBins;
            var tensor = new DenseTensor<float>(new[] { 1, MelWindowFrames, MelBins, 1 });
            var dst = tensor.Buffer.Span;
            _melBuffer.AsSpan(srcStart, MelWindowFrames * MelBins).CopyTo(dst);

            using var results = _embeddingSession.Run(
                new[] { NamedOnnxValue.CreateFromTensor(_embeddingInputName, tensor) });

            var outTensor = results.First(r => r.Name == _embeddingOutputName).AsTensor<float>();
            float[] flat = outTensor.ToArray();
            if (flat.Length < EmbeddingDim)
                throw new InvalidOperationException(
                    $"Embedding model returned unexpected length {flat.Length}, expected at least {EmbeddingDim}.");

            // embedding 模型输出形状为 [1, 1, 1, 96] 或 [1, 96], 取前 96 维即可
            AppendFeatureFrame(flat);
            return true;
        }

        private void AppendFeatureFrame(float[] embedding)
        {
            if (_featureBufferFrames >= MaxFeatureBufferFrames)
            {
                // 左移一格
                Array.Copy(_featureBuffer, EmbeddingDim, _featureBuffer, 0, (MaxFeatureBufferFrames - 1) * EmbeddingDim);
                _featureBufferFrames = MaxFeatureBufferFrames - 1;
            }
            Array.Copy(embedding, 0, _featureBuffer, _featureBufferFrames * EmbeddingDim, EmbeddingDim);
            _featureBufferFrames++;
        }

        private float[] RunEmbeddingBatch(float[] batchInput, int batchSize)
        {
            var tensor = new DenseTensor<float>(new[] { batchSize, MelWindowFrames, MelBins, 1 });
            batchInput.AsSpan(0, batchSize * MelWindowFrames * MelBins).CopyTo(tensor.Buffer.Span);

            using var results = _embeddingSession.Run(
                new[] { NamedOnnxValue.CreateFromTensor(_embeddingInputName, tensor) });

            var outTensor = results.First(r => r.Name == _embeddingOutputName).AsTensor<float>();
            float[] flat = outTensor.ToArray();
            int expected = batchSize * EmbeddingDim;
            if (flat.Length < expected)
                throw new InvalidOperationException(
                    $"Embedding model batch output length {flat.Length} < expected {expected}.");
            if (flat.Length == expected) return flat;
            // 模型可能返回 [batch, 1, 1, 96]; flat 总长是 expected
            var trimmed = new float[expected];
            Array.Copy(flat, trimmed, expected);
            return trimmed;
        }

        public void Dispose()
        {
            _melSession.Dispose();
            _embeddingSession.Dispose();
            _melSessionOptions.Dispose();
            _embeddingSessionOptions.Dispose();
        }
    }
}
