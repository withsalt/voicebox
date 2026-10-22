using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 自回归 audio frame 生成主循环。
    /// 与 Python `OrtCpuRuntime.generate_audio_frames` 1:1 对齐:
    ///   1) prefill 跑完整 prompt → 得到 global_hidden + KV cache;
    ///   2) 循环 max_new_frames 步, 每步:
    ///      - 通过 local 子图采样得到一帧 [n_vq] 个 codebook 索引;
    ///      - 把这帧塞回 decode_step, 更新 KV cache 并刷新 global_hidden;
    ///   3) 命中 audio_end_token / 子图判定终止时退出。
    ///
    /// </summary>
    internal sealed class AudioFrameGenerator
    {
        private readonly InferenceEngine _engine;
        private readonly ModelBundle _bundle;

        // ---- 复用 buffer (生成器实例字段, 主循环热路径) ----
        private float[] _hiddenBuffer = Array.Empty<float>();
        private readonly KvCacheStore _kvCache = new();
        private readonly KvCacheStore _localKvCache = new();
        private int[] _nextRowBuffer = Array.Empty<int>();
        private readonly int[] _pastValidBuffer = new int[1];

        // decode_step 的 input OrtValue 列表 (复用 List<OrtValue>, 避免每帧 new)
        private readonly List<OrtValue> _decodeDisposable = new();

        // fixed_sampled_frame 模式下的 seen mask (增量更新)
        private int[] _seenMask = Array.Empty<int>();
        private int _seenMaskNVq;
        private int _seenMaskCodebookSize;

        public AudioFrameGenerator(InferenceEngine engine)
        {
            _engine = engine;
            _bundle = engine.Bundle;
        }

        /// <summary>
        /// 调用方负责传入 generation 参数 (manifest 默认 + TtsSynthesisOptions 覆盖)。
        /// </summary>
        public List<int[]> Generate(
            VoiceCloneRequestRows requestRows,
            GenerationParams gen,
            OnFrameCallback? onFrame,
            CancellationToken ct)
        {
            int nVq = _bundle.Manifest.TtsConfig.NVq;
            int rowWidth = nVq + 1;
            int audioPad = _bundle.Manifest.TtsConfig.AudioPadTokenId;
            int audioAssistantSlot = _bundle.Manifest.TtsConfig.AudioAssistantSlotTokenId;
            int hiddenSize = _bundle.TtsMeta.ModelConfig.HiddenSize;

            // ---- Prefill ----
            int rows = requestRows.InputIds.Length;
            int[] prefillInputIds = Flatten3D(requestRows.InputIds, rowWidth);
            int[] prefillMask = (int[])requestRows.AttentionMask.Clone();

            EnsureHiddenBuffer(hiddenSize);
            _kvCache.Clear();

            using (var inputIdsOrt = OrtValue.CreateTensorValueFromMemory(prefillInputIds, new long[] { 1, rows, rowWidth }))
            using (var maskOrt = OrtValue.CreateTensorValueFromMemory(prefillMask, new long[] { 1, rows }))
            {
                var inputValues = new OrtValue[] { inputIdsOrt, maskOrt };
                var inputNames = new[] { "input_ids", "attention_mask" };
                using var outputs = _bundle.Prefill.Run(_engine.RunOptions, inputNames, inputValues, _bundle.PrefillOutputNames);

                ExtractLastHiddenInto(outputs[0], hiddenSize, _hiddenBuffer);
                for (int i = 1; i < _bundle.PrefillOutputNames.Length; i++)
                {
                    string outName = _bundle.PrefillOutputNames[i];
                    string pastName = outName.Replace("present_", "past_");
                    _kvCache.StoreFrom(pastName, outputs[i]);
                }
            }

            int pastValidLength = 0;
            for (int i = 0; i < requestRows.AttentionMask.Length; i++) pastValidLength += requestRows.AttentionMask[i];

            // ---- Generation loop ----
            var generatedFrames = new List<int[]>(gen.MaxNewFrames);
            var previousTokensByChannel = new List<List<int>>(nVq);
            var previousTokenSetsByChannel = new List<HashSet<int>>(nVq);
            for (int c = 0; c < nVq; c++)
            {
                previousTokensByChannel.Add(new List<int>(gen.MaxNewFrames));
                previousTokenSetsByChannel.Add(new HashSet<int>());
            }

            // fixed 模式所需的 seen mask 在循环外初始化, 后续按帧增量更新。
            ResetSeenMaskIfNeeded(nVq, _bundle.TtsMeta.ModelConfig.AudioCodebookSizes[0]);

            // decode_step 的 nextRow buffer (固定大小, 一次分配)
            EnsureNextRowBuffer(rowWidth);

            for (int step = 0; step < gen.MaxNewFrames; step++)
            {
                ct.ThrowIfCancellationRequested();

                int[]? frame;

                if (gen.SampleMode == MossNanoSampleMode.Fixed && _bundle.LocalFixedSampledFrame is not null)
                {
                    var (shouldContinue, sampledFrame) = _engine.RunLocalFixedSampledFrame(_hiddenBuffer, _seenMask, nVq);
                    if (!shouldContinue) break;
                    frame = sampledFrame;
                    int codebookSize = _seenMaskCodebookSize;
                    for (int c = 0; c < frame.Length; c++)
                    {
                        int tokenId = frame[c];
                        previousTokensByChannel[c].Add(tokenId);
                        previousTokenSetsByChannel[c].Add(tokenId);
                        if ((uint)tokenId < (uint)codebookSize)
                            _seenMask[c * codebookSize + tokenId] = 1;
                    }
                }
                else if (_bundle.LocalCachedStep is not null && gen.SampleMode == MossNanoSampleMode.Full)
                {
                    frame = SampleFrameViaCachedStep(_hiddenBuffer, previousTokensByChannel, previousTokenSetsByChannel, gen, out bool stop);
                    if (stop) break;
                }
                else
                {
                    frame = SampleFrameViaLocalDecoder(_hiddenBuffer, previousTokensByChannel, previousTokenSetsByChannel, gen, out bool stop);
                    if (stop) break;
                }

                generatedFrames.Add(frame);

                // ---- Decode step: 把刚生成的一帧塞回主网络, 推一步 ----
                for (int j = 0; j < rowWidth; j++) _nextRowBuffer[j] = audioPad;
                _nextRowBuffer[0] = audioAssistantSlot;
                for (int i = 0; i < frame.Length; i++) _nextRowBuffer[i + 1] = frame[i];

                _pastValidBuffer[0] = pastValidLength;

                _decodeDisposable.Clear();
                using var nextRowOrt = OrtValue.CreateTensorValueFromMemory(_nextRowBuffer, new long[] { 1, 1, rowWidth });
                using var pastValidOrt = OrtValue.CreateTensorValueFromMemory(_pastValidBuffer, new long[] { 1 });

                try
                {
                    int decodeInputCount = _bundle.DecodeInputNames.Length;
                    var decodeInputs = _engine.GetDecodeInputArray(decodeInputCount);
                    decodeInputs[0] = nextRowOrt;
                    decodeInputs[1] = pastValidOrt;
                    for (int i = 2; i < decodeInputCount; i++)
                    {
                        string name = _bundle.DecodeInputNames[i];
                        decodeInputs[i] = _kvCache.CreateInputView(name);
                        _decodeDisposable.Add(decodeInputs[i]);
                    }

                    using var decodeOutputs = _bundle.Decode.Run(
                        _engine.RunOptions,
                        _bundle.DecodeInputNames,
                        decodeInputs,
                        _bundle.DecodeOutputNames);

                    ExtractLastHiddenInto(decodeOutputs[0], hiddenSize, _hiddenBuffer);
                    pastValidLength += 1;

                    for (int i = 1; i < _bundle.DecodeOutputNames.Length; i++)
                    {
                        string outName = _bundle.DecodeOutputNames[i];
                        string pastName = outName.Replace("present_", "past_");
                        _kvCache.StoreFrom(pastName, decodeOutputs[i]);
                    }
                }
                finally
                {
                    foreach (var d in _decodeDisposable) d.Dispose();
                    _decodeDisposable.Clear();
                }

                onFrame?.Invoke(generatedFrames, step, frame);
            }

            return generatedFrames;
        }

        /// <summary>
        /// 走 local_decoder 子图 (每个 channel 单独 run, 文本判定 + 音频采样)。
        /// 对应 Python 主循环最后的 else 分支。
        /// </summary>
        private int[] SampleFrameViaLocalDecoder(
            float[] globalHidden,
            List<List<int>> previousTokensByChannel,
            List<HashSet<int>> previousTokenSetsByChannel,
            GenerationParams gen,
            out bool stop)
        {
            int nVq = _bundle.Manifest.TtsConfig.NVq;
            int audioAssistantSlot = _bundle.Manifest.TtsConfig.AudioAssistantSlotTokenId;
            int audioEnd = _bundle.Manifest.TtsConfig.AudioEndTokenId;

            // 第一次空 frame run, 判定 text branch
            var (textLogits, _, _) = _engine.RunLocalDecoder(globalHidden, 0, Array.Empty<int>());
            int nextTextToken = Sampler.SampleAssistantTextToken(
                textLogits, audioAssistantSlot, audioEnd,
                gen.DoSample, gen.TextTemperature, gen.TextTopK, gen.TextTopP, _engine.Rng, _engine.SamplerWorkspace);
            if (nextTextToken != audioAssistantSlot) { stop = true; return Array.Empty<int>(); }

            var frame = new int[nVq];
            var framePrefix = new List<int>(nVq);
            for (int c = 0; c < nVq; c++)
            {
                var (_, audioLogits, perChannel) = _engine.RunLocalDecoder(globalHidden, nextTextToken, framePrefix);
                var channelLogits = SliceAudioChannelLogits(audioLogits, perChannel, c);
                int sampled = Sampler.SampleAudioToken(
                    channelLogits, previousTokenSetsByChannel[c],
                    gen.DoSample, gen.AudioTemperature, gen.AudioTopK, gen.AudioTopP, gen.AudioRepetitionPenalty,
                    _engine.Rng, _engine.SamplerWorkspace);
                frame[c] = sampled;
                framePrefix.Add(sampled);
                previousTokensByChannel[c].Add(sampled);
                previousTokenSetsByChannel[c].Add(sampled);
            }

            stop = false;
            return frame;
        }

        /// <summary>
        /// 走 local_cached_step 子图: 每帧 step_type=0 判文本, =1 出第 0 通道, =2 滚动出后续通道。
        /// 对应 Python 主循环中间分支。
        /// </summary>
        private int[] SampleFrameViaCachedStep(
            float[] globalHidden,
            List<List<int>> previousTokensByChannel,
            List<HashSet<int>> previousTokenSetsByChannel,
            GenerationParams gen,
            out bool stop)
        {
            int nVq = _bundle.Manifest.TtsConfig.NVq;
            int audioAssistantSlot = _bundle.Manifest.TtsConfig.AudioAssistantSlotTokenId;
            int audioEnd = _bundle.Manifest.TtsConfig.AudioEndTokenId;

            _localKvCache.Clear();
            InitEmptyLocalCachedPast(_localKvCache);
            int localPastValidLength = 0;

            var (textLogits, _, _) = RunLocalCachedStepFull(globalHidden, 0, 0, 0, 0, localPastValidLength);
            localPastValidLength++;
            int nextTextToken = Sampler.SampleAssistantTextToken(
                textLogits, audioAssistantSlot, audioEnd,
                gen.DoSample, gen.TextTemperature, gen.TextTopK, gen.TextTopP, _engine.Rng, _engine.SamplerWorkspace);
            if (nextTextToken != audioAssistantSlot) { stop = true; return Array.Empty<int>(); }

            var frame = new int[nVq];
            var (_, audioLogits0, perChannel0) = RunLocalCachedStepFull(globalHidden, nextTextToken, 0, 0, 1, localPastValidLength);
            localPastValidLength++;
            var firstChannel = SliceAudioChannelLogits(audioLogits0, perChannel0, 0);
            int sampled = Sampler.SampleAudioToken(
                firstChannel, previousTokenSetsByChannel[0],
                gen.DoSample, gen.AudioTemperature, gen.AudioTopK, gen.AudioTopP, gen.AudioRepetitionPenalty,
                _engine.Rng, _engine.SamplerWorkspace);
            frame[0] = sampled;
            previousTokensByChannel[0].Add(sampled);
            previousTokenSetsByChannel[0].Add(sampled);
            int previousToken = sampled;

            for (int c = 1; c < nVq; c++)
            {
                var (_, audioLogitsC, perChannelC) = RunLocalCachedStepFull(globalHidden, 0, previousToken, c - 1, 2, localPastValidLength);
                localPastValidLength++;
                var channelLogits = SliceAudioChannelLogits(audioLogitsC, perChannelC, c);
                sampled = Sampler.SampleAudioToken(
                    channelLogits, previousTokenSetsByChannel[c],
                    gen.DoSample, gen.AudioTemperature, gen.AudioTopK, gen.AudioTopP, gen.AudioRepetitionPenalty,
                    _engine.Rng, _engine.SamplerWorkspace);
                frame[c] = sampled;
                previousTokensByChannel[c].Add(sampled);
                previousTokenSetsByChannel[c].Add(sampled);
                previousToken = sampled;
            }
            stop = false;
            return frame;
        }

        private void InitEmptyLocalCachedPast(KvCacheStore store)
        {
            int localLayers = _bundle.TtsMeta.ModelConfig.LocalLayers;
            int localHeads = _bundle.TtsMeta.ModelConfig.LocalHeads;
            int localHeadDim = _bundle.TtsMeta.ModelConfig.LocalHeadDim;
            for (int l = 0; l < localLayers; l++)
            {
                long[] shape = { 1, 0, localHeads, localHeadDim };
                store.SetEmpty($"local_past_key_{l}", shape);
                store.SetEmpty($"local_past_value_{l}", shape);
            }
        }

        private (float[] textLogits, float[] audioLogits, int audioPerChannel)
            RunLocalCachedStepFull(
                float[] globalHidden,
                int textTokenId,
                int audioTokenId,
                int channelIndex,
                int stepType,
                int pastValidLength)
        {
            if (_bundle.LocalCachedStep is null)
                throw new InvalidOperationException("local_cached_step session 不存在。");

            int hiddenSize = _bundle.TtsMeta.ModelConfig.HiddenSize;

            var disposable = new List<OrtValue>();
            try
            {
                var hiddenOrt = OrtValue.CreateTensorValueFromMemory(globalHidden, new long[] { 1, hiddenSize });
                disposable.Add(hiddenOrt);
                var textTokenOrt = OrtValue.CreateTensorValueFromMemory(new int[] { textTokenId }, new long[] { 1 });
                disposable.Add(textTokenOrt);
                var audioTokenOrt = OrtValue.CreateTensorValueFromMemory(new int[] { audioTokenId }, new long[] { 1 });
                disposable.Add(audioTokenOrt);
                var channelOrt = OrtValue.CreateTensorValueFromMemory(new int[] { channelIndex }, new long[] { 1 });
                disposable.Add(channelOrt);
                var stepTypeOrt = OrtValue.CreateTensorValueFromMemory(new int[] { stepType }, new long[] { 1 });
                disposable.Add(stepTypeOrt);
                var pastLenOrt = OrtValue.CreateTensorValueFromMemory(new int[] { pastValidLength }, new long[] { 1 });
                disposable.Add(pastLenOrt);

                var inputNames = new List<string> { "global_hidden", "text_token_id", "audio_token_id", "channel_index", "step_type", "past_valid_lengths" };
                var inputs = new List<OrtValue> { hiddenOrt, textTokenOrt, audioTokenOrt, channelOrt, stepTypeOrt, pastLenOrt };

                foreach (var name in _localKvCache.Names)
                {
                    var ov = _localKvCache.CreateInputView(name);
                    disposable.Add(ov);
                    inputNames.Add(name);
                    inputs.Add(ov);
                }

                using var outputs = _bundle.LocalCachedStep.Run(
                    _engine.RunOptions, inputNames, inputs, _bundle.LocalCachedOutputNames);

                var textLogits = outputs[0].GetTensorDataAsSpan<float>().ToArray();
                var audioLogitsShape = outputs[1].GetTensorTypeAndShape().Shape;
                int perChannel = (int)audioLogitsShape[^1];
                var audioLogits = outputs[1].GetTensorDataAsSpan<float>().ToArray();

                for (int i = 2; i < _bundle.LocalCachedOutputNames.Length; i++)
                {
                    string outName = _bundle.LocalCachedOutputNames[i];
                    string nextName = outName.Replace("local_present_", "local_past_");
                    _localKvCache.StoreFrom(nextName, outputs[i]);
                }
                return (textLogits, audioLogits, perChannel);
            }
            finally
            {
                foreach (var d in disposable) d.Dispose();
            }
        }

        // ---------------- Buffer management ----------------

        private void EnsureHiddenBuffer(int hiddenSize)
        {
            if (_hiddenBuffer.Length < hiddenSize) _hiddenBuffer = new float[hiddenSize];
        }

        private void EnsureNextRowBuffer(int rowWidth)
        {
            if (_nextRowBuffer.Length < rowWidth) _nextRowBuffer = new int[rowWidth];
        }

        private void ResetSeenMaskIfNeeded(int nVq, int audioCodebookSize)
        {
            int needed = nVq * audioCodebookSize;
            if (_seenMask.Length < needed
                || _seenMaskNVq != nVq
                || _seenMaskCodebookSize != audioCodebookSize)
            {
                _seenMask = new int[needed];
                _seenMaskNVq = nVq;
                _seenMaskCodebookSize = audioCodebookSize;
            }
            else
            {
                Array.Clear(_seenMask, 0, needed);
            }
        }

        // ---------------- Tensor utilities ----------------

        private static float[] SliceAudioChannelLogits(float[] audioLogits, int perChannel, int channelIndex)
        {
            var output = new float[perChannel];
            int offset = channelIndex * perChannel;
            Array.Copy(audioLogits, offset, output, 0, perChannel);
            return output;
        }

        /// <summary>
        /// 把 global_hidden ([1, T, H] 或 [1, H]) 拍平为 [H] 的最后一行, 写入 <paramref name="dest"/>。
        /// </summary>
        private static void ExtractLastHiddenInto(OrtValue hiddenStates, int hiddenSize, float[] dest)
        {
            var shape = hiddenStates.GetTensorTypeAndShape().Shape;
            var span = hiddenStates.GetTensorDataAsSpan<float>();
            if (shape.Length == 2)
            {
                span.Slice(span.Length - hiddenSize, hiddenSize).CopyTo(dest);
                return;
            }
            if (shape.Length != 3 || shape[0] != 1)
                throw new InvalidDataException($"Unexpected global_hidden shape: [{string.Join(",", shape)}]");
            int last = (int)shape[1] - 1;
            int H = (int)shape[2];
            int offset = last * H;
            span.Slice(offset, H).CopyTo(dest);
        }

        /// <summary>
        /// 拍平 [rows][rowWidth] 为 [rows * rowWidth] 的一维 int 数组。
        /// </summary>
        private static int[] Flatten3D(int[][] rows, int rowWidth)
        {
            int n = rows.Length;
            var data = new int[n * rowWidth];
            for (int i = 0; i < n; i++)
            {
                var row = rows[i];
                for (int j = 0; j < rowWidth; j++) data[i * rowWidth + j] = j < row.Length ? row[j] : 0;
            }
            return data;
        }
    }

    /// <summary>
    /// 自回归 KV cache 复用容器: 每个命名 entry 持有一段 float[] 与 shape long[],
    /// 跨 step 复用同一份 buffer, 仅在容量不足时增长。
    /// 不是线程安全的; 上层串行化。
    /// </summary>
    internal sealed class KvCacheStore
    {
        private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);

        public IEnumerable<string> Names => _entries.Keys;

        private sealed class Entry
        {
            public float[] Buffer = Array.Empty<float>();
            public long[] Shape = Array.Empty<long>();
            public int ValidLength;
        }

        public void Clear() => _entries.Clear();

        public void StoreFrom(string name, OrtValue source)
        {
            var srcShape = source.GetTensorTypeAndShape().Shape;
            var span = source.GetTensorDataAsSpan<float>();
            int len = span.Length;

            if (!_entries.TryGetValue(name, out var entry))
            {
                entry = new Entry();
                _entries[name] = entry;
            }
            if (entry.Buffer.Length < len)
            {
                int newCap = Math.Max(len, entry.Buffer.Length + (entry.Buffer.Length >> 2) + 16);
                entry.Buffer = new float[newCap];
            }
            if (entry.Shape.Length != srcShape.Length) entry.Shape = new long[srcShape.Length];
            for (int i = 0; i < srcShape.Length; i++) entry.Shape[i] = srcShape[i];
            span.CopyTo(entry.Buffer);
            entry.ValidLength = len;
        }

        /// <summary>
        /// 创建一个 size=0 的占位 entry (用于 local cached step 初始空 cache)。
        /// </summary>
        public void SetEmpty(string name, long[] shape)
        {
            if (!_entries.TryGetValue(name, out var entry))
            {
                entry = new Entry();
                _entries[name] = entry;
            }
            // 给一个最小 buffer (ORT 不接受 0 长度内存, 留 1 个元素, 实际不会被读取因 shape 中存在 0 维)。
            if (entry.Buffer.Length < 1) entry.Buffer = new float[1];
            if (entry.Shape.Length != shape.Length) entry.Shape = new long[shape.Length];
            for (int i = 0; i < shape.Length; i++) entry.Shape[i] = shape[i];
            entry.ValidLength = 0;
        }

        public OrtValue CreateInputView(string name)
        {
            var entry = _entries[name];
            // 长度为 0 时 (空 cache), 用 buffer 整体长度构造但 shape 中包含 0 维, 让 ORT 视为空 tensor。
            int viewLen = entry.ValidLength == 0 ? entry.Buffer.Length : entry.ValidLength;
            return OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(entry.Buffer, 0, viewLen),
                entry.Shape);
        }
    }
}
