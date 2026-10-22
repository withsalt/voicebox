using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 流式 codec 解码器。维护 transformer offset / attention KV cache 等内部状态, 让连续解码可以走 decode_step 子图。
    /// 与 Python `CodecStreamingDecodeSession` 1:1 对齐。
    /// 不是线程安全的; 上层通过 SemaphoreSlim 串行化即可。
    /// </summary>
    internal sealed class CodecStreamingDecodeSession : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly CodecMeta _codecMeta;
        private readonly RunOptions _runOptions = new();

        // 状态以原始数组 + shape 形式持有, 每次 Run 时临时构造 OrtValue, 避免跨调用持有原生指针的生命周期问题。
        private readonly Dictionary<string, IntStateBuffer> _intState = new(StringComparer.Ordinal);
        private readonly Dictionary<string, FloatStateBuffer> _floatState = new(StringComparer.Ordinal);

        // 输入名称序列。固定顺序: audio_codes, audio_code_lengths, transformer_offset_*, attn_offset_/keys_/values_/positions_*
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;

        public CodecStreamingDecodeSession(InferenceSession session, CodecMeta codecMeta)
        {
            _session = session;
            _codecMeta = codecMeta;

            var allInputs = new List<string> { "audio_codes", "audio_code_lengths" };
            foreach (var spec in _codecMeta.StreamingDecode.TransformerOffsets)
                allInputs.Add(spec.InputName);
            foreach (var spec in _codecMeta.StreamingDecode.AttentionCaches)
            {
                allInputs.Add(spec.OffsetInputName);
                allInputs.Add(spec.CachedKeysInputName);
                allInputs.Add(spec.CachedValuesInputName);
                allInputs.Add(spec.CachedPositionsInputName);
            }
            _inputNames = allInputs.ToArray();
            _outputNames = _session.OutputNames.ToArray();

            Reset();
        }

        public void Reset()
        {
            _intState.Clear();
            _floatState.Clear();

            foreach (var spec in _codecMeta.StreamingDecode.TransformerOffsets)
            {
                long[] shape = spec.Shape.Select(x => (long)x).ToArray();
                _intState[spec.InputName] = new IntStateBuffer(new int[ShapeProduct(shape)], shape);
            }
            foreach (var spec in _codecMeta.StreamingDecode.AttentionCaches)
            {
                long[] offsetShape = spec.OffsetShape.Select(x => (long)x).ToArray();
                long[] cacheShape = spec.CacheShape.Select(x => (long)x).ToArray();
                long[] positionsShape = spec.PositionsShape.Select(x => (long)x).ToArray();

                _intState[spec.OffsetInputName] = new IntStateBuffer(new int[ShapeProduct(offsetShape)], offsetShape);
                _floatState[spec.CachedKeysInputName] = new FloatStateBuffer(new float[ShapeProduct(cacheShape)], cacheShape);
                _floatState[spec.CachedValuesInputName] = new FloatStateBuffer(new float[ShapeProduct(cacheShape)], cacheShape);
                int[] positions = new int[ShapeProduct(positionsShape)];
                Array.Fill(positions, -1);
                _intState[spec.CachedPositionsInputName] = new IntStateBuffer(positions, positionsShape);
            }
        }

        /// <summary>
        /// 喂入若干 audio frames, 解码出本段 PCM。返回值 audio 为 channel-major [channels, totalSamples];
        /// 调用方应取前 audioLength 个样本。
        /// </summary>
        public StreamingDecodeOutput? RunFrames(IReadOnlyList<int[]> frameRows)
        {
            if (frameRows.Count == 0) return null;

            int numQuantizers = _codecMeta.CodecConfig.NumQuantizers;
            int frameCount = frameRows.Count;
            var audioCodes = new int[1 * frameCount * numQuantizers];
            for (int f = 0; f < frameCount; f++)
            {
                int[] row = frameRows[f];
                int rowLen = row.Length;
                for (int q = 0; q < numQuantizers; q++)
                    audioCodes[f * numQuantizers + q] = q < rowLen ? row[q] : 0;
            }
            int[] codeLengths = { frameCount };

            var disposables = new List<OrtValue>(_inputNames.Length);
            try
            {
                var inputValues = new OrtValue[_inputNames.Length];
                inputValues[0] = OrtValue.CreateTensorValueFromMemory(audioCodes, new long[] { 1, frameCount, numQuantizers });
                disposables.Add(inputValues[0]);
                inputValues[1] = OrtValue.CreateTensorValueFromMemory(codeLengths, new long[] { 1 });
                disposables.Add(inputValues[1]);

                for (int i = 2; i < _inputNames.Length; i++)
                {
                    string name = _inputNames[i];
                    if (_intState.TryGetValue(name, out var ib))
                    {
                        var v = OrtValue.CreateTensorValueFromMemory(ib.Data, ib.Shape);
                        inputValues[i] = v; disposables.Add(v);
                    }
                    else if (_floatState.TryGetValue(name, out var fb))
                    {
                        var v = OrtValue.CreateTensorValueFromMemory(fb.Data, fb.Shape);
                        inputValues[i] = v; disposables.Add(v);
                    }
                    else
                    {
                        throw new InvalidOperationException($"streaming decode 状态缺失: {name}");
                    }
                }

                using var outputs = _session.Run(_runOptions, _inputNames, inputValues, _outputNames);

                var byName = new Dictionary<string, OrtValue>(_outputNames.Length, StringComparer.Ordinal);
                for (int i = 0; i < _outputNames.Length; i++) byName[_outputNames[i]] = outputs[i];

                var audioOrt = byName["audio"];
                var audioShape = audioOrt.GetTensorTypeAndShape().Shape;
                int channels = (int)audioShape[1];
                int totalSamples = (int)audioShape[2];
                var audioCopy = new float[channels * totalSamples];
                audioOrt.GetTensorDataAsSpan<float>().CopyTo(audioCopy);
                int audioLength = byName["audio_lengths"].GetTensorDataAsSpan<int>()[0];

                // 把 outputs 中的状态拷贝到状态字典 (outputs 在本函数返回后会失效)。
                foreach (var spec in _codecMeta.StreamingDecode.TransformerOffsets)
                    UpdateIntState(spec.InputName, byName[spec.OutputName]);
                foreach (var spec in _codecMeta.StreamingDecode.AttentionCaches)
                {
                    UpdateIntState(spec.OffsetInputName, byName[spec.OffsetOutputName]);
                    UpdateFloatState(spec.CachedKeysInputName, byName[spec.CachedKeysOutputName]);
                    UpdateFloatState(spec.CachedValuesInputName, byName[spec.CachedValuesOutputName]);
                    UpdateIntState(spec.CachedPositionsInputName, byName[spec.CachedPositionsOutputName]);
                }

                return new StreamingDecodeOutput(audioCopy, channels, totalSamples, audioLength);
            }
            finally
            {
                foreach (var d in disposables) d.Dispose();
            }
        }

        private void UpdateIntState(string name, OrtValue source)
        {
            var shape = source.GetTensorTypeAndShape().Shape;
            var data = source.GetTensorDataAsSpan<int>();
            var copy = new int[data.Length];
            data.CopyTo(copy);
            _intState[name] = new IntStateBuffer(copy, shape);
        }

        private void UpdateFloatState(string name, OrtValue source)
        {
            var shape = source.GetTensorTypeAndShape().Shape;
            var data = source.GetTensorDataAsSpan<float>();
            var copy = new float[data.Length];
            data.CopyTo(copy);
            _floatState[name] = new FloatStateBuffer(copy, shape);
        }

        private static int ShapeProduct(long[] shape)
        {
            int total = 1;
            foreach (long d in shape) total *= (int)d;
            return total;
        }

        public void Dispose()
        {
            _runOptions.Dispose();
            _intState.Clear();
            _floatState.Clear();
        }

        private sealed record IntStateBuffer(int[] Data, long[] Shape);
        private sealed record FloatStateBuffer(float[] Data, long[] Shape);
    }

    /// <summary>
    /// 单次 decode_step 输出。audio 为 channel-major [channels, totalSamples]; 实际有效长度由 audioLength 标识。
    /// </summary>
    internal sealed class StreamingDecodeOutput
    {
        public StreamingDecodeOutput(float[] audio, int channels, int totalSamples, int audioLength)
        {
            Audio = audio;
            Channels = channels;
            TotalSamples = totalSamples;
            AudioLength = audioLength;
        }

        public float[] Audio { get; }
        public int Channels { get; }
        public int TotalSamples { get; }
        public int AudioLength { get; }
    }
}
