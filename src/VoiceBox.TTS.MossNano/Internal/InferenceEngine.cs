using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// MOSS-TTS-Nano 推理核心。把 token 序列翻译为 audio frames, 再把 audio frames 翻译为 PCM。
    ///
    /// 与 Python `OrtCpuRuntime` 中下列方法 1:1 对齐:
    ///   build_text_rows / build_audio_prefix_rows / build_voice_clone_request_rows
    ///   run_local_decoder / run_local_cached_step / run_local_fixed_sampled_frame
    ///   generate_audio_frames / decode_full_audio / encode_reference_audio
    /// </summary>
    internal sealed class InferenceEngine : IDisposable
    {
        private readonly ModelBundle _bundle;
        private readonly RunOptions _runOptions = new();
        private Random _rng;
        private readonly SamplerWorkspace _samplerWs = new();

        // decode_step 的输入数组复用 (大小 = DecodeInputNames.Length, 整个生命周期内不变)
        private OrtValue[]? _decodeInputArray;

        // local_fixed_sampled_frame 的小输入复用
        private readonly float[] _assistantUBuffer = new float[1];
        private float[] _audioUBuffer = Array.Empty<float>();

        public InferenceEngine(ModelBundle bundle, int? seed = null)
        {
            _bundle = bundle;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random(1234);
        }

        public Random Rng => _rng;

        public ModelBundle Bundle => _bundle;

        public SamplerWorkspace SamplerWorkspace => _samplerWs;

        public void ResetRandom(int seed) => _rng = new Random(seed);

        /// <summary>
        /// 复用一个 decode_step 输入 OrtValue[] 数组, 避免主循环每帧 new。
        /// 由调用方填充每个槽位。
        /// </summary>
        internal OrtValue[] GetDecodeInputArray(int size)
        {
            if (_decodeInputArray is null || _decodeInputArray.Length != size)
                _decodeInputArray = new OrtValue[size];
            return _decodeInputArray;
        }

        // ---------------- Prompt building ----------------

        /// <summary>
        /// 文本 token 行 (n_vq+1 列, 0 列为文本 token, 其它列填 audio_pad)。
        /// </summary>
        private int[][] BuildTextRows(IReadOnlyList<int> tokenIds)
        {
            int rowWidth = _bundle.Manifest.TtsConfig.NVq + 1;
            int audioPad = _bundle.Manifest.TtsConfig.AudioPadTokenId;
            var rows = new int[tokenIds.Count][];
            for (int i = 0; i < tokenIds.Count; i++)
            {
                var row = new int[rowWidth];
                for (int j = 0; j < rowWidth; j++) row[j] = audioPad;
                row[0] = tokenIds[i];
                rows[i] = row;
            }
            return rows;
        }

        /// <summary>
        /// 参考音频 prompt 行: 第 0 列填 audio_user_slot_token_id, 后续列填 codebook 索引。
        /// </summary>
        private int[][] BuildAudioPrefixRows(IReadOnlyList<int[]> promptAudioCodes, int? slotTokenId = null)
        {
            int nVq = _bundle.Manifest.TtsConfig.NVq;
            int rowWidth = nVq + 1;
            int audioPad = _bundle.Manifest.TtsConfig.AudioPadTokenId;
            int slot = slotTokenId ?? _bundle.Manifest.TtsConfig.AudioUserSlotTokenId;
            var rows = new int[promptAudioCodes.Count][];
            for (int i = 0; i < promptAudioCodes.Count; i++)
            {
                var src = promptAudioCodes[i];
                var row = new int[rowWidth];
                for (int j = 0; j < rowWidth; j++) row[j] = audioPad;
                row[0] = slot;
                int copy = Math.Min(src.Length, nVq);
                for (int j = 0; j < copy; j++) row[j + 1] = src[j];
                rows[i] = row;
            }
            return rows;
        }

        /// <summary>
        /// 拼接最终 prompt: [user_prefix][audio_start][prompt_audio_codes][audio_end][after_ref][text][assistant_prefix][audio_start]
        /// </summary>
        public VoiceCloneRequestRows BuildVoiceCloneRequestRows(IReadOnlyList<int[]> promptAudioCodes, IReadOnlyList<int> textTokenIds)
        {
            var t = _bundle.Manifest.TtsConfig;
            var prompts = _bundle.Manifest.PromptTemplates;

            var prefixTokens = new List<int>(prompts.UserPromptPrefixTokenIds.Count + 1);
            prefixTokens.AddRange(prompts.UserPromptPrefixTokenIds);
            prefixTokens.Add(t.AudioStartTokenId);

            var suffixTokens = new List<int>(
                1 + prompts.UserPromptAfterReferenceTokenIds.Count + textTokenIds.Count + prompts.AssistantPromptPrefixTokenIds.Count + 1);
            suffixTokens.Add(t.AudioEndTokenId);
            suffixTokens.AddRange(prompts.UserPromptAfterReferenceTokenIds);
            suffixTokens.AddRange(textTokenIds);
            suffixTokens.AddRange(prompts.AssistantPromptPrefixTokenIds);
            suffixTokens.Add(t.AudioStartTokenId);

            var prefixRows = BuildTextRows(prefixTokens);
            var audioRows = BuildAudioPrefixRows(promptAudioCodes);
            var suffixRows = BuildTextRows(suffixTokens);

            var rows = new int[prefixRows.Length + audioRows.Length + suffixRows.Length][];
            int o = 0;
            Array.Copy(prefixRows, 0, rows, o, prefixRows.Length); o += prefixRows.Length;
            Array.Copy(audioRows, 0, rows, o, audioRows.Length); o += audioRows.Length;
            Array.Copy(suffixRows, 0, rows, o, suffixRows.Length);

            var attentionMask = new int[rows.Length];
            Array.Fill(attentionMask, 1);
            return new VoiceCloneRequestRows(rows, attentionMask);
        }

        // ---------------- Reference audio encoding ----------------

        /// <summary>
        /// 调用 codec encoder 把参考音频 PCM 编码成 audio codes。返回每帧 numQuantizers 个 codebook 索引。
        /// </summary>
        public List<int[]> EncodeReferenceAudio(string referenceAudioPath)
        {
            int sampleRate = _bundle.CodecMeta.CodecConfig.SampleRate;
            int channels = _bundle.CodecMeta.CodecConfig.Channels;
            int numQuantizers = _bundle.CodecMeta.CodecConfig.NumQuantizers;

            // [1, channels, samples]
            float[] waveform = ReferenceWavReader.LoadAsChannelMajor(referenceAudioPath, sampleRate, channels);
            int samples = waveform.Length / channels;

            using var waveformOrt = OrtValue.CreateTensorValueFromMemory(waveform, new long[] { 1, channels, samples });
            using var lengthsOrt = OrtValue.CreateTensorValueFromMemory(new int[] { samples }, new long[] { 1 });

            using var outputs = _bundle.CodecEncode.Run(_runOptions,
                new[] { "waveform", "input_lengths" },
                new[] { waveformOrt, lengthsOrt },
                _bundle.CodecEncodeOutputNames);

            var byName = ToNameMap(_bundle.CodecEncodeOutputNames, outputs);
            var audioCodesShape = byName["audio_codes"].GetTensorTypeAndShape().Shape;
            int codeFrames = (int)audioCodesShape[1];
            var audioCodesSpan = byName["audio_codes"].GetTensorDataAsSpan<int>();
            int codeLength = byName["audio_code_lengths"].GetTensorDataAsSpan<int>()[0];

            var result = new List<int[]>(codeLength);
            for (int f = 0; f < codeLength; f++)
            {
                var row = new int[numQuantizers];
                for (int q = 0; q < numQuantizers; q++)
                    row[q] = audioCodesSpan[f * numQuantizers + q];
                result.Add(row);
            }
            return result;
        }

        // ---------------- Local subgraph runs ----------------

        /// <summary>
        /// Python `run_local_decoder`: text_logits 与 audio_logits。audio_logits shape [1, n_vq, codebook_size]。
        /// </summary>
        public (float[] textLogits, float[] audioLogits, int audioPerChannel) RunLocalDecoder(
            ReadOnlySpan<float> globalHidden,
            int textTokenId,
            IReadOnlyList<int> framePrefix)
        {
            int nVq = _bundle.Manifest.TtsConfig.NVq;
            int audioPad = _bundle.Manifest.TtsConfig.AudioPadTokenId;
            int hiddenSize = _bundle.TtsMeta.ModelConfig.HiddenSize;

            var hidden = globalHidden.ToArray();

            int prefixLen = nVq - 1;
            var prefix = new int[prefixLen];
            for (int i = 0; i < prefixLen; i++) prefix[i] = audioPad;
            int copy = Math.Min(framePrefix.Count, prefixLen);
            for (int i = 0; i < copy; i++) prefix[i] = framePrefix[i];

            using var hiddenOrt = OrtValue.CreateTensorValueFromMemory(hidden, new long[] { 1, hiddenSize });
            using var textTokenOrt = OrtValue.CreateTensorValueFromMemory(new int[] { textTokenId }, new long[] { 1 });
            using var prefixOrt = OrtValue.CreateTensorValueFromMemory(prefix, new long[] { 1, prefixLen });

            using var outputs = _bundle.LocalDecoder.Run(_runOptions,
                new[] { "global_hidden", "text_token_id", "audio_prefix_token_ids" },
                new[] { hiddenOrt, textTokenOrt, prefixOrt },
                _bundle.LocalDecoderOutputNames);

            var byName = ToNameMap(_bundle.LocalDecoderOutputNames, outputs);
            var textLogits = byName["text_logits"].GetTensorDataAsSpan<float>().ToArray();
            var audioLogitsShape = byName["audio_logits"].GetTensorTypeAndShape().Shape;
            int perChannel = (int)audioLogitsShape[^1];
            var audioLogits = byName["audio_logits"].GetTensorDataAsSpan<float>().ToArray();
            return (textLogits, audioLogits, perChannel);
        }

        /// <summary>
        /// Python `run_local_fixed_sampled_frame`: 一次性输出整帧 token + should_continue。
        /// 调用方传入复用的 <paramref name="seenMask"/> (shape [1, nVq, codebookSize]) 与 <paramref name="nVq"/>;
        /// engine 内部不再每帧 new mask, 由调用方在拿到结果后增量写入。
        /// </summary>
        public (bool shouldContinue, int[] frame) RunLocalFixedSampledFrame(
            float[] globalHidden,
            int[] seenMask,
            int nVq)
        {
            if (_bundle.LocalFixedSampledFrame is null)
                throw new InvalidOperationException("local_fixed_sampled_frame session 不存在。");

            int audioCodebookSize = _bundle.TtsMeta.ModelConfig.AudioCodebookSizes[0];
            int hiddenSize = _bundle.TtsMeta.ModelConfig.HiddenSize;

            _assistantUBuffer[0] = (float)Math.Min(0.99999994, Math.Max(0.0, _rng.NextDouble()));
            if (_audioUBuffer.Length < nVq) _audioUBuffer = new float[nVq];
            for (int i = 0; i < nVq; i++)
                _audioUBuffer[i] = (float)Math.Min(0.99999994, Math.Max(0.0, _rng.NextDouble()));

            using var hiddenOrt = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(globalHidden, 0, hiddenSize),
                new long[] { 1, hiddenSize });
            using var seenOrt = OrtValue.CreateTensorValueFromMemory<int>(
                OrtMemoryInfo.DefaultInstance,
                new Memory<int>(seenMask, 0, nVq * audioCodebookSize),
                new long[] { 1, nVq, audioCodebookSize });
            using var assistantUOrt = OrtValue.CreateTensorValueFromMemory(_assistantUBuffer, new long[] { 1 });
            using var audioUOrt = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                new Memory<float>(_audioUBuffer, 0, nVq),
                new long[] { 1, nVq });

            using var outputs = _bundle.LocalFixedSampledFrame.Run(_runOptions,
                new[] { "global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u" },
                new[] { hiddenOrt, seenOrt, assistantUOrt, audioUOrt },
                _bundle.LocalFixedSampledFrameOutputNames);

            var byName = ToNameMap(_bundle.LocalFixedSampledFrameOutputNames, outputs);
            int shouldContinueRaw = byName["should_continue"].GetTensorDataAsSpan<int>()[0];
            var tokenSpan = byName["frame_token_ids"].GetTensorDataAsSpan<int>();
            var frame = new int[nVq];
            for (int i = 0; i < nVq; i++) frame[i] = tokenSpan[i];
            return (shouldContinueRaw != 0, frame);
        }

        // ---------------- Codec full decode ----------------

        /// <summary>
        /// Python `decode_full_audio`: 一次性把所有 frames 解码成 channel-major PCM。
        /// 返回 [channels][samples] 数组。
        /// </summary>
        public (float[][] channelArrays, int audioLength) DecodeFullAudio(IReadOnlyList<int[]> generatedFrames)
        {
            if (generatedFrames.Count == 0) return (Array.Empty<float[]>(), 0);

            int numQuantizers = _bundle.CodecMeta.CodecConfig.NumQuantizers;
            int frameCount = generatedFrames.Count;
            var codes = new int[1 * frameCount * numQuantizers];
            for (int f = 0; f < frameCount; f++)
            {
                var row = generatedFrames[f];
                for (int q = 0; q < numQuantizers; q++)
                    codes[f * numQuantizers + q] = q < row.Length ? row[q] : 0;
            }

            using var codesOrt = OrtValue.CreateTensorValueFromMemory(codes, new long[] { 1, frameCount, numQuantizers });
            using var lengthsOrt = OrtValue.CreateTensorValueFromMemory(new int[] { frameCount }, new long[] { 1 });

            using var outputs = _bundle.CodecDecodeFull.Run(_runOptions,
                new[] { "audio_codes", "audio_code_lengths" },
                new[] { codesOrt, lengthsOrt },
                _bundle.CodecDecodeFullOutputNames);

            var byName = ToNameMap(_bundle.CodecDecodeFullOutputNames, outputs);
            int audioLength = byName["audio_lengths"].GetTensorDataAsSpan<int>()[0];
            var audioOrt = byName["audio"];
            var shape = audioOrt.GetTensorTypeAndShape().Shape;
            int channels = (int)shape[1];
            int totalSamples = (int)shape[2];
            var audioSpan = audioOrt.GetTensorDataAsSpan<float>();
            var channelArrays = new float[channels][];
            for (int c = 0; c < channels; c++)
            {
                var arr = new float[audioLength];
                for (int t = 0; t < audioLength; t++)
                    arr[t] = audioSpan[c * totalSamples + t];
                channelArrays[c] = arr;
            }
            return (channelArrays, audioLength);
        }

        // ---------------- Helpers ----------------

        /// <summary>
        /// 把 outputs 与对应的 output-name 数组合成一个 name->OrtValue 字典, 便于按名称取值。
        /// 注意: 返回的字典在 outputs 生命周期内有效, 不要持有到 outputs Dispose 之后。
        /// </summary>
        internal static Dictionary<string, OrtValue> ToNameMap(
            string[] names,
            IDisposableReadOnlyCollection<OrtValue> outputs)
        {
            var d = new Dictionary<string, OrtValue>(names.Length, StringComparer.Ordinal);
            for (int i = 0; i < names.Length; i++) d[names[i]] = outputs[i];
            return d;
        }

        internal RunOptions RunOptions => _runOptions;

        public void Dispose()
        {
            // RunOptions 持有原生 OrtRunOptions 句柄 (DisposableObject), 必须确定性释放,
            // 否则每个服务实例泄漏 1 个句柄 (与 CodecStreamingDecodeSession 一致)。
            _runOptions.Dispose();
        }
    }
}
