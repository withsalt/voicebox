using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 持有 MOSS-TTS-Nano 所需的全部 ONNX session 及 metadata。
    /// 对应 Python `OrtCpuRuntime` 的 session 部分; 推理调度由 <see cref="InferenceEngine"/> 完成。
    /// 不是线程安全的; 上层串行化即可。
    /// </summary>
    internal sealed class ModelBundle : IDisposable
    {
        public ModelManifest Manifest { get; }
        public TtsMeta TtsMeta { get; }
        public CodecMeta CodecMeta { get; }

        public InferenceSession Prefill { get; }
        public InferenceSession Decode { get; }
        public InferenceSession LocalDecoder { get; }
        public InferenceSession? LocalCachedStep { get; }
        public InferenceSession? LocalFixedSampledFrame { get; }

        public InferenceSession CodecEncode { get; }
        public InferenceSession CodecDecodeFull { get; }
        public InferenceSession CodecDecodeStep { get; }

        // 缓存的输出名 (避免每次 Run 都 ToArray())
        public string[] PrefillOutputNames { get; }
        public string[] DecodeOutputNames { get; }
        public string[] DecodeInputNames { get; }
        public string[] LocalDecoderOutputNames { get; }
        public string[] LocalCachedOutputNames { get; }
        public string[] LocalFixedSampledFrameOutputNames { get; }
        public string[] CodecEncodeOutputNames { get; }
        public string[] CodecDecodeFullOutputNames { get; }

        public ModelBundle(
            ModelManifest manifest,
            TtsMeta ttsMeta,
            CodecMeta codecMeta,
            string ttsDir,
            string codecDir,
            SessionOptions ttsSessionOptions,
            SessionOptions codecSessionOptions)
        {
            Manifest = manifest;
            TtsMeta = ttsMeta;
            CodecMeta = codecMeta;

            Prefill = LoadSession(Path.Combine(ttsDir, ttsMeta.Files.Prefill), ttsSessionOptions);
            Decode = LoadSession(Path.Combine(ttsDir, ttsMeta.Files.DecodeStep), ttsSessionOptions);
            LocalDecoder = LoadSession(Path.Combine(ttsDir, ttsMeta.Files.LocalDecoder), ttsSessionOptions);
            LocalCachedStep = LoadOptional(ttsDir, ttsMeta.Files.LocalCachedStep, ttsSessionOptions);
            LocalFixedSampledFrame = LoadOptional(ttsDir, ttsMeta.Files.LocalFixedSampledFrame, ttsSessionOptions);

            CodecEncode = LoadSession(Path.Combine(codecDir, codecMeta.Files.Encode), codecSessionOptions);
            CodecDecodeFull = LoadSession(Path.Combine(codecDir, codecMeta.Files.DecodeFull), codecSessionOptions);
            CodecDecodeStep = LoadSession(Path.Combine(codecDir, codecMeta.Files.DecodeStep), codecSessionOptions);

            PrefillOutputNames = Prefill.OutputNames.ToArray();
            DecodeOutputNames = Decode.OutputNames.ToArray();
            DecodeInputNames = Decode.InputNames.ToArray();
            LocalDecoderOutputNames = LocalDecoder.OutputNames.ToArray();
            LocalCachedOutputNames = LocalCachedStep?.OutputNames.ToArray() ?? Array.Empty<string>();
            LocalFixedSampledFrameOutputNames = LocalFixedSampledFrame?.OutputNames.ToArray() ?? Array.Empty<string>();
            CodecEncodeOutputNames = CodecEncode.OutputNames.ToArray();
            CodecDecodeFullOutputNames = CodecDecodeFull.OutputNames.ToArray();
        }

        private static InferenceSession LoadSession(string path, SessionOptions options)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"ONNX 模型文件不存在: {path}", path);
            return new InferenceSession(path, options);
        }

        private static InferenceSession? LoadOptional(string dir, string? fileName, SessionOptions options)
        {
            if (string.IsNullOrEmpty(fileName)) return null;
            string path = Path.Combine(dir, fileName);
            return File.Exists(path) ? new InferenceSession(path, options) : null;
        }

        public void Dispose()
        {
            Prefill.Dispose();
            Decode.Dispose();
            LocalDecoder.Dispose();
            LocalCachedStep?.Dispose();
            LocalFixedSampledFrame?.Dispose();
            CodecEncode.Dispose();
            CodecDecodeFull.Dispose();
            CodecDecodeStep.Dispose();
        }
    }
}
