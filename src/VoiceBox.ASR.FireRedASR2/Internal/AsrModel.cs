using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Loads the three FireRed ONNX models from disk, parses encoder metadata and exposes
    /// an immutable <see cref="AsrModelMetadata"/> alongside owned <see cref="AsrInferenceSessions"/>.
    /// Use <see cref="Builder"/> to assemble the file paths.
    /// </summary>
    internal sealed class AsrModel : IDisposable
    {
        public const int EncoderHiddenDim = 1280;
        public const int DecoderCacheLayers = 16;
        public const int DefaultBlankId = 0;
        public const int DefaultUnkId = 1;
        public const int DefaultPadId = 2;
        public const int DefaultSosId = 3;
        public const int DefaultEosId = 4;
        public const int DefaultFeatureDim = 80;
        public const int DefaultSampleRate = 16000;

        public AsrInferenceSessions Sessions { get; }
        public AsrModelMetadata Metadata { get; }
        public CustomMetadata RawMetadata { get; }

        private AsrModel(AsrInferenceSessions sessions, AsrModelMetadata metadata, CustomMetadata raw)
        {
            Sessions = sessions;
            Metadata = metadata;
            RawMetadata = raw;
        }

        public void Dispose() => Sessions.Dispose();

        /// <summary>
        /// Fluent builder for assembling an <see cref="AsrModel"/> from a directory of files.
        /// </summary>
        public sealed class Builder
        {
            public string? EncoderFilePath { get; set; }
            public string? DecoderFilePath { get; set; }
            public string? CtcFilePath { get; set; }
            public OnnxExecutionOptions? OnnxOptions { get; set; }

            public Builder WithEncoder(string path) { EncoderFilePath = path; return this; }
            public Builder WithDecoder(string? path) { DecoderFilePath = path; return this; }
            public Builder WithCtc(string? path) { CtcFilePath = path; return this; }
            public Builder WithOnnxOptions(OnnxExecutionOptions? options) { OnnxOptions = options; return this; }

            public AsrModel Build()
            {
                if (string.IsNullOrWhiteSpace(EncoderFilePath))
                    throw new InvalidOperationException("Encoder file path is required.");

                using var sessionOptions = CreateSessionOptions(OnnxOptions);

                InferenceSession encoder = LoadSession(EncoderFilePath, sessionOptions)
                    ?? throw new FileNotFoundException("Encoder ONNX file not found.", EncoderFilePath);
                InferenceSession? decoder = null;
                InferenceSession? ctc = null;

                try
                {
                    decoder = LoadSession(DecoderFilePath, sessionOptions);
                    ctc = LoadSession(CtcFilePath, sessionOptions);

                    var raw = ParseEncoderMetadata(encoder, fallbackSos: DefaultSosId, fallbackEos: DefaultEosId);
                    var meta = BuildMetadata(raw);

                    return new AsrModel(new AsrInferenceSessions(encoder, decoder, ctc), meta, raw);
                }
                catch
                {
                    decoder?.Dispose();
                    ctc?.Dispose();
                    encoder.Dispose();
                    throw;
                }
            }
        }

        private static AsrModelMetadata BuildMetadata(CustomMetadata raw)
        {
            int requiredCacheSize = raw.LeftChunks <= 0
                ? 0
                : raw.ChunkSize * raw.LeftChunks;

            int chunkLength = (raw.ChunkSize - 1) * raw.SubsamplingRate + raw.RightContext + 1;
            int shiftLength = raw.SubsamplingRate * raw.ChunkSize;

            if (chunkLength <= 0) chunkLength = raw.DecodingWindow;
            if (shiftLength <= 0) shiftLength = chunkLength;

            int sosId = raw.SosSymbol != 0 ? raw.SosSymbol : DefaultSosId;
            int eosId = raw.EosSymbol != 0 ? raw.EosSymbol : DefaultEosId;

            // FBank frame shift is fixed at 10 ms (matches FrontendConfEntity defaults).
            // Encoder frame shift = fbank_frame_shift_ms * subsampling_rate / 1000.
            // SubsamplingRate falls back to 4 (FireRed default) when the encoder metadata
            // doesn't carry it.
            int subsampling = raw.SubsamplingRate > 0 ? raw.SubsamplingRate : 4;
            double encoderFrameShiftSeconds = 0.010 * subsampling;

            return new AsrModelMetadata
            {
                BlankId = DefaultBlankId,
                UnkId = DefaultUnkId,
                PadId = DefaultPadId,
                SosId = sosId,
                EosId = eosId,
                FeatureDim = DefaultFeatureDim,
                SampleRate = DefaultSampleRate,
                ChunkLength = chunkLength,
                ShiftLength = shiftLength,
                RequiredCacheSize = requiredCacheSize,
                EncoderHiddenDim = EncoderHiddenDim,
                DecoderCacheLayers = DecoderCacheLayers,
                EncoderFrameShiftSeconds = encoderFrameShiftSeconds,
            };
        }

        private static CustomMetadata ParseEncoderMetadata(InferenceSession encoder, int fallbackSos, int fallbackEos)
        {
            var map = encoder.ModelMetadata.CustomMetadataMap;
            var m = new CustomMetadata
            {
                SosSymbol = fallbackSos,
                EosSymbol = fallbackEos,
            };

            if (map.TryGetValue("output_dir", out var outputDir)) m.OutputDir = outputDir;
            if (map.TryGetValue("onnx.infer", out var onnxInfer)) m.OnnxInfer = onnxInfer;
            if (map.TryGetValue("decoder", out var decoder)) m.Decoder = decoder;
            if (map.TryGetValue("encoder", out var encoderName)) m.Encoder = encoderName;

            m.OutputSize = ParseInt(map, "output_size", m.OutputSize);
            m.LeftChunks = ParseInt(map, "left_chunks", m.LeftChunks);
            m.Batch = ParseInt(map, "batch", m.Batch);
            m.ReverseWeight = ParseFloat(map, "reverse_weight", m.ReverseWeight);
            m.ChunkSize = ParseInt(map, "chunk_size", m.ChunkSize);
            m.NumBlocks = ParseInt(map, "num_blocks", m.NumBlocks);
            m.CnnModuleKernel = ParseInt(map, "cnn_module_kernel", m.CnnModuleKernel);
            m.Head = ParseInt(map, "head", m.Head);
            m.FeatureSize = ParseInt(map, "feature_size", m.FeatureSize);
            m.VocabSize = ParseInt(map, "vocab_size", m.VocabSize);
            m.DecodingWindow = ParseInt(map, "decoding_window", m.DecodingWindow);
            m.SubsamplingRate = ParseInt(map, "subsampling_rate", m.SubsamplingRate);
            m.RightContext = ParseInt(map, "right_context", m.RightContext);
            m.EosSymbol = ParseInt(map, "eos_symbol", m.EosSymbol);
            m.SosSymbol = ParseInt(map, "sos_symbol", m.SosSymbol);
            m.IsBidirectionalDecoder = ParseBool(map, "is_bidirectional_decoder", m.IsBidirectionalDecoder);

            return m;
        }

        private static int ParseInt(IReadOnlyDictionary<string, string> map, string key, int fallback)
        {
            if (!map.TryGetValue(key, out var raw) || raw is null) return fallback;
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : fallback;
        }

        private static float ParseFloat(IReadOnlyDictionary<string, string> map, string key, float fallback)
        {
            if (!map.TryGetValue(key, out var raw) || raw is null) return fallback;
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;
        }

        private static bool ParseBool(IReadOnlyDictionary<string, string> map, string key, bool fallback)
        {
            if (!map.TryGetValue(key, out var raw) || raw is null) return fallback;
            return bool.TryParse(raw, out var v) ? v : fallback;
        }

        private static InferenceSession? LoadSession(string? path, SessionOptions options)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;

            // The legacy code allowed loading an ONNX from an embedded resource when the path
            // had no slashes. Keeping that here preserves behaviour for users who package
            // models as resources, even though current callers always pass real paths.
            bool isPath = path.IndexOfAny(new[] { '/', '\\' }) >= 0;
            if (isPath) return new InferenceSession(path, options);

            byte[] bytes = ReadEmbeddedResource(path);
            return new InferenceSession(bytes, options);
        }

        private static byte[] ReadEmbeddedResource(string resourceName)
        {
            using Stream? stream = typeof(AsrModel).Assembly.GetManifestResourceStream(resourceName)
                ?? throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            return ms.ToArray();
        }

        private static SessionOptions CreateSessionOptions(OnnxExecutionOptions? onnxOptions)
        {
            var options = onnxOptions is null
                ? OnnxSessionFactory.CreateCpu(0)
                : OnnxSessionFactory.Create(onnxOptions);
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            return options;
        }
    }
}
