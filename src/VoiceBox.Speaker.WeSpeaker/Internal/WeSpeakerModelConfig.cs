using System.Globalization;

namespace VoiceBox.Speaker.WeSpeaker.Internal
{
    /// <summary>Strongly typed and validated view of the model directory's config.ini.</summary>
    internal sealed class WeSpeakerModelConfig
    {
        public string ModelFile { get; private set; } = "cnceleb_resnet34.onnx";
        public string ModelSha256 { get; private set; } = string.Empty;
        public string InputName { get; private set; } = "feats";
        public string OutputName { get; private set; } = "embs";
        public int SampleRate { get; private set; } = 16000;
        public int FeatureDim { get; private set; } = 80;
        public int EmbeddingSize { get; private set; } = 256;
        public float Threshold { get; private set; } = 0.55f;

        public int FrameLengthMs { get; private set; } = 25;
        public int FrameShiftMs { get; private set; } = 10;
        public int NumMelBins { get; private set; } = 80;
        public string WindowType { get; private set; } = "hamming";
        public float Dither { get; private set; }
        public float Preemphasis { get; private set; } = 0.97f;
        public bool RemoveDcOffset { get; private set; } = true;
        public bool SnipEdges { get; private set; } = true;
        public float LowFreq { get; private set; } = 20f;
        public float HighFreq { get; private set; }
        public float WaveformScale { get; private set; } = 32768f;
        public bool ApplyCmn { get; private set; } = true;

        public static WeSpeakerModelConfig Load(string configPath)
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"WeSpeaker model config not found: {configPath}");

            Dictionary<string, string> map = ParseIni(configPath);
            var config = new WeSpeakerModelConfig
            {
                ModelFile = GetString(map, "model_file", "cnceleb_resnet34.onnx"),
                ModelSha256 = GetString(map, "model_sha256", string.Empty),
                InputName = GetString(map, "input_name", "feats"),
                OutputName = GetString(map, "output_name", "embs"),
                SampleRate = GetInt(map, "sample_rate", 16000, configPath),
                FeatureDim = GetInt(map, "feature_dim", 80, configPath),
                EmbeddingSize = GetInt(map, "embedding_size", 256, configPath),
                Threshold = GetFloat(map, "threshold", 0.55f, configPath),
                FrameLengthMs = GetInt(map, "frame_length_ms", 25, configPath),
                FrameShiftMs = GetInt(map, "frame_shift_ms", 10, configPath),
                NumMelBins = GetInt(map, "num_mel_bins", 80, configPath),
                WindowType = GetString(map, "window_type", "hamming").ToLowerInvariant(),
                Dither = GetFloat(map, "dither", 0f, configPath),
                Preemphasis = GetFloat(map, "preemphasis", 0.97f, configPath),
                RemoveDcOffset = GetInt(map, "remove_dc_offset", 1, configPath) != 0,
                SnipEdges = GetInt(map, "snip_edges", 1, configPath) != 0,
                LowFreq = GetFloat(map, "low_freq", 20f, configPath),
                HighFreq = GetFloat(map, "high_freq", 0f, configPath),
                WaveformScale = GetFloat(map, "waveform_scale", 32768f, configPath),
                ApplyCmn = GetInt(map, "apply_cmn", 1, configPath) != 0,
            };
            config.Validate(configPath);
            return config;
        }

        private void Validate(string configPath)
        {
            if (Path.GetFileName(ModelFile) != ModelFile ||
                !ModelFile.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                throw Invalid($"Invalid model_file '{ModelFile}'.", configPath);
            }
            if (ModelSha256.Length != 64 ||
                ModelSha256.Any(c => !Uri.IsHexDigit(c)))
            {
                throw Invalid("model_sha256 must contain exactly 64 hexadecimal characters.", configPath);
            }
            if (InputName != "feats" || OutputName != "embs")
                throw Invalid($"Unsupported tensor names '{InputName}'/'{OutputName}' (expect feats/embs).", configPath);
            if (SampleRate != 16000)
                throw Invalid($"Unsupported sample_rate {SampleRate} (expect 16000).", configPath);
            if (FeatureDim != 80 || NumMelBins != 80)
                throw Invalid($"Unsupported feature dimensions {FeatureDim}/{NumMelBins} (expect 80/80).", configPath);
            if (EmbeddingSize != 256)
                throw Invalid($"Unsupported embedding_size {EmbeddingSize} (expect 256).", configPath);
            if (FrameLengthMs != 25 || FrameShiftMs != 10)
                throw Invalid($"Unsupported frame config {FrameLengthMs}/{FrameShiftMs}ms (expect 25/10).", configPath);
            if (WindowType != "hamming")
                throw Invalid($"Unsupported window_type '{WindowType}' (expect hamming).", configPath);
            if (Dither != 0f)
                throw Invalid($"Unsupported dither {Dither} (expect 0).", configPath);
            if (Preemphasis != 0.97f)
                throw Invalid($"Unsupported preemphasis {Preemphasis} (expect 0.97).", configPath);
            if (!RemoveDcOffset || !SnipEdges)
                throw Invalid("remove_dc_offset and snip_edges must both be enabled.", configPath);
            if (LowFreq != 20f || HighFreq != 0f)
                throw Invalid($"Unsupported mel range {LowFreq}/{HighFreq} (expect 20/0).", configPath);
            if (WaveformScale != 32768f)
                throw Invalid($"Unsupported waveform_scale {WaveformScale} (expect 32768).", configPath);
            if (!ApplyCmn)
                throw Invalid("apply_cmn must be enabled.", configPath);
            if (!(Threshold >= -1f && Threshold <= 1f))
                throw Invalid($"Invalid threshold {Threshold}.", configPath);
        }

        private static InvalidOperationException Invalid(string message, string path) =>
            new($"{message} [{path}]");

        private static Dictionary<string, string> ParseIni(string path)
        {
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string raw in File.ReadLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] is '[' or '#' or ';') continue;
                int separator = line.IndexOf('=');
                if (separator <= 0) continue;
                map[line[..separator].Trim()] = line[(separator + 1)..].Trim();
            }
            return map;
        }

        private static string GetString(Dictionary<string, string> values, string key, string fallback) =>
            values.TryGetValue(key, out string? value) ? value : fallback;

        private static int GetInt(
            Dictionary<string, string> values,
            string key,
            int fallback,
            string configPath)
        {
            if (!values.TryGetValue(key, out string? value)) return fallback;
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
                return parsed;
            throw Invalid($"Invalid integer value '{value}' for {key}.", configPath);
        }

        private static float GetFloat(
            Dictionary<string, string> values,
            string key,
            float fallback,
            string configPath)
        {
            if (!values.TryGetValue(key, out string? value)) return fallback;
            if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                return parsed;
            throw Invalid($"Invalid floating-point value '{value}' for {key}.", configPath);
        }
    }
}
