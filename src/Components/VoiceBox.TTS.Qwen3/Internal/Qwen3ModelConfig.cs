using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Parsed Qwen3-TTS package config (<c>embeddings/config.json</c>) + <c>metadata.json</c>.
    /// Mirrors the schema written by scripts/qwen3-tts-onnx-export/export_embeddings.py + package.py.
    /// </summary>
    internal sealed class Qwen3ModelConfig
    {
        public string? TtsModelSize { get; init; }
        public string? TtsModelType { get; init; }
        public int SampleRate { get; init; } = 24000;
        public int SamplesPerFrame { get; init; } = 1920;
        public int NumCodeGroups { get; init; } = 16;
        /// <summary>Fixed KV length of talker_static.onnx (CUDA-graph path); 0 = no static graph.</summary>
        public int StaticCacheLen { get; init; }
        public SpecialTokensCfg SpecialTokens { get; init; } = new();
        public Dictionary<string, int> CodecLanguageId { get; init; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, int> SpeakerId { get; init; } = new(StringComparer.OrdinalIgnoreCase);
        /// <summary>speaker → dialect language key, or null when not a dialect.</summary>
        public Dictionary<string, string?> SpeakerIsDialect { get; init; } = new(StringComparer.OrdinalIgnoreCase);
        public SuppressCfg SuppressTokens { get; init; } = new();
        public TalkerCfg Talker { get; init; } = new();
        public CodePredictorCfg CodePredictor { get; init; } = new();
        public GenerationCfg Generation { get; init; } = new();
        public EmbeddingsManifest Embeddings { get; init; } = new();

        /// <summary>I/O dtype of the talker/cp graphs ("fp32"/"fp16"/"int8"), from metadata.json.</summary>
        public string Dtype { get; private set; } = "fp32";

        public static Qwen3ModelConfig Load(string modelDir)
        {
            string cfgPath = Path.Combine(modelDir, "embeddings", "config.json");
            if (!File.Exists(cfgPath))
                throw new FileNotFoundException($"Qwen3-TTS 配置不存在: {cfgPath}", cfgPath);

            using var doc = JsonDocument.Parse(File.ReadAllText(cfgPath));
            var root = doc.RootElement;
            var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

            var cfg = new Qwen3ModelConfig
            {
                TtsModelSize = GetString(root, "ttsModelSize"),
                TtsModelType = GetString(root, "ttsModelType"),
                SampleRate = GetInt(root, "sampleRate", 24000),
                SamplesPerFrame = GetInt(root, "samplesPerFrame", 1920),
                NumCodeGroups = GetInt(root, "numCodeGroups", 16),
                StaticCacheLen = GetInt(root, "staticCacheLen", 0),
                SpecialTokens = Deser<SpecialTokensCfg>(root, "specialTokens", opts) ?? new(),
                CodecLanguageId = DeserDict(root, "codecLanguageId"),
                SpeakerId = DeserDict(root, "speakerId"),
                SpeakerIsDialect = ParseDialect(root),
                SuppressTokens = Deser<SuppressCfg>(root, "suppressTokens", opts) ?? new(),
                Talker = Deser<TalkerCfg>(root, "talker", opts) ?? new(),
                CodePredictor = Deser<CodePredictorCfg>(root, "codePredictor", opts) ?? new(),
                Generation = Deser<GenerationCfg>(root, "generation", opts) ?? new(),
                Embeddings = Deser<EmbeddingsManifest>(root, "embeddings", opts) ?? new(),
            };

            string metaPath = Path.Combine(modelDir, "metadata.json");
            if (File.Exists(metaPath))
            {
                using var meta = JsonDocument.Parse(File.ReadAllText(metaPath));
                cfg.Dtype = GetString(meta.RootElement, "dtype") ?? "fp32";
            }
            return cfg;
        }

        private static string? GetString(JsonElement e, string name)
            => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

        private static int GetInt(JsonElement e, string name, int dflt)
            => e.TryGetProperty(name, out var v) && v.TryGetInt32(out int i) ? i : dflt;

        private static T? Deser<T>(JsonElement e, string name, JsonSerializerOptions o)
            => e.TryGetProperty(name, out var v) ? v.Deserialize<T>(o) : default;

        private static Dictionary<string, int> DeserDict(JsonElement e, string name)
        {
            var d = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            if (e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Object)
                foreach (var p in v.EnumerateObject())
                    if (p.Value.TryGetInt32(out int i)) d[p.Name] = i;
            return d;
        }

        private static Dictionary<string, string?> ParseDialect(JsonElement root)
        {
            var d = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
            if (root.TryGetProperty("speakerIsDialect", out var v) && v.ValueKind == JsonValueKind.Object)
                foreach (var p in v.EnumerateObject())
                    d[p.Name] = p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() : null;
            return d;
        }
    }

    internal sealed class SpecialTokensCfg
    {
        public int ImStart { get; init; }
        public int ImEnd { get; init; }
        public int Assistant { get; init; }
        public int TtsBos { get; init; }
        public int TtsEos { get; init; }
        public int TtsPad { get; init; }
        public int CodecBos { get; init; }
        public int CodecEos { get; init; }
        public int CodecPad { get; init; }
        public int CodecThink { get; init; }
        public int CodecNothink { get; init; }
        public int CodecThinkBos { get; init; }
        public int CodecThinkEos { get; init; }
    }

    internal sealed class SuppressCfg
    {
        public int RangeStart { get; init; }
        public int RangeEnd { get; init; }
        public int[] Keep { get; init; } = [];
    }

    internal sealed class TalkerCfg
    {
        public int HiddenSize { get; init; }
        public int NumLayers { get; init; }
        public int NumHeads { get; init; }
        public int NumKvHeads { get; init; }
        public int HeadDim { get; init; }
        public int TextHiddenSize { get; init; }
        public int TextVocabSize { get; init; }
        public int CodecVocabSize { get; init; }
        public float RopeTheta { get; init; }
        public float RmsNormEps { get; init; }
    }

    internal sealed class CodePredictorCfg
    {
        public int HiddenSize { get; init; }
        public int NumLayers { get; init; }
        public int NumHeads { get; init; }
        public int NumKvHeads { get; init; }
        public int HeadDim { get; init; }
        public int VocabSize { get; init; }
        public int InputDim { get; init; }
    }

    internal sealed class GenerationCfg
    {
        public bool DoSample { get; init; } = true;
        public float Temperature { get; init; } = 0.9f;
        public int TopK { get; init; } = 50;
        public float TopP { get; init; } = 1.0f;
        public float RepetitionPenalty { get; init; } = 1.05f;
        public bool SubtalkerDoSample { get; init; } = true;
        public float SubtalkerTemperature { get; init; } = 0.9f;
        public int SubtalkerTopK { get; init; } = 50;
        public float SubtalkerTopP { get; init; } = 1.0f;
        public int MaxNewTokens { get; init; } = 8192;
        public int MinNewTokens { get; init; } = 2;
    }

    internal sealed class EmbeddingEntry
    {
        public string File { get; init; } = string.Empty;
        public int[] Shape { get; init; } = [];
    }

    internal sealed class EmbeddingsManifest
    {
        public EmbeddingEntry? TextProjectedEmbedding { get; init; }
        public EmbeddingEntry TextEmbedding { get; init; } = new();
        public EmbeddingEntry TextProjFc1Weight { get; init; } = new();
        public EmbeddingEntry TextProjFc1Bias { get; init; } = new();
        public EmbeddingEntry TextProjFc2Weight { get; init; } = new();
        public EmbeddingEntry TextProjFc2Bias { get; init; } = new();
        public EmbeddingEntry CodecEmbedding { get; init; } = new();
        public EmbeddingEntry CpCodecEmbedding { get; init; } = new();
    }
}
