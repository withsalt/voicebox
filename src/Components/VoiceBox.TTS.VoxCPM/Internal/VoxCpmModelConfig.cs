using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// 部署包 config.json (由 export_embeddings.py 生成) 的强类型映射。
    /// </summary>
    internal sealed class VoxCpmModelConfig
    {
        [JsonPropertyName("model_type")] public string ModelType { get; set; } = "";
        [JsonPropertyName("version")] public string Version { get; set; } = "";
        [JsonPropertyName("sample_rate")] public int SampleRate { get; set; }
        [JsonPropertyName("hidden_size")] public int HiddenSize { get; set; }
        [JsonPropertyName("feat_dim")] public int FeatDim { get; set; }
        [JsonPropertyName("patch_size")] public int PatchSize { get; set; }
        [JsonPropertyName("vae_chunk_size")] public int VaeChunkSize { get; set; }
        [JsonPropertyName("samples_per_step")] public int SamplesPerStep { get; set; }
        [JsonPropertyName("vocab_size")] public int VocabSize { get; set; }
        [JsonPropertyName("base_lm_layers")] public int BaseLmLayers { get; set; }
        [JsonPropertyName("residual_lm_layers")] public int ResidualLmLayers { get; set; }
        [JsonPropertyName("num_attention_heads")] public int NumAttentionHeads { get; set; }
        [JsonPropertyName("num_key_value_heads")] public int NumKeyValueHeads { get; set; }
        [JsonPropertyName("head_dim")] public int HeadDim { get; set; }
        [JsonPropertyName("max_length")] public int MaxLength { get; set; }
        [JsonPropertyName("audio_start_token")] public int AudioStartToken { get; set; }
        [JsonPropertyName("generation")] public GenerationDefaults Generation { get; set; } = new();
        [JsonPropertyName("files")] public Dictionary<string, string> Files { get; set; } = [];
        [JsonPropertyName("text_embedding_shape")] public int[] TextEmbeddingShape { get; set; } = [];

        internal sealed class GenerationDefaults
        {
            [JsonPropertyName("inference_timesteps")] public int InferenceTimesteps { get; set; } = 10;
            [JsonPropertyName("cfg_value")] public float CfgValue { get; set; } = 2.0f;
            [JsonPropertyName("min_len")] public int MinLen { get; set; } = 2;
            [JsonPropertyName("max_len_text_ratio")] public float MaxLenTextRatio { get; set; } = 6.0f;
            [JsonPropertyName("max_len_text_bias")] public int MaxLenTextBias { get; set; } = 10;
            [JsonPropertyName("streaming_prefix_patches")] public int StreamingPrefixPatches { get; set; } = 3;
        }

        public static VoxCpmModelConfig Load(string modelPath)
        {
            string p = Path.Combine(modelPath, "config.json");
            if (!File.Exists(p))
                throw new FileNotFoundException($"VoxCPM 部署包缺少 config.json: {p}");
            var cfg = JsonSerializer.Deserialize<VoxCpmModelConfig>(File.ReadAllText(p))
                      ?? throw new InvalidDataException($"config.json 解析失败: {p}");
            if (!string.Equals(cfg.ModelType, "voxcpm", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"不是 VoxCPM 部署包 (model_type={cfg.ModelType}): {p}");
            if (cfg.HiddenSize <= 0 || cfg.PatchSize <= 0 || cfg.FeatDim <= 0)
                throw new InvalidDataException($"config.json 维度字段非法: {p}");
            return cfg;
        }

        public string FilePath(string modelPath, string key)
        {
            if (!Files.TryGetValue(key, out string? rel) || string.IsNullOrEmpty(rel))
                throw new InvalidDataException($"config.json files 缺少 '{key}'。");
            string p = Path.Combine(modelPath, rel);
            if (!File.Exists(p))
                throw new FileNotFoundException($"VoxCPM 部署包缺少 {key}: {p}");
            return p;
        }
    }
}
