using System.Text.Json.Serialization;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// browser_poc_manifest.json 的 1:1 数据映射。仅保留 C# 推理实际需要的字段。
    /// </summary>
    internal sealed class ModelManifest
    {
        [JsonPropertyName("model_files")]
        public ManifestModelFiles ModelFiles { get; set; } = new();

        [JsonPropertyName("tts_config")]
        public ManifestTtsConfig TtsConfig { get; set; } = new();

        [JsonPropertyName("prompt_templates")]
        public ManifestPromptTemplates PromptTemplates { get; set; } = new();

        [JsonPropertyName("generation_defaults")]
        public ManifestGenerationDefaults GenerationDefaults { get; set; } = new();

        [JsonPropertyName("builtin_voices")]
        public List<ManifestBuiltinVoice> BuiltinVoices { get; set; } = new();
    }

    internal sealed class ManifestModelFiles
    {
        [JsonPropertyName("tts_meta")] public string TtsMeta { get; set; } = "tts_browser_onnx_meta.json";
        [JsonPropertyName("codec_meta")] public string CodecMeta { get; set; } = string.Empty;
        [JsonPropertyName("tokenizer_model")] public string TokenizerModel { get; set; } = "tokenizer.model";
    }

    internal sealed class ManifestTtsConfig
    {
        [JsonPropertyName("n_vq")] public int NVq { get; set; }
        [JsonPropertyName("audio_pad_token_id")] public int AudioPadTokenId { get; set; }
        [JsonPropertyName("audio_start_token_id")] public int AudioStartTokenId { get; set; }
        [JsonPropertyName("audio_end_token_id")] public int AudioEndTokenId { get; set; }
        [JsonPropertyName("audio_user_slot_token_id")] public int AudioUserSlotTokenId { get; set; }
        [JsonPropertyName("audio_assistant_slot_token_id")] public int AudioAssistantSlotTokenId { get; set; }
    }

    internal sealed class ManifestPromptTemplates
    {
        [JsonPropertyName("user_prompt_prefix_token_ids")] public List<int> UserPromptPrefixTokenIds { get; set; } = new();
        [JsonPropertyName("user_prompt_after_reference_token_ids")] public List<int> UserPromptAfterReferenceTokenIds { get; set; } = new();
        [JsonPropertyName("assistant_prompt_prefix_token_ids")] public List<int> AssistantPromptPrefixTokenIds { get; set; } = new();
    }

    internal sealed class ManifestGenerationDefaults
    {
        [JsonPropertyName("max_new_frames")] public int MaxNewFrames { get; set; } = 375;
        [JsonPropertyName("do_sample")] public bool DoSample { get; set; } = true;
        [JsonPropertyName("text_temperature")] public float TextTemperature { get; set; } = 1f;
        [JsonPropertyName("text_top_p")] public float TextTopP { get; set; } = 1f;
        [JsonPropertyName("text_top_k")] public int TextTopK { get; set; } = 50;
        [JsonPropertyName("audio_temperature")] public float AudioTemperature { get; set; } = 0.8f;
        [JsonPropertyName("audio_top_p")] public float AudioTopP { get; set; } = 0.95f;
        [JsonPropertyName("audio_top_k")] public int AudioTopK { get; set; } = 25;
        [JsonPropertyName("audio_repetition_penalty")] public float AudioRepetitionPenalty { get; set; } = 1.2f;
        [JsonPropertyName("sample_mode")] public string SampleMode { get; set; } = "fixed";
    }

    internal sealed class ManifestBuiltinVoice
    {
        [JsonPropertyName("voice")] public string Voice { get; set; } = string.Empty;
        [JsonPropertyName("display_name")] public string? DisplayName { get; set; }
        [JsonPropertyName("group")] public string? Group { get; set; }
        [JsonPropertyName("prompt_audio_codes")] public List<List<int>> PromptAudioCodes { get; set; } = new();
    }

    /// <summary>
    /// tts_browser_onnx_meta.json 1:1 映射。
    /// </summary>
    internal sealed class TtsMeta
    {
        [JsonPropertyName("files")] public TtsMetaFiles Files { get; set; } = new();
        [JsonPropertyName("model_config")] public TtsMetaModelConfig ModelConfig { get; set; } = new();
        [JsonPropertyName("onnx")] public TtsMetaOnnx Onnx { get; set; } = new();
    }

    internal sealed class TtsMetaFiles
    {
        [JsonPropertyName("prefill")] public string Prefill { get; set; } = "moss_tts_prefill.onnx";
        [JsonPropertyName("decode_step")] public string DecodeStep { get; set; } = "moss_tts_decode_step.onnx";
        [JsonPropertyName("local_decoder")] public string LocalDecoder { get; set; } = "moss_tts_local_decoder.onnx";
        [JsonPropertyName("local_cached_step")] public string? LocalCachedStep { get; set; }
        [JsonPropertyName("local_greedy_frame")] public string? LocalGreedyFrame { get; set; }
        [JsonPropertyName("local_fixed_sampled_frame")] public string? LocalFixedSampledFrame { get; set; }
    }

    internal sealed class TtsMetaModelConfig
    {
        [JsonPropertyName("n_vq")] public int NVq { get; set; }
        [JsonPropertyName("hidden_size")] public int HiddenSize { get; set; }
        [JsonPropertyName("global_layers")] public int GlobalLayers { get; set; }
        [JsonPropertyName("local_layers")] public int LocalLayers { get; set; }
        [JsonPropertyName("local_heads")] public int LocalHeads { get; set; }
        [JsonPropertyName("local_head_dim")] public int LocalHeadDim { get; set; }
        [JsonPropertyName("audio_codebook_sizes")] public List<int> AudioCodebookSizes { get; set; } = new();
    }

    internal sealed class TtsMetaOnnx
    {
        [JsonPropertyName("prefill_output_names")] public List<string> PrefillOutputNames { get; set; } = new();
        [JsonPropertyName("decode_input_names")] public List<string> DecodeInputNames { get; set; } = new();
        [JsonPropertyName("decode_output_names")] public List<string> DecodeOutputNames { get; set; } = new();
        [JsonPropertyName("local_cached_output_names")] public List<string> LocalCachedOutputNames { get; set; } = new();
    }

    /// <summary>
    /// codec_browser_onnx_meta.json 1:1 映射。
    /// </summary>
    internal sealed class CodecMeta
    {
        [JsonPropertyName("files")] public CodecMetaFiles Files { get; set; } = new();
        [JsonPropertyName("codec_config")] public CodecMetaConfig CodecConfig { get; set; } = new();
        [JsonPropertyName("streaming_decode")] public CodecStreamingDecodeMeta StreamingDecode { get; set; } = new();
    }

    internal sealed class CodecMetaFiles
    {
        [JsonPropertyName("encode")] public string Encode { get; set; } = "moss_audio_tokenizer_encode.onnx";
        [JsonPropertyName("decode_full")] public string DecodeFull { get; set; } = "moss_audio_tokenizer_decode_full.onnx";
        [JsonPropertyName("decode_step")] public string DecodeStep { get; set; } = "moss_audio_tokenizer_decode_step.onnx";
    }

    internal sealed class CodecMetaConfig
    {
        [JsonPropertyName("sample_rate")] public int SampleRate { get; set; } = 48000;
        [JsonPropertyName("channels")] public int Channels { get; set; } = 2;
        [JsonPropertyName("num_quantizers")] public int NumQuantizers { get; set; } = 16;

        // 每个 audio frame 对应的原始采样点数 (codec hop)。SampleRate / DownsampleRate = 帧率。
        // 48000 / 3840 = 12.5 fps, 即每帧 80ms。
        [JsonPropertyName("downsample_rate")] public int DownsampleRate { get; set; } = 3840;
    }

    internal sealed class CodecStreamingDecodeMeta
    {
        [JsonPropertyName("transformer_offsets")] public List<TransformerOffsetSpec> TransformerOffsets { get; set; } = new();
        [JsonPropertyName("attention_caches")] public List<AttentionCacheSpec> AttentionCaches { get; set; } = new();
    }

    internal sealed class TransformerOffsetSpec
    {
        [JsonPropertyName("input_name")] public string InputName { get; set; } = string.Empty;
        [JsonPropertyName("output_name")] public string OutputName { get; set; } = string.Empty;
        [JsonPropertyName("shape")] public List<int> Shape { get; set; } = new();
    }

    internal sealed class AttentionCacheSpec
    {
        [JsonPropertyName("offset_input_name")] public string OffsetInputName { get; set; } = string.Empty;
        [JsonPropertyName("offset_output_name")] public string OffsetOutputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_keys_input_name")] public string CachedKeysInputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_keys_output_name")] public string CachedKeysOutputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_values_input_name")] public string CachedValuesInputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_values_output_name")] public string CachedValuesOutputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_positions_input_name")] public string CachedPositionsInputName { get; set; } = string.Empty;
        [JsonPropertyName("cached_positions_output_name")] public string CachedPositionsOutputName { get; set; } = string.Empty;
        [JsonPropertyName("offset_shape")] public List<int> OffsetShape { get; set; } = new();
        [JsonPropertyName("cache_shape")] public List<int> CacheShape { get; set; } = new();
        [JsonPropertyName("positions_shape")] public List<int> PositionsShape { get; set; } = new();
    }
}
