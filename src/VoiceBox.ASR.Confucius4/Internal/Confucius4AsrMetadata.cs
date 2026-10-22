using System.Text.Json.Serialization;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Mirror of the <c>metadata.json</c> bundled with the Confucius4 ONNX export.
    /// Only the fields the C# runtime needs are deserialised; unknown keys are tolerated.
    /// </summary>
    internal sealed class Confucius4AsrMetadata
    {
        [JsonPropertyName("dtype")] public string Dtype { get; set; } = "float32";
        [JsonPropertyName("encoder_model")] public string? EncoderModel { get; set; }
        [JsonPropertyName("conv_embed_model")] public string? ConvEmbedModel { get; set; }
        [JsonPropertyName("audio_stack_model")] public string? AudioStackModel { get; set; }
        [JsonPropertyName("prefill_model")] public string? PrefillModel { get; set; }
        [JsonPropertyName("decode_model")] public string? DecodeModel { get; set; }
        [JsonPropertyName("decoder_model")] public string? DecoderModel { get; set; }
        [JsonPropertyName("decode_cache_mode")] public string DecodeCacheMode { get; set; } = "full";
        [JsonPropertyName("n_window")] public int NWindow { get; set; } = 100;
        [JsonPropertyName("dynamic_chunk_frames")] public bool DynamicChunkFrames { get; set; }
        [JsonPropertyName("audio_output_dim")] public int AudioOutputDim { get; set; }
        [JsonPropertyName("audio_token")] public string AudioToken { get; set; } = "<|AUDIO|>";
        [JsonPropertyName("static_cache_len")] public int StaticCacheLen { get; set; } = 2048;
        [JsonPropertyName("eos_token_ids")] public List<int> EosTokenIds { get; set; } = new();

        /// <summary>Resolve the prefill ONNX file name, falling back to legacy <c>decoder_model</c>.</summary>
        public string ResolvePrefillModel()
            => PrefillModel ?? DecoderModel
               ?? throw new InvalidOperationException("metadata.json missing prefill_model / decoder_model.");

        /// <summary>Resolve the decode ONNX file name, falling back to prefill if shared.</summary>
        public string ResolveDecodeModel()
            => DecodeModel ?? DecoderModel ?? ResolvePrefillModel();
    }
}
