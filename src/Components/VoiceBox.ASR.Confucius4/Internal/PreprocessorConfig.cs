using System.Text.Json.Serialization;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Subset of the bundled <c>preprocessor_config.json</c>. Drives the Whisper-style
    /// log-mel feature extractor used to feed the audio encoder.
    /// </summary>
    internal sealed class PreprocessorConfig
    {
        [JsonPropertyName("feature_size")] public int FeatureSize { get; set; } = 128;
        [JsonPropertyName("sampling_rate")] public int SamplingRate { get; set; } = 16000;
        [JsonPropertyName("hop_length")] public int HopLength { get; set; } = 160;
        [JsonPropertyName("n_fft")] public int NFft { get; set; } = 400;
        [JsonPropertyName("chunk_length")] public int ChunkLengthSec { get; set; } = 30;
        [JsonPropertyName("padding_value")] public float PaddingValue { get; set; } = 0f;
        [JsonPropertyName("mel_filters")] public float[][]? MelFilters { get; set; }

        public int WindowLength => NFft;
        public int FftBins => NFft / 2 + 1;
        public int ChunkLengthSamples => ChunkLengthSec * SamplingRate;
        public int MaxFrames => ChunkLengthSamples / HopLength;
    }
}
