namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    /// <summary>
    /// Model metadata parsed from the encoder ONNX file. Keys match the manyeyes
    /// FireRed export convention so the names cannot be freely changed.
    /// </summary>
    internal sealed class CustomMetadata
    {
        public string? OutputDir { get; set; } = string.Empty;

        public int OutputSize { get; set; } = 512;

        public string? OnnxInfer { get; set; }

        public int LeftChunks { get; set; } = -1;

        public int Batch { get; set; } = 1;

        public float ReverseWeight { get; set; } = 0.5f;

        public int ChunkSize { get; set; }

        public int NumBlocks { get; set; }

        public int CnnModuleKernel { get; set; }

        public int Head { get; set; } = 8;

        public string? Decoder { get; set; } = "transformer";

        public int FeatureSize { get; set; }

        public int VocabSize { get; set; }

        public int DecodingWindow { get; set; }

        public string? Encoder { get; set; } = "conformer";

        public int SubsamplingRate { get; set; } = 4;

        public int RightContext { get; set; }

        public bool IsBidirectionalDecoder { get; set; } = true;

        public int EosSymbol { get; set; }

        public int SosSymbol { get; set; }
    }
}
