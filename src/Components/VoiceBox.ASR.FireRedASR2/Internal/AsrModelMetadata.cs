namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Immutable description of a loaded ASR model: shape constants, special token ids and
    /// the parsed encoder metadata. Inference engines read from this; nothing mutates it
    /// after load.
    /// </summary>
    internal sealed record AsrModelMetadata
    {
        public required int BlankId { get; init; }
        public required int UnkId { get; init; }
        public required int PadId { get; init; }
        public required int SosId { get; init; }
        public required int EosId { get; init; }

        public required int FeatureDim { get; init; }
        public required int SampleRate { get; init; }

        public required int ChunkLength { get; init; }
        public required int ShiftLength { get; init; }
        public required int RequiredCacheSize { get; init; }

        public required int EncoderHiddenDim { get; init; }
        public required int DecoderCacheLayers { get; init; }

        /// <summary>
        /// Seconds per encoder output frame. Used by CTC forced alignment to map frame
        /// indices to timestamps. Equals <c>FbankFrameShiftMs * SubsamplingRate / 1000</c>.
        /// </summary>
        public required double EncoderFrameShiftSeconds { get; init; }
    }
}
