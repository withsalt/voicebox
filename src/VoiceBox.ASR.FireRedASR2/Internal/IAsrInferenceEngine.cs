using VoiceBox.ASR.FireRedASR2.Internal.Model;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Pure inference behaviour over the encoder / decoder / CTC ONNX graphs. Does not own
    /// metadata or sessions — those live in <see cref="AsrModelMetadata"/> and
    /// <see cref="AsrInferenceSessions"/> respectively.
    /// </summary>
    internal interface IAsrInferenceEngine : IDisposable
    {
        AsrModelMetadata Metadata { get; }

        bool HasDecoder { get; }

        bool HasCtc { get; }

        EncoderOutputEntity RunEncoder(IReadOnlyList<AsrInputEntity> modelInputs);

        /// <summary>
        /// Run one decoder step with batch=1. <paramref name="tokens"/> is the live hypothesis
        /// buffer; only the first <paramref name="tokensLen"/> elements are used (and never
        /// mutated). Callers may reuse the same backing array across steps to avoid the
        /// legacy O(maxLen²) per-step copy.
        /// </summary>
        DecoderOutputEntity RunDecoderStep(
            long[] tokens,
            int tokensLen,
            float[] encoderOutputs,
            bool[] srcMask,
            IReadOnlyList<float[]> cacheList);

        CtcOutputEntity RunCtc(float[] encoderOutputs, int batchSize = 1);
    }
}
