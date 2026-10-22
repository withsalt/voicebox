namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// Native inference boundary used by the streaming service. Keeping lifecycle
    /// coordination outside the ONNX implementation makes shutdown races testable
    /// without loading a model or execution provider.
    /// </summary>
    internal interface IQwen3AsrRuntime : IDisposable
    {
        void Warmup(double audioSeconds = 1.0, int maxNewTokens = 4, CancellationToken ct = default);

        string? BuildStreamingPrefix(
            int chunkId,
            string accumulatedRaw,
            int unfixedChunkNum,
            int unfixedTokenNum);

        Qwen3AsrTranscription Transcribe(
            ReadOnlySpan<float> waveform,
            string context,
            string? language,
            string? prefixText,
            CancellationToken ct,
            IncrementalMelState? melState = null);
    }
}
