namespace VoiceBox.Core.Services.DNRService
{
    /// <summary>
    /// Stateful mono streaming noise reduction. <see cref="Process"/> consumes every input
    /// sample and returns only output backed by the oldest unconsumed samples from the same
    /// stream. Startup padding and other synthetic warm-up output must never be returned.
    /// </summary>
    public interface IDigitalNoiseReductionService : IDisposable
    {
        /// <summary>Preferred processing frame size in configured input-rate samples.</summary>
        int FrameSize { get; }

        /// <summary>
        /// Nominal number of configured input-rate samples retained before source-backed output
        /// becomes available. This is a backlog bound, not an instruction to shift source ranges.
        /// </summary>
        int DelaySamples { get; }

        /// <summary>
        /// Consumes all samples in <paramref name="buffer"/> and writes zero or more oldest
        /// source-backed output samples to its prefix.
        /// </summary>
        int Process(Span<float> buffer);

        /// <summary>
        /// Discards every buffered sample and all model/resampler state without producing output.
        /// The next <see cref="Process"/> call starts an independent stream.
        /// </summary>
        void Reset();
    }
}
