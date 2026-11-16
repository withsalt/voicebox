namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Capture audio emitted after the asynchronous record-processing chain.
    /// The source range keeps the output aligned to the original PortAudio callback
    /// timeline even when a streaming processor buffers or coalesces samples.
    /// Audio memory is valid only for the duration of the event callback.
    /// </summary>
    public readonly record struct ProcessedCaptureFrame(
        CaptureRange SourceRange,
        uint SampleRate,
        uint Channels,
        ReadOnlyMemory<float> Audio,
        int DelaySamples,
        bool HasDiscontinuity)
    {
        public int SampleCount => Audio.Length;
    }

    public delegate void ProcessedCaptureFrameHandler(in ProcessedCaptureFrame frame);
}
