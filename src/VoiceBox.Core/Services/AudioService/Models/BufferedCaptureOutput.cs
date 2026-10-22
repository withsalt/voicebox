namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Describes the source-backed portion of a buffered processor output.
    /// Any leading samples before <see cref="BufferOffset"/> are synthetic warm-up output.
    /// </summary>
    /// <param name="Timeline">Source range represented by the valid output suffix.</param>
    /// <param name="HasDiscontinuity">
    /// True on a reset frame and on the first subsequent source-backed output.
    /// </param>
    /// <param name="MappingReset">
    /// True only on the input callback that reset mapper state. Duplex publishers use this
    /// immediate signal; processed-output publishers use <paramref name="HasDiscontinuity"/>.
    /// </param>
    public readonly record struct BufferedCaptureOutput(
        CaptureSignalTimeline Timeline,
        bool HasDiscontinuity,
        bool MappingReset = false)
    {
        public int BufferOffset => Timeline.BufferOffset;
        public int SampleCount => Timeline.SampleCount;
        public CaptureRange SourceRange => Timeline.SourceRange;
        public bool IsValid => Timeline.IsValid;
    }
}
