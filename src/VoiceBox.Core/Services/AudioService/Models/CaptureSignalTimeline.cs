namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Maps the valid portion of one signal buffer to its source capture interval.
    /// Buffer offsets and sample counts refer to interleaved samples.
    /// </summary>
    public readonly record struct CaptureSignalTimeline(
        CaptureRange SourceRange,
        int BufferOffset,
        int SampleCount)
    {
        public bool IsValid =>
            SourceRange.IsValid &&
            BufferOffset >= 0 &&
            SampleCount > 0 &&
            BufferOffset <= int.MaxValue - SampleCount &&
            SourceRange.Length == SampleCount;

        public int BufferEndExclusive => checked(BufferOffset + SampleCount);

        public ReadOnlyMemory<float> Slice(ReadOnlyMemory<float> buffer)
            => IsValid && BufferEndExclusive <= buffer.Length
                ? buffer.Slice(BufferOffset, SampleCount)
                : ReadOnlyMemory<float>.Empty;

        public static CaptureSignalTimeline FromWholeBuffer(
            in CaptureRange sourceRange,
            int bufferLength)
            => sourceRange.IsValid &&
               bufferLength > 0 &&
               sourceRange.Length == bufferLength
                ? new CaptureSignalTimeline(sourceRange, 0, bufferLength)
                : default;
    }

}
