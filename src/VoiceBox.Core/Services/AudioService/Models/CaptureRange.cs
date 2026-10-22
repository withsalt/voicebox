namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Identifies an interval on one uninterrupted capture stream timeline.
    /// Sample positions count interleaved samples and use a half-open [start, end) range.
    /// </summary>
    public readonly record struct CaptureRange(
        long StreamEpoch,
        long StartSequenceNumber,
        long EndSequenceNumber,
        long StartSample,
        long EndSampleExclusive,
        long TimestampTicks)
    {
        public long Length => Math.Max(0, EndSampleExclusive - StartSample);

        public bool IsValid =>
            StreamEpoch > 0 &&
            StartSequenceNumber >= 0 &&
            EndSequenceNumber >= StartSequenceNumber &&
            StartSample >= 0 &&
            EndSampleExclusive >= StartSample;

        public bool IsContiguousWith(in CaptureRange next)
            => IsValid &&
               next.IsValid &&
               StreamEpoch == next.StreamEpoch &&
               next.StartSequenceNumber >= EndSequenceNumber &&
               next.StartSequenceNumber <= EndSequenceNumber + 1 &&
               EndSampleExclusive == next.StartSample;
    }
}
