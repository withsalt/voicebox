namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Describes the render reference carried by one duplex callback on its own stream timeline.
    /// </summary>
    public readonly record struct RenderSignalTimeline(
        long StreamEpoch,
        long StartSample,
        long EndSampleExclusive,
        RenderTimelineStatus Status,
        bool HasDiscontinuity = false)
    {
        public long Length => Math.Max(0, EndSampleExclusive - StartSample);

        public bool IsValid =>
            StreamEpoch > 0 &&
            StartSample >= 0 &&
            EndSampleExclusive >= StartSample &&
            Status is RenderTimelineStatus.Valid or
                RenderTimelineStatus.Gap or
                RenderTimelineStatus.Missing &&
            (Status != RenderTimelineStatus.Gap || !HasDiscontinuity);

        public bool IsContiguousWith(in RenderSignalTimeline next)
            => IsValid &&
               next.IsValid &&
               !next.HasDiscontinuity &&
               StreamEpoch == next.StreamEpoch &&
               EndSampleExclusive == next.StartSample;
    }
}
