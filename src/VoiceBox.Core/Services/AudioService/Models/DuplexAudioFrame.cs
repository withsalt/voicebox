namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Full-duplex audio callback snapshot containing the render reference and capture
    /// signals around the capture-side processor. Frame memories are valid only for the
    /// duration of the event callback; subscribers that need to retain data must copy it.
    /// </summary>
    public readonly record struct DuplexAudioFrame(
        long SequenceNumber,
        long TimestampTicks,
        uint SampleRate,
        uint Channels,
        uint FrameCount,
        bool CaptureProcessed,
        ReadOnlyMemory<float> RenderReference,
        ReadOnlyMemory<float> CaptureRaw,
        ReadOnlyMemory<float> CaptureBeforeAec,
        ReadOnlyMemory<float> CaptureAfterAec,
        long StreamEpoch = 0,
        long StartSample = 0,
        bool HasDiscontinuity = false,
        DuplexCaptureTimeline Timeline = default,
        RenderSignalTimeline RenderTimeline = default,
        float PlaybackGain = 1f)
    {
        public long EndSampleExclusive => checked(StartSample + (long)FrameCount * Channels);

        public CaptureRange CaptureRange => new(
            StreamEpoch,
            SequenceNumber,
            SequenceNumber,
            StartSample,
            EndSampleExclusive,
            TimestampTicks);

        public CaptureSignalTimeline CaptureRawTimeline =>
            ResolveTimeline(Timeline.Raw, CaptureRaw.Length);

        public CaptureSignalTimeline CaptureBeforeAecTimeline =>
            ResolveTimeline(Timeline.BeforeAec, CaptureBeforeAec.Length);

        public CaptureSignalTimeline CaptureAfterAecTimeline =>
            ResolveTimeline(Timeline.AfterAec, CaptureAfterAec.Length);

        public ReadOnlyMemory<float> CaptureRawValid =>
            CaptureRawTimeline.Slice(CaptureRaw);

        public ReadOnlyMemory<float> CaptureBeforeAecValid =>
            CaptureBeforeAecTimeline.Slice(CaptureBeforeAec);

        public ReadOnlyMemory<float> CaptureAfterAecValid =>
            CaptureAfterAecTimeline.Slice(CaptureAfterAec);

        private CaptureSignalTimeline ResolveTimeline(
            in CaptureSignalTimeline timeline,
            int bufferLength)
            => Timeline.HasAnyTimeline
                ? timeline
                : CaptureSignalTimeline.FromWholeBuffer(CaptureRange, bufferLength);
    }

    public delegate void DuplexAudioFrameHandler(in DuplexAudioFrame frame);
}
