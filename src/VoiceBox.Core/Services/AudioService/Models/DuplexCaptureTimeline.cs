namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Source-timeline metadata for the capture signals carried by a duplex frame.
    /// A buffered processor may expose only a suffix of its output buffer as valid source audio.
    /// </summary>
    public readonly record struct DuplexCaptureTimeline(
        CaptureSignalTimeline Raw,
        CaptureSignalTimeline BeforeAec,
        CaptureSignalTimeline AfterAec)
    {
        public bool HasAnyTimeline => Raw.IsValid || BeforeAec.IsValid || AfterAec.IsValid;

        public static DuplexCaptureTimeline Unbuffered(
            in CaptureRange sourceRange,
            int bufferLength)
        {
            CaptureSignalTimeline signal =
                CaptureSignalTimeline.FromWholeBuffer(in sourceRange, bufferLength);
            return new DuplexCaptureTimeline(signal, signal, signal);
        }
    }
}
