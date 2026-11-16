using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Core.Services.VadService.Models
{
    /// <summary>
    /// Per-frame VAD probability observation emitted before segment-level speech events.
    /// </summary>
    public sealed class VadProbabilityEventArgs : EventArgs
    {
        public VadProbabilityEventArgs(
            float probability,
            float smoothedProbability,
            bool isSpeech,
            long frameIndex,
            TimeSpan timestamp,
            TimeSpan frameDuration,
            CaptureRange sourceRange = default)
        {
            if (frameIndex <= 0)
                throw new ArgumentOutOfRangeException(nameof(frameIndex), frameIndex, "Frame index must be 1-based.");
            if (timestamp < TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(timestamp), timestamp, "Timestamp must be >= 0.");
            if (frameDuration <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(frameDuration), frameDuration, "Frame duration must be > 0.");

            Probability = NormalizeProbability(probability);
            SmoothedProbability = NormalizeProbability(smoothedProbability);
            IsSpeech = isSpeech;
            FrameIndex = frameIndex;
            Timestamp = timestamp;
            FrameDuration = frameDuration;
            SourceRange = sourceRange;
        }

        public float Probability { get; }

        public float SmoothedProbability { get; }

        public bool IsSpeech { get; }

        public long FrameIndex { get; }

        public TimeSpan Timestamp { get; }

        public TimeSpan FrameDuration { get; }

        /// <summary>
        /// Capture interval that produced this probability, when the source has timeline data.
        /// </summary>
        public CaptureRange SourceRange { get; }

        private static float NormalizeProbability(float value)
            => float.IsFinite(value) ? Math.Clamp(value, 0f, 1f) : 0f;
    }
}
