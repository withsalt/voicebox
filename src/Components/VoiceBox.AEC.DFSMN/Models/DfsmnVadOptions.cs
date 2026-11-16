namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Near-end VAD state-machine settings for the DFSMN mask+VAD model.
    /// All frame counts are expressed in 20 ms DFSMN hops.
    /// </summary>
    public sealed class DfsmnVadOptions
    {
        /// <summary>
        /// Enables near-end VAD eventing. AEC processing remains active when this is false.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Smoothed VAD probability threshold for speech.
        /// </summary>
        public float SpeechThreshold { get; set; } = 0.55f;

        /// <summary>
        /// Number of recent 20 ms probabilities used for moving-average smoothing.
        /// </summary>
        public int SmoothWindowHops { get; set; } = 3;

        /// <summary>
        /// Number of consecutive speech hops required before a speech segment starts.
        /// </summary>
        public int MinSpeechHops { get; set; } = 4;

        /// <summary>
        /// Number of consecutive non-speech hops required before a speech segment ends.
        /// </summary>
        public int MinSilenceHops { get; set; } = 30;

        /// <summary>
        /// Number of hops to include before the detected start. This is bounded by buffered audio.
        /// </summary>
        public int PadStartHops { get; set; } = 3;

        /// <summary>
        /// Extra pre-speech audio retained before the VAD state machine enters speech.
        /// </summary>
        public int PreSpeechPaddingMs { get; set; } = 300;

        /// <summary>
        /// Maximum segment length in hops. When reached, the current segment is closed and a
        /// continuous speech stream starts a new segment on the next speech hop. 0 disables the limit.
        /// </summary>
        public int MaxSpeechHops { get; set; } = 1500;

        /// <summary>
        /// Audio representation emitted via OnSpeaking/OnSpeechEnded.
        /// </summary>
        public DfsmnVadAudioSource EventAudioSource { get; set; } = DfsmnVadAudioSource.ProcessedCapture;

        internal void Validate()
        {
            if (float.IsNaN(SpeechThreshold) || SpeechThreshold < 0f || SpeechThreshold > 1f)
                throw new ArgumentOutOfRangeException(nameof(SpeechThreshold), SpeechThreshold, "SpeechThreshold must be in [0, 1].");
            if (SmoothWindowHops <= 0)
                throw new ArgumentOutOfRangeException(nameof(SmoothWindowHops), SmoothWindowHops, "SmoothWindowHops must be > 0.");
            if (MinSpeechHops <= 0)
                throw new ArgumentOutOfRangeException(nameof(MinSpeechHops), MinSpeechHops, "MinSpeechHops must be > 0.");
            if (MinSilenceHops <= 0)
                throw new ArgumentOutOfRangeException(nameof(MinSilenceHops), MinSilenceHops, "MinSilenceHops must be > 0.");
            if (PadStartHops < 0)
                throw new ArgumentOutOfRangeException(nameof(PadStartHops), PadStartHops, "PadStartHops must be >= 0.");
            if (PreSpeechPaddingMs < 0)
                throw new ArgumentOutOfRangeException(nameof(PreSpeechPaddingMs), PreSpeechPaddingMs, "PreSpeechPaddingMs must be >= 0.");
            if (MaxSpeechHops < 0)
                throw new ArgumentOutOfRangeException(nameof(MaxSpeechHops), MaxSpeechHops, "MaxSpeechHops must be >= 0.");
            if (!Enum.IsDefined(EventAudioSource))
                throw new ArgumentOutOfRangeException(nameof(EventAudioSource), EventAudioSource, "Unsupported VAD audio source.");
        }
    }
}
