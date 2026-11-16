namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Double-talk detector (DTD) settings for <see cref="DfsmnAecService"/>.
    ///
    /// When near-end (user) speech is detected on top of the far-end echo, the built-in linear
    /// echo canceller's coefficient adaptation is frozen for that hop. Without this, a barging-in
    /// talker is treated as filter error and drags the adaptive filter — and therefore the echo
    /// estimate feeding the downstream neural mask — off the true echo path, which makes the mask
    /// suppress the near-end voice. Freezing during double-talk preserves the converged filter so
    /// the user's interruption survives to the VAD/ASR.
    ///
    /// On echo-only audio the detector never flags, so steady-state echo cancellation is identical
    /// to having no detector at all.
    /// </summary>
    public sealed class DfsmnAecDoubleTalkOptions
    {
        /// <summary>Freeze adaptation during detected double-talk. Default true.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Far-end RMS below which the render reference is treated as silent: with no echo there
        /// is nothing to mistake for double-talk and the canceller is left free to adapt.
        /// Default 1e-3 (~ -60 dBFS).
        /// </summary>
        public float RenderActivityThreshold { get; set; } = 1e-3f;

        /// <summary>
        /// Mic-to-expected-echo RMS ratio above which a hop is declared double-talk. 2.0 means the
        /// microphone must carry ~6 dB more energy than the learned echo coupling alone explains.
        /// Lower = more sensitive (freezes sooner, protects the near-end harder, but lets more echo
        /// leak); higher = more conservative. Tune by ear.
        /// </summary>
        public float NearEndThreshold { get; set; } = 2.0f;

        /// <summary>
        /// EMA factor (0..1) used to track the render→mic echo coupling on echo-only hops. Closer
        /// to 1 tracks more slowly / more stably. Default 0.95 (~0.4 s time constant at 20 ms hops).
        /// </summary>
        public float CouplingSmoothing { get; set; } = 0.95f;

        /// <summary>
        /// Keep the adaptation freeze active this many milliseconds after the last double-talk hop,
        /// so brief gaps between words do not unfreeze the filter mid-interruption. Default 120 ms.
        /// </summary>
        public float HangoverMs { get; set; } = 120f;

        internal void Validate()
        {
            if (RenderActivityThreshold < 0 || float.IsNaN(RenderActivityThreshold))
                throw new ArgumentOutOfRangeException(nameof(RenderActivityThreshold), RenderActivityThreshold, "RenderActivityThreshold must be >= 0.");
            if (NearEndThreshold <= 0 || float.IsNaN(NearEndThreshold))
                throw new ArgumentOutOfRangeException(nameof(NearEndThreshold), NearEndThreshold, "NearEndThreshold must be > 0.");
            if (CouplingSmoothing < 0 || CouplingSmoothing >= 1 || float.IsNaN(CouplingSmoothing))
                throw new ArgumentOutOfRangeException(nameof(CouplingSmoothing), CouplingSmoothing, "CouplingSmoothing must be in [0, 1).");
            if (HangoverMs < 0 || float.IsNaN(HangoverMs))
                throw new ArgumentOutOfRangeException(nameof(HangoverMs), HangoverMs, "HangoverMs must be >= 0.");
        }
    }
}
