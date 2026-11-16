namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Model-gated near-end protection. The DFSMN mask+VAD model emits a per-hop near-end speech
    /// probability from the same inference that produces the suppression mask; these options use
    /// that probability to bound how hard the pipeline may suppress while the model itself says
    /// "a near-end talker is speaking over the echo".
    ///
    /// Without this, the PSM mask plus the scalar residual suppressor routinely remove the
    /// barging-in user together with the echo (the known double-talk weakness of mask-based AEC),
    /// which makes TTS-playback interruption impossible downstream. With it, echo-only hops keep
    /// the exact legacy behavior, while confirmed near-end hops keep the user audible.
    ///
    /// All thresholds are probability-domain (device independent); only time constants are in ms.
    /// Requires a model that exposes the VAD probability output; on mask-only models the
    /// protection silently disables itself.
    /// </summary>
    public sealed class DfsmnAecNearEndProtectionOptions
    {
        /// <summary>Enables model-gated near-end protection. Default true.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Maximum spectral attenuation (dB) any mask bin may apply while near-end speech is fully
        /// confirmed (probability at/above <see cref="FullProbability"/>). Between engage and full
        /// probability the bound relaxes proportionally from <see cref="DfsmnAecOptions.MaskFloorDb"/>.
        /// 9 dB keeps the talker clearly audible while still shaping echo bins. Default 9.
        /// </summary>
        public float MaxSuppressionDb { get; set; } = 9f;

        /// <summary>Near-end probability where protection starts to engage. Default 0.35.</summary>
        public float EngageProbability { get; set; } = 0.35f;

        /// <summary>Near-end probability where protection reaches full strength. Default 0.65.</summary>
        public float FullProbability { get; set; } = 0.65f;

        /// <summary>
        /// Release time constant for the protection envelope. Attack is instantaneous (the
        /// probability comes from the same hop as the mask it gates), release is smoothed so word
        /// gaps inside one utterance do not slam the suppression back to full. Default 240 ms.
        /// </summary>
        public float ReleaseMs { get; set; } = 240f;

        /// <summary>
        /// Also gate the scalar residual echo suppressor: while near-end speech is confirmed its
        /// extra attenuation backs off proportionally, and it recovers when near-end speech ends instead of
        /// waiting out the slow release. Default true.
        /// </summary>
        public bool GateResidualSuppressor { get; set; } = true;

        /// <summary>
        /// Freeze linear-AEC adaptation while the model reports near-end speech, complementing the
        /// energy-based double-talk detector (which needs the mic to out-power the expected echo
        /// by ~6 dB and therefore misses soft talkers over loud playback). Default true.
        /// </summary>
        public bool FreezeAdaptation { get; set; } = true;

        /// <summary>
        /// Near-end probability that triggers the adaptation freeze. Kept just above
        /// <see cref="EngageProbability"/> so there is no band where the mask keeps the talker
        /// audible while NLMS still adapts into (and the DTD coupling absorbs) that same voice.
        /// The freeze also persists while the protection envelope is engaged (release hangover).
        /// Default 0.40.
        /// </summary>
        public float FreezeProbability { get; set; } = 0.40f;

        internal void Validate()
        {
            if (MaxSuppressionDb < 0 || float.IsNaN(MaxSuppressionDb))
                throw new ArgumentOutOfRangeException(nameof(MaxSuppressionDb), MaxSuppressionDb, "MaxSuppressionDb must be >= 0.");
            if (EngageProbability < 0 || EngageProbability > 1 || float.IsNaN(EngageProbability))
                throw new ArgumentOutOfRangeException(nameof(EngageProbability), EngageProbability, "EngageProbability must be in [0, 1].");
            if (FullProbability <= EngageProbability || FullProbability > 1 || float.IsNaN(FullProbability))
                throw new ArgumentOutOfRangeException(nameof(FullProbability), FullProbability, "FullProbability must be in (EngageProbability, 1].");
            if (ReleaseMs <= 0 || float.IsNaN(ReleaseMs))
                throw new ArgumentOutOfRangeException(nameof(ReleaseMs), ReleaseMs, "ReleaseMs must be positive.");
            if (FreezeProbability < 0 || FreezeProbability > 1 || float.IsNaN(FreezeProbability))
                throw new ArgumentOutOfRangeException(nameof(FreezeProbability), FreezeProbability, "FreezeProbability must be in [0, 1].");
        }
    }
}
