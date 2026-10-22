namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>Configuration for the scalar residual echo suppressor.</summary>
    public sealed class DfsmnAecResidualOptions
    {
        /// <summary>
        /// Enables a conservative scalar residual-echo suppressor after the neural mask/linear AEC stage.
        /// It is driven by the linear echo estimate and mainly targets far-end tails and room decay.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Maximum additional attenuation applied by the residual suppressor.</summary>
        public float EchoSuppressionDb { get; set; } = 18f;

        /// <summary>
        /// Suppression starts when estimated echo energy exceeds residual output energy by this ratio.
        /// Larger values are safer for double-talk; smaller values suppress more echo.
        /// </summary>
        public float EchoSuppressionRatio { get; set; } = 1.5f;

        /// <summary>How quickly the suppressor enters attenuation, in milliseconds.</summary>
        public float AttackMs { get; set; } = 20f;

        /// <summary>How slowly the suppressor releases attenuation, in milliseconds.</summary>
        public float ReleaseMs { get; set; } = 160f;

        internal void Validate()
        {
            if (EchoSuppressionDb < 0 || float.IsNaN(EchoSuppressionDb))
                throw new ArgumentOutOfRangeException(nameof(EchoSuppressionDb), EchoSuppressionDb, "EchoSuppressionDb must be >= 0.");
            if (EchoSuppressionRatio <= 0 || float.IsNaN(EchoSuppressionRatio))
                throw new ArgumentOutOfRangeException(nameof(EchoSuppressionRatio), EchoSuppressionRatio, "EchoSuppressionRatio must be positive.");
            if (AttackMs <= 0 || float.IsNaN(AttackMs))
                throw new ArgumentOutOfRangeException(nameof(AttackMs), AttackMs, "AttackMs must be positive.");
            if (ReleaseMs <= 0 || float.IsNaN(ReleaseMs))
                throw new ArgumentOutOfRangeException(nameof(ReleaseMs), ReleaseMs, "ReleaseMs must be positive.");
        }
    }
}
