namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>Configuration for the built-in NLMS linear echo canceller.</summary>
    public sealed class DfsmnAecLinearOptions
    {
        /// <summary>Length of the adaptive filter used by the built-in linear echo canceller.</summary>
        public int FilterLength { get; set; } = 4096;

        /// <summary>Normalized LMS step size. Larger values can be less stable.</summary>
        public float AdaptationStep { get; set; } = 0.12f;

        /// <summary>Leak factor applied to the adaptive filter coefficients once per hop.</summary>
        public float Leakage { get; set; } = 0.9998f;

        /// <summary>Small positive constant used in the NLMS denominator.</summary>
        public float Epsilon { get; set; } = 1e-6f;

        internal void Validate()
        {
            if (FilterLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(FilterLength), FilterLength, "FilterLength must be positive.");
            if (AdaptationStep < 0 || float.IsNaN(AdaptationStep))
                throw new ArgumentOutOfRangeException(nameof(AdaptationStep), AdaptationStep, "AdaptationStep must be >= 0.");
            if (Leakage <= 0 || Leakage > 1 || float.IsNaN(Leakage))
                throw new ArgumentOutOfRangeException(nameof(Leakage), Leakage, "Leakage must be in (0, 1].");
            if (Epsilon <= 0 || float.IsNaN(Epsilon))
                throw new ArgumentOutOfRangeException(nameof(Epsilon), Epsilon, "Epsilon must be positive.");
        }
    }
}
