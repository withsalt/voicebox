namespace VoiceBox.Assistant.Modules.BargeIn.Models
{
    /// <summary>
    /// Immutable temporal guards used to aggregate render-aware DTD observations.
    /// </summary>
    internal readonly record struct BargeInDecisionParameters(
        double MinNearendFraction,
        double MaxEchoFraction,
        double MinContiguousNearendFraction)
    {
        public static BargeInDecisionParameters Default { get; } = new(
            MinNearendFraction: 0.58,
            MaxEchoFraction: 1.0 / 3.0,
            // Require about 60 ms (three default hops) of continuous speech; brief gaps are OK.
            MinContiguousNearendFraction: 1.0 / 3.0);
    }
}
