namespace VoiceBox.Assistant.Api.Configuration;

/// <summary>Adaptive render/capture alignment settings for playback DTD.</summary>
public sealed class VoiceAssistantRenderAlignmentOptions
{
    /// <summary>Maximum age of a primary-AEC delay hint before local estimation takes over.</summary>
    public int DelayHintMaxAgeMs { get; set; } = 500;

    /// <summary>
    /// How long a runtime delay estimate may be extrapolated from its measured drift when
    /// speech content temporarily prevents a fresh correlation observation.
    /// </summary>
    public int EstimateHoldoverMs { get; set; } = 30000;

    public int MaxTrackedDelayMs { get; set; } = 300;
    public int PathChangeThresholdMs { get; set; } = 40;
    public int DelayUpdateIntervalMs { get; set; } = 100;
    public double MaximumDriftPpm { get; set; } = 250;
    public double MaximumDelaySlewPerSecond { get; set; } = 160;
    public double MinimumAlignmentConfidence { get; set; } = 0.55;
}
