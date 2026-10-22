namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderDelayEstimate(
    bool HasEstimate,
    double DelaySamples,
    double Confidence,
    bool IsRefined = false,
    bool IsCorrelationFallback = false,
    bool IsExternalHint = false);
