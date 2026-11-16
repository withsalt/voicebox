namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderAlignmentUpdate(
    bool IsAligned,
    bool PathChanged,
    double DelaySamples,
    double DriftPpm,
    double Confidence,
    long Generation);
