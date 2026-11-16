namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderAwareDtdCausalEvidence(
    double ModelSupport,
    double EchoSupport,
    double PreservedSupport,
    double Score,
    bool TrustedEcho,
    bool AcousticPathChange);
