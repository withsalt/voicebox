namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderAwareDtdEvidenceDecision(
    bool Accepted,
    bool EchoLike,
    bool PreservedNearEnd,
    bool TrustedEcho,
    bool AcousticPathChange,
    string Reason);
