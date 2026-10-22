namespace VoiceBox.Assistant.Kernel.Models;

internal readonly record struct BargeInDecisionDiagnostics(
    int FrameCount,
    int AcceptedFrameCount,
    int RejectedFrameCount,
    int EchoLikeFrameCount,
    double AcceptedDurationMs,
    double RequiredDurationMs,
    double ScoreDurationMs,
    double PeakScoreDurationMs,
    double LastFrameRms,
    double LastRequiredRms,
    double MinRequiredRms,
    double MaxRequiredRms,
    string LastRejectReason);
