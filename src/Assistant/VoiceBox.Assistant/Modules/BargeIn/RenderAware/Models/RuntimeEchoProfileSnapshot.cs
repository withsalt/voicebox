namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RuntimeEchoProfileSnapshot(
    double Reliability,
    double TrustedEchoDurationMs,
    double RelevantEchoDurationMs,
    double PredictedCaptureRms,
    double PredictedPostAecRms,
    double PredictedResidualRatio,
    float PureEchoProbabilityUpper,
    float PureEchoSmoothedProbabilityUpper)
{
    public bool IsReliable => Reliability >= 0.75;
}
