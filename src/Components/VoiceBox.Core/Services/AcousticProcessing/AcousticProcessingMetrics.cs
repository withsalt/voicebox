namespace VoiceBox.Core.Services.AcousticProcessing;

/// <summary>
/// AEC/APM 暴露的不可变运行时统计快照。实现可能只填充其中一部分字段。
/// </summary>
public readonly record struct AcousticProcessingMetrics(
    bool HasEchoReturnLoss,
    double EchoReturnLoss,
    bool HasEchoReturnLossEnhancement,
    double EchoReturnLossEnhancement,
    bool HasDelayMs,
    int DelayMs,
    bool HasDelayMedianMs,
    int DelayMedianMs,
    bool HasResidualEchoLikelihood,
    double ResidualEchoLikelihood,
    long TimestampTicks = 0,
    bool HasProcessedCaptureRms = false,
    double ProcessedCaptureRms = 0,
    bool HasInputCaptureRms = false,
    double InputCaptureRms = 0);
