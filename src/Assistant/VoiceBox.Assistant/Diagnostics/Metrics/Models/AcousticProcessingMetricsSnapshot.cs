namespace VoiceBox.Assistant.Diagnostics.Metrics.Models;

/// <summary>
/// Owned, immutable acoustic diagnostics snapshot. It carries only normalized values; no provider
/// DTO, handle or mutable reference crosses this boundary.
/// </summary>
internal readonly record struct AcousticProcessingMetricsSnapshot(
    long TimestampTicks,
    bool HasDelayMs,
    int DelayMs,
    bool HasDelayMedianMs,
    int DelayMedianMs,
    bool HasResidualEchoLikelihood,
    double ResidualEchoLikelihood)
{
    public AcousticProcessingMetricsSnapshot WithTimestamp(long timestampTicks)
        => this with { TimestampTicks = timestampTicks };
}
