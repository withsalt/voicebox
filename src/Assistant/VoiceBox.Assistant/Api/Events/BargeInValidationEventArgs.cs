namespace VoiceBox.Assistant.Api.Events;

public sealed class BargeInValidationEventArgs : EventArgs
{
    public required BargeInValidationKind Kind { get; init; }
    public string? Text { get; init; }
    public string? Reason { get; init; }
    public BargeInTriggerSource Source { get; init; } = BargeInTriggerSource.Vad;
    public int AudioFrameCount { get; init; }
    public int AudioAcceptedFrameCount { get; init; }
    public int AudioEchoLikeFrameCount { get; init; }
    public double AudioAcceptedDurationMs { get; init; }
    public double AudioScoreDurationMs { get; init; }
    public double AudioPeakScoreDurationMs { get; init; }
}
