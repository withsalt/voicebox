namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;

internal interface ICapturePositionView
{
    long CurrentCaptureSamplePosition { get; }
}
