namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;

internal interface IInterruptionAdmissionProgressView
{
    bool HasSettledAllInterruptionAdmissions { get; }
}
