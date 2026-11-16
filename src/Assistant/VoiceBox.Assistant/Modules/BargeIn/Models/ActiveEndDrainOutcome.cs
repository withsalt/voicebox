namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>Immutable result of one completed near-end end-drain.</summary>
internal readonly record struct ActiveEndDrainOutcome(
    string Reason,
    long StreamEpoch,
    long TargetSample,
    long FedThroughSample,
    long MissingSamples);
