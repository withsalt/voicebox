namespace VoiceBox.Assistant.Runtime.Contracts;

/// <summary>Read-only run state shared by capture, render and near-end processing.</summary>
internal interface IRuntimeLivenessView
{
    bool IsStopping { get; }
}
