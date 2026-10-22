namespace VoiceBox.Assistant.Runtime.Lifecycle.Contracts;

/// <summary>
/// Requests the runtime's ordered stop. Used by fault convergence, which must not re-enter the
/// lifecycle state machine directly.
/// </summary>
internal interface IRuntimeStopPort
{
    Task StopAsync();
}
