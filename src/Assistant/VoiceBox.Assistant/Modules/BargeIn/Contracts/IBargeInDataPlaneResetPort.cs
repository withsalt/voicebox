namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Wider reset used when a wake word takes the turn away from playback: it clears the detector's
/// decision state and the decision journal together.
/// </summary>
internal interface IBargeInDataPlaneResetPort : IBargeInDecisionResetPort
{
    void ResetAll();
}
