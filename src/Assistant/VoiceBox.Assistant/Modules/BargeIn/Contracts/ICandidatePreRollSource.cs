namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Hands candidate pre-roll audio to the recognition session that takes the turn over. It exposes
/// only the transfer, never the candidate's thresholds or decision state.
/// </summary>
internal interface ICandidatePreRollSource
{
    void DrainPendingAudioInto(List<float[]> destination);
}
