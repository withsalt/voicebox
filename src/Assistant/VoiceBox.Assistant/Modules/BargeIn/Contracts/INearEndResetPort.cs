namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Lifecycle/reset operations; no PCM or confirmation state is exposed.</summary>
internal interface INearEndResetPort
{
    void Reset();

    void ClearAudioReservoirs();
}
