namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Lifecycle/reset operations; no PCM or confirmation state is exposed.</summary>
internal interface INearEndResetPort
{
    void Reset();

    /// <summary>
    /// Playback-start boundary. <paramref name="playbackTurnId"/> identifies the turn whose TTS
    /// just started; zero is the unconditional reset used by tests and lifecycle teardown.
    /// </summary>
    void ClearAudioReservoirs(long playbackTurnId = 0);
}
