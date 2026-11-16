namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Turn-scoped reset of the playback text echo guard. The response module calls it when a new
/// response starts; matching stays inside the guard and never crosses this boundary.
/// </summary>
internal interface IPlaybackTextEchoResetPort
{
    void BeginResponse(long turnId);
}
