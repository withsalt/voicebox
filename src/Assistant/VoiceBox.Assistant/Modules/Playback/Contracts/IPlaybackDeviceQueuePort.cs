namespace VoiceBox.Assistant.Modules.Playback.Contracts;

/// <summary>Device controls required by the logical playback owner during interruption.</summary>
internal interface IPlaybackDeviceQueuePort
{
    void ClearQueue();

    /// <summary>Updates the gain target synchronously, without calling back into the playback owner.</summary>
    void SetGain(float gain);
}
