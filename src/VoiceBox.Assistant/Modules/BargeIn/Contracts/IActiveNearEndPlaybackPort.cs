namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Playback facts and owner-scoped temporary gain controls. Only candidates with identity
/// confirmation disabled may lower gain; speaker candidates leave playback untouched.
/// </summary>
internal interface IActiveNearEndPlaybackPort
{
    bool IsPlaybackActive { get; }

    /// <summary>The turn whose reply currently owns device audio, or null.</summary>
    long? ActiveOwnerTurnId { get; }

    bool TryDuckPlayback(long ownerTurnId, float gain, float originalGain);

    bool TryRestorePlayback(long ownerTurnId);
}
