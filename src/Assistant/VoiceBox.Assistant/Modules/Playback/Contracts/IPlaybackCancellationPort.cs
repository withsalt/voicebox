using VoiceBox.Assistant.Modules.Playback.Models;

namespace VoiceBox.Assistant.Modules.Playback.Contracts;

/// <summary>
/// Owner-scoped playback cancellation. Every stop names the turn whose device audio it may remove;
/// a request for any other owner is a no-op so a late decision can never clear a newer reply, and a
/// repeated stop for the same request is a no-op so one candidate produces one physical stop.
/// </summary>
internal interface IPlaybackCancellationPort
{
    /// <summary>Cancels the active request's synthesis without touching device audio.</summary>
    void CancelActivePlayback();

    /// <summary>
    /// Stops and clears device audio of the active request only if it is owned by
    /// <paramref name="expectedOwnerTurnId"/> and has not been stopped yet.
    /// </summary>
    PlaybackStopOutcome TryStopActivePlayback(long expectedOwnerTurnId);

    /// <summary>Temporarily lowers only the named active request's gain; repeated calls preserve its original gain.</summary>
    bool TryDuckActivePlayback(long expectedOwnerTurnId, float gain, float originalGain);

    /// <summary>Restores a false candidate's gain only while the named request still owns playback.</summary>
    bool TryRestoreActivePlayback(long expectedOwnerTurnId);

    /// <summary>The turn that currently owns device audio, or null when no owned request is active.</summary>
    long? ActiveOwnerTurnId { get; }
}
