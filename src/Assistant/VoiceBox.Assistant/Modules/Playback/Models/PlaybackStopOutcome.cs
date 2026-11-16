namespace VoiceBox.Assistant.Modules.Playback.Models;

/// <summary>Result of an owner-scoped stop request.</summary>
internal enum PlaybackStopOutcome
{
    /// <summary>Nothing owned by the expected turn is active (never started, finished, or a newer owner plays).</summary>
    NotOwned = 0,
    /// <summary>This call cancelled the active request and removed its device audio.</summary>
    Stopped = 1,
    /// <summary>The active request of that owner was already stopped by an earlier call; nothing was done.</summary>
    AlreadyStopped = 2,
}
