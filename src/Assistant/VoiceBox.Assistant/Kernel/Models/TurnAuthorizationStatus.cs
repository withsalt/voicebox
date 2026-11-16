namespace VoiceBox.Assistant.Kernel.Models;

/// <summary>
/// Identity authorization of the current interruption candidate. It is orthogonal to the public
/// conversation state: a pending candidate overlays the still-playing owner turn and never changes
/// what the user hears until it becomes <see cref="Authorized"/>.
/// </summary>
internal enum TurnAuthorizationStatus
{
    /// <summary>KWS is disabled: no identity gate applies and an acoustic interrupt may stop playback directly.</summary>
    NotRequired,

    /// <summary>KWS is enabled and the current candidate is waiting for its own speaker verdict.</summary>
    PendingSpeaker,

    /// <summary>The current candidate's speaker verdict was Match.</summary>
    Authorized,

    DeniedMismatch,

    DeniedInsufficientEvidence,

    DeniedUnavailable,
}
