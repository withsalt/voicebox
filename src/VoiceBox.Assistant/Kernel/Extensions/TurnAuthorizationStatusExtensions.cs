using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Kernel.Extensions;

internal static class TurnAuthorizationStatusExtensions
{
    public static bool IsDenied(this TurnAuthorizationStatus status)
        => status is TurnAuthorizationStatus.DeniedMismatch
            or TurnAuthorizationStatus.DeniedInsufficientEvidence
            or TurnAuthorizationStatus.DeniedUnavailable;

    /// <summary>Whether this status permits playback to be stopped and the command to proceed.</summary>
    public static bool PermitsInterruption(this TurnAuthorizationStatus status)
        => status is TurnAuthorizationStatus.NotRequired or TurnAuthorizationStatus.Authorized;
}
