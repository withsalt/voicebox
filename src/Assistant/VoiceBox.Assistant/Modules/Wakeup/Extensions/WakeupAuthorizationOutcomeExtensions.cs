using VoiceBox.Assistant.Modules.Wakeup.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Extensions;

internal static class WakeupAuthorizationOutcomeExtensions
{
    public static bool IsDenied(this WakeupAuthorizationOutcome outcome)
        => outcome is WakeupAuthorizationOutcome.Mismatch
            or WakeupAuthorizationOutcome.InsufficientEvidence
            or WakeupAuthorizationOutcome.Unavailable;
}
