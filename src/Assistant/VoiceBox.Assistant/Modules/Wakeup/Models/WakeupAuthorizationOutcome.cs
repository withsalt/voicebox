namespace VoiceBox.Assistant.Modules.Wakeup.Models;

internal enum WakeupAuthorizationOutcome
{
    /// <summary>The owner was verified on the current keyword candidate.</summary>
    Authorized,

    /// <summary>Evidence is still forming; keep the pending wake event.</summary>
    Waiting,

    /// <summary>Enough valid speech scored clearly below threshold: not the owner.</summary>
    Mismatch,

    /// <summary>The candidate ended without enough owner-verifiable speech.</summary>
    InsufficientEvidence,

    /// <summary>Timeline, capacity, model or timeout problem; the candidate is dropped without an identity verdict.</summary>
    Unavailable,
}
