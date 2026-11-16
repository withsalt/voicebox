namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal enum SpeakerVerificationStatus
{
    /// <summary>Evidence is still forming or inference is in flight.</summary>
    Pending,

    /// <summary>Enough valid speech windows scored clearly above threshold.</summary>
    Match,

    /// <summary>Enough valid speech windows scored clearly below threshold.</summary>
    Mismatch,

    /// <summary>The candidate ended without enough owner-verifiable speech.</summary>
    InsufficientEvidence,

    /// <summary>A system, timeline, model or capacity problem prevented a trustworthy verdict.</summary>
    Unavailable,

    /// <summary>The candidate was superseded by a new turn, stop or dispose.</summary>
    Cancelled,
}
