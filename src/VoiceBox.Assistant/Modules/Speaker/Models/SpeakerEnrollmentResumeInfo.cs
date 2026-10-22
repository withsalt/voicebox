namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// Where the next <c>EnrollAsync</c> on the same coordinator will continue after a failed attempt.
/// Accepted embeddings stay in memory until the profile is saved or <c>DiscardPendingEnrollment</c> is called.
/// </summary>
public readonly record struct SpeakerEnrollmentResumeInfo(
    SpeakerEnrollmentStage Stage,
    int SegmentIndex,
    int AcceptedCleanSegments,
    int AcceptedDoubleTalkSegments,
    bool CleanConfirmationPassed);
