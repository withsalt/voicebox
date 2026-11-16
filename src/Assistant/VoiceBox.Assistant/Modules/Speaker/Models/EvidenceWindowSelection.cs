namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// Hop-aligned windows chosen for one verification decision. <see cref="WindowCount"/> is the
/// number of complete windows already covered by retained audio; <see cref="RequiredEvidenceEnd"/>
/// is the sample at which the full <c>WindowCount</c> geometry would be complete.
/// </summary>
internal readonly record struct EvidenceWindowSelection(
    long FirstWindowStart,
    int WindowCount,
    int AvailableCandidateWindowCount,
    int RetainedCandidateWindowCount,
    long RequiredEvidenceEnd);
