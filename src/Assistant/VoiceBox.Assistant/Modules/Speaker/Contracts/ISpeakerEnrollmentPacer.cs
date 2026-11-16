using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Interactive pacing for enrollment. The coordinator calls <see cref="WaitForSegmentStartAsync"/>
/// before every segment attempt (clean, double-talk and each confirmation) so the host can show the
/// prompt and wait for the user; collection and the segment timeout start only after it returns.
/// Hosts that record back-to-back pass no pacer.
/// </summary>
public interface ISpeakerEnrollmentPacer
{
    /// <param name="segment">
    /// The announced attempt: <see cref="SpeakerEnrollmentProgressEventArgs.Stage"/>,
    /// <see cref="SpeakerEnrollmentProgressEventArgs.Condition"/>, segment index/count and attempt.
    /// </param>
    Task WaitForSegmentStartAsync(SpeakerEnrollmentProgressEventArgs segment, CancellationToken cancellationToken);
}
