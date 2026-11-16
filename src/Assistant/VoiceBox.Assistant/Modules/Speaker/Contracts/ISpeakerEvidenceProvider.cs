using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Non-blocking identity evidence for one candidate interval. <see cref="Evaluate"/> reads the
/// window cache, schedules only the missing windows and decides immediately;
/// <see cref="WaitForDecisionAsync"/> is the bounded control-plane wait; <see cref="Cancel"/>
/// drops queued candidate jobs without discarding reusable embeddings.
/// </summary>
internal interface ISpeakerEvidenceProvider
{
    SpeakerCandidateId NewCandidateId();

    SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query);

    Task<SpeakerVerificationResult> WaitForDecisionAsync(
        SpeakerEvidenceQuery query,
        TimeSpan timeout,
        CancellationToken cancellationToken = default);

    void Cancel(SpeakerCandidateId candidateId);

    /// <summary>Bounded worker wait a purpose tolerates once its first complete window exists.</summary>
    TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose);
}
