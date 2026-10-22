using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Candidate admission surface used by the turn boundary. Speech facts are classified here as
/// either candidate-owned or recognition-bound; the candidate's own detector, lock and buffered
/// audio stay inside the owner.
/// </summary>
internal interface ISpeechCandidatePort
{
    /// <summary>A confirmed barge-in candidate still waiting for the speaker gate verdict.</summary>
    bool IsAwaitingSpeaker { get; }

    /// <summary>A listening or barge-in candidate is still accumulating (not yet confirmed or discarded).</summary>
    bool IsCandidateActive { get; }

    SpeechStartedDecision ObserveSpeechStarted(bool enableBargeIn, long proposedTurnId);

    /// <summary>
    /// Adds capture audio to an active candidate. Returns false when no candidate owns this audio,
    /// which routes it to the recognition path instead.
    /// </summary>
    bool TryObserveCandidateAudio(
        ReadOnlySpan<float> audio,
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId,
        out SpeakingDecision decision);

    SpeakingDecision ResolveSpeakerGate(
        BargeInSpeakerGateStatus speakerGateStatus,
        long proposedTurnId);

    bool TryDiscardOnSpeechEnded(TimeSpan duration, out SpeechEndedDecision decision);

    bool CanMarkConfirmed(SpeechCandidateKind expectedKind);

    bool TryMarkConfirmed(SpeechCandidateKind expectedKind);

    void Clear();
}
