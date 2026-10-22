namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct ActiveNearEndDiagnosticSnapshot(
    bool CandidateActive,
    bool CandidateAuthorized,
    long CandidateDecisionEndSample,
    long MainAudioEndSample,
    long ActiveTurnId,
    string LastCandidateOutcome,
    bool ConfirmationPending,
    long CandidateId = 0,
    bool DtdInterruptObserved = false);
