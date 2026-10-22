namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Capture-callback view of the VAD speaker gate. While a gate candidate is active every VAD chunk
/// has to reach the control loop, where the gate aligns and evaluates it; the ingress therefore
/// suspends its direct ASR feed instead of letting audio bypass the candidate.
/// </summary>
internal interface ISpeakerGateCapturePort
{
    bool HasActiveCandidate { get; }

    /// <summary>Discards the current candidate before the turn boundary admits a new one.</summary>
    void Reset();
}
