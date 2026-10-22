namespace VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

/// <summary>
/// Marks the synchronous VAD observation that replays audio already authorized by the initial
/// wake-word speaker gate. The control loop consumes the marker when it admits the corresponding
/// speech start, so later speech candidates still run the normal SingleTalk verification.
/// </summary>
internal interface IWakeupReplayAuthorizationPort
{
    void BeginWakeupReplay();

    void EndWakeupReplay();

    bool TryConsumeWakeupReplayAuthorization(long admissionEpoch);
}
