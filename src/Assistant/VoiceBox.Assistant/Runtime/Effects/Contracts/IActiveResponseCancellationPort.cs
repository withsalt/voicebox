namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

/// <summary>Runtime-owned cancellation of the active response CTS.</summary>
internal interface IActiveResponseCancellationPort
{
    void CancelActiveResponse();
}
