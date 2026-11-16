namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IRuntimeFaultPublicationGate
{
    bool TryClaim(out Exception? fatalException);
}
