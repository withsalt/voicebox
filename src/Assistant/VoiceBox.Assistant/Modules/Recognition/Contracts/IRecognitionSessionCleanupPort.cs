namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>Cleans up a provider session after no turn owns it.</summary>
internal interface IRecognitionSessionCleanupPort
{
    Task CleanupAbandonedSessionAsync(string sessionId);
}
