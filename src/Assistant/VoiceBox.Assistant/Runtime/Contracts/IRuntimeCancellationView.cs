namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IRuntimeCancellationView
{
    CancellationToken RunToken { get; }
}
