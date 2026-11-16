namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IBackgroundResponsePlaybackPort
{
    Task PlayAsync(string text, CancellationToken cancellationToken);
}
