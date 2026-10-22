namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IBackgroundToolResultSink
{
    Task PublishAsync(long turnId, string text, CancellationToken cancellationToken);
}
