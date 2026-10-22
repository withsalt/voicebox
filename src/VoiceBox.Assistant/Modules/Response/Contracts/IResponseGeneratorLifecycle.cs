namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IResponseGeneratorLifecycle
{
    Task StartAsync(CancellationToken cancellationToken);

    void CancelBackgroundOperations();

    Task StopAsync();
}
