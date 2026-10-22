namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IBackgroundToolResultSource
{
    void BindBackgroundResultSink(IBackgroundToolResultSink sink);
}
