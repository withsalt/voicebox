namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IRenderAwareFaultSink
{
    void ReportFatalDetectorFault(Exception exception);
}
