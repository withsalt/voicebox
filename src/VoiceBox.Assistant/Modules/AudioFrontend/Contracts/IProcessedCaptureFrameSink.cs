using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

internal interface IProcessedCaptureFrameSink
{
    void OnProcessedCapture(in ProcessedCaptureFrame frame);
}
