using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>Optional processed capture timeline exposed by a capable audio frontend.</summary>
internal interface IProcessedCaptureSource
{
    event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;
}
