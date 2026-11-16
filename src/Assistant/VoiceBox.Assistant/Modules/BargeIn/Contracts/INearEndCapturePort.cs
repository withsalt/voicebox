using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Primary-capture operations used by the hot capture path.</summary>
internal interface INearEndCapturePort
{
    /// <summary>Holds end finalization until VAD and its matching recognition feed complete.</summary>
    void BeginEndpointObservation();

    /// <summary>Releases the observation hold. Must pair with Begin in a finally block.</summary>
    void CompleteEndpointObservation();

    MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame);

    void FeedFromRecognitionAudio(ReadOnlySpan<float> audio, MainAudioObservation observation);
}
