using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Primary-capture operations used by the hot capture path.</summary>
internal interface INearEndCapturePort
{
    MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame);

    void FeedFromRecognitionAudio(ReadOnlySpan<float> audio, MainAudioObservation observation);
}
