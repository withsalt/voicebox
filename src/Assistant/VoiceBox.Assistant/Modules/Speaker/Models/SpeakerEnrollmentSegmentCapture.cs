using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Raw material of one enrollment segment attempt, collected from the identity tap.</summary>
internal sealed record SpeakerEnrollmentSegmentCapture(
    float[] SpeechAudio,
    long TotalSamples,
    int FrameCount,
    int RenderActiveFrames,
    int NearEndAcceptedFrames,
    bool NearEndObserved);
