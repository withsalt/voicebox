namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct MainAudioObservation(
    long Generation,
    long StreamEpoch,
    long StartSample,
    long EndSample,
    bool HasDiscontinuity);
