namespace VoiceBox.Assistant.Modules.Playback.Models;

internal readonly record struct SynthesizedAudioChunk(
    float[] Samples,
    int SampleRate,
    int Channels);
