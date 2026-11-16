namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Speaker provider identity and streaming geometry the profile embeddings were extracted with.</summary>
internal sealed record SpeakerProfileProvider(
    string ServiceId,
    string ServiceType,
    string ModelDigest,
    int SampleRate,
    int EmbeddingSize,
    int WindowSamples,
    int HopSamples);
