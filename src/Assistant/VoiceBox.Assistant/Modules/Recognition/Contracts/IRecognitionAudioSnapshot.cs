namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>Receives the owned PCM snapshot of a completed recognition turn.</summary>
internal interface IRecognitionAudioSnapshot
{
    /// <summary>The samples will not be reused or mutated. A retired turn must not be recreated.</summary>
    void Store(long turnId, ReadOnlyMemory<float> samples);
}
