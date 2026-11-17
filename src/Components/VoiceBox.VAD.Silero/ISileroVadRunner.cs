namespace VoiceBox.VAD.Silero;

internal interface ISileroVadRunner : IDisposable
{
    float[] Call(float[][] audio, int sampleRate);

    void ResetStates();
}
