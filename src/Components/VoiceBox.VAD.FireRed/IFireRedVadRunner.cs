namespace VoiceBox.VAD.FireRed;

internal interface IFireRedVadRunner : IDisposable
{
    float[] Run(float[][] fbankFrames);

    void ResetCaches();
}
