namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    internal sealed record CmvnEntity(float[] NegativeMeans, float[] InverseStddevs)
    {
        public int Dim => NegativeMeans.Length;
    }
}
