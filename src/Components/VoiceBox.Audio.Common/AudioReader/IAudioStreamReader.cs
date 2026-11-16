namespace VoiceBox.Audio.Common.AudioReader
{
    public interface IAudioStreamReader : IDisposable
    {
        int ReadSamples(Span<float> buffer);

        void Reset();
    }
}
