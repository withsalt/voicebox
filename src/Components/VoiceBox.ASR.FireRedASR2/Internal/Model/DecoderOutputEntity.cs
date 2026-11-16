namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    internal sealed class DecoderOutputEntity
    {
        public List<List<long>> TokensList { get; set; } = new();

        public List<float[]> CacheList { get; set; } = new();
    }
}
