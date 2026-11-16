namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    internal sealed class EncoderOutputEntity
    {
        public float[]? Output { get; set; }

        public long[] OutputLengths { get; set; } = Array.Empty<long>();

        public bool[]? Mask { get; set; }
    }
}
