namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    internal sealed class OfflineRecognizerResultEntity
    {
        public string? Text { get; set; }

        public int TextLen { get; set; }

        public List<string> Tokens { get; } = new();

        public List<int[]> Timestamps { get; } = new();
    }
}
