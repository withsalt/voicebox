namespace VoiceBox.Core.Services.TTSService.Models
{
    public sealed class TtsResult
    {
        public required float[] Samples { get; init; }

        public required int SampleRate { get; init; }

        public required int Channels { get; init; }

        public required string Voice { get; init; }

        public double ElapsedSeconds { get; init; }
    }
}
