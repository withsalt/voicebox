namespace VoiceBox.Core.Services.TTSService.Models
{
    public sealed class TtsVoice
    {
        public string Name { get; init; } = string.Empty;

        public string? Description { get; init; }

        public string? Language { get; init; }
    }
}
