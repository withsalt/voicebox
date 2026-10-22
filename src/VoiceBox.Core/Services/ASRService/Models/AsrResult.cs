using System.Text.Json.Serialization;

namespace VoiceBox.Core.Services.ASRService.Models
{
    public class AsrResult
    {
        [JsonPropertyName("language")]
        public string Language { get; set; } = "";


        [JsonPropertyName("text")]
        public string Text { get; set; } = "";

        /// <summary>
        /// Streaming-only: the stable prefix that subsequent re-transcription rounds
        /// will keep verbatim. Null if the underlying ASR does not distinguish a
        /// committed / uncommitted split.
        /// </summary>
        [JsonPropertyName("committed_text")]
        public string? CommittedText { get; set; }

        /// <summary>
        /// Streaming-only: the tail emitted by the latest decode pass; may be revised
        /// in the next round. Null if no such split is available.
        /// </summary>
        [JsonPropertyName("uncommitted_text")]
        public string? UncommittedText { get; set; }
    }
}

