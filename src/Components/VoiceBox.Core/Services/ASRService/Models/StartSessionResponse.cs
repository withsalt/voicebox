using System.Text.Json.Serialization;

namespace VoiceBox.Core.Services.ASRService.Models
{
    public class StartSessionResponse
    {
        [JsonPropertyName("session_id")]
        public string SessionId { get; set; } = "";
    }
}
