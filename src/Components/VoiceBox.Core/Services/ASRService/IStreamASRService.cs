using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Core.Services.ASRService
{
    public interface IStreamASRService : IASRService
    {
        IAsyncEnumerable<AsrResult> GetStreamResultAsync(string sessionId, CancellationToken ct = default);
    }
}
