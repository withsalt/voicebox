using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.Core.Services.TTSService
{
    public interface IStreamTTSService : ITTSService
    {
        IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default);
    }
}
