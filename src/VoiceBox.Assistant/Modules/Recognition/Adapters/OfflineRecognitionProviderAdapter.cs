using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Adapters;

internal sealed class OfflineRecognitionProviderAdapter : IRecognitionProviderAdapter
{
    private readonly IASRService _service;

    public OfflineRecognitionProviderAdapter(IASRService service)
        => _service = service ?? throw new ArgumentNullException(nameof(service));

    public async Task<string> StartSessionAsync(CancellationToken cancellationToken)
    {
        SessionInfo session = await _service.StartSessionAsync(cancellationToken).ConfigureAwait(false);
        return session.SessionId;
    }

    public void FeedAudio(string sessionId, ReadOnlySpan<float> audio)
        => _service.FeedAudio(sessionId, audio);

    public async Task<string?> FinishSessionAsync(
        string sessionId,
        CancellationToken cancellationToken)
    {
        AsrResult? result = await _service.FinishSessionAsync(
            sessionId,
            cancellationToken).ConfigureAwait(false);
        return result?.Text;
    }

    public async Task<bool> TryCancelSessionAsync(string sessionId, CancellationToken cancellationToken)
    {
        await _service.CancelSessionAsync(sessionId, cancellationToken).ConfigureAwait(false);
        return true;
    }
}
