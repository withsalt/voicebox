using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Core.Services.ASRService
{
    public interface IASRService : IDisposable
    {
        Task<SessionInfo> StartSessionAsync(CancellationToken ct = default);

        void FeedAudio(string sessionId, ReadOnlySpan<float> audioData);

        /// <summary>
        /// Finishes recognition and releases the session. Fatal provider failures also
        /// release automatically and are rethrown here, including for recently released
        /// sessions (up to five minutes and the latest 128 failed sessions per service).
        /// </summary>
        Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default);

        /// <summary>
        /// Releases a session whose result the caller has abandoned, after in-flight provider
        /// work has stopped. Cancelling a Finish waiter need not release that session.
        /// Stops audio admission and skips final recognition. A valid missing/already-released
        /// session is a no-op. The token only cancels the wait: provider cleanup continues and
        /// retains capacity until in-flight work exits. Successful completion confirms release.
        /// </summary>
        Task CancelSessionAsync(string sessionId, CancellationToken ct = default);
    }
}
