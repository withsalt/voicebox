using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Models;

namespace VoiceBox.Assistant.Modules.Playback.Controllers
{
    /// <summary>
    /// Owns the physical stop of an interrupted reply. Every entry point is owner-scoped and
    /// idempotent: the active playback is cancelled and its device PCM removed only while it still
    /// belongs to the expected owner turn. There is no fallback to an unscoped
    /// <c>ClearQueue()</c>. Temporary gain changes also go through the logical playback owner.
    /// </summary>
    internal sealed class PlaybackInterruptionController :
        IActiveNearEndPlaybackPort,
        IPlaybackInterruptionPort,
        IPlaybackActivityView
    {
        private readonly IAssistantAudioOutput _audio;
        private readonly PlaybackStartupGuard _startupGuard;
        private readonly IPlaybackCancellationPort _playbackCancellation;
        private readonly ILogger _logger;
        private long _ownedStops;
        private long _ignoredStops;
        private long _duplicateStops;

        public PlaybackInterruptionController(
            IAssistantAudioOutput audio,
            PlaybackStartupGuard startupGuard,
            IPlaybackCancellationPort playbackCancellation,
            ILogger logger)
        {
            _audio = audio ?? throw new ArgumentNullException(nameof(audio));
            _startupGuard = startupGuard ?? throw new ArgumentNullException(nameof(startupGuard));
            _playbackCancellation = playbackCancellation ?? throw new ArgumentNullException(nameof(playbackCancellation));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>Owner-scoped stops actually executed.</summary>
        public long OwnedStops => Volatile.Read(ref _ownedStops);

        /// <summary>Stops ignored because the expected owner no longer held the device audio.</summary>
        public long IgnoredStops => Volatile.Read(ref _ignoredStops);

        /// <summary>Repeated stops for a request that was already stopped (idempotent no-ops).</summary>
        public long DuplicateStops => Volatile.Read(ref _duplicateStops);

        public bool InterruptActiveResponse(long expectedOwnerTurnId, CancellationTokenSource? responseCts)
        {
            bool stopped = StopOwnedPlayback(expectedOwnerTurnId);
            if (!stopped)
                return false;

            try { responseCts?.Cancel(); }
            catch (ObjectDisposedException) { }
            return true;
        }

        /// <summary>
        /// The single physical stop gate: atomically checks the owner, then cancels the request and
        /// clears its device audio. Idempotent per owner turn: the Kernel's StopPlayback and
        /// ClearPlaybackQueue effects for one candidate reach the device exactly once.
        /// </summary>
        public bool StopOwnedPlayback(long expectedOwnerTurnId)
        {
            if (expectedOwnerTurnId <= 0)
            {
                Interlocked.Increment(ref _ignoredStops);
                return false;
            }

            PlaybackStopOutcome outcome;
            try
            {
                outcome = _playbackCancellation.TryStopActivePlayback(expectedOwnerTurnId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "停止回合 {TurnId} 的播放失败。", expectedOwnerTurnId);
                return false;
            }

            switch (outcome)
            {
                case PlaybackStopOutcome.Stopped:
                    _startupGuard.Clear();
                    Interlocked.Increment(ref _ownedStops);
                    _logger.LogDebug("physical-interrupt-issued: ownerTurnId={TurnId}.", expectedOwnerTurnId);
                    return true;
                case PlaybackStopOutcome.AlreadyStopped:
                    Interlocked.Increment(ref _duplicateStops);
                    return true;
                default:
                    Interlocked.Increment(ref _ignoredStops);
                    _logger.LogDebug(
                        "physical-interrupt-ignored: ownerTurnId={TurnId}, activeOwner={ActiveOwner}.",
                        expectedOwnerTurnId,
                        _playbackCancellation.ActiveOwnerTurnId);
                    return false;
            }
        }

        public bool IsPlaybackActive => _audio.IsPlaying || _audio.CurrentQueueSize > 0;

        public long? ActiveOwnerTurnId => _playbackCancellation.ActiveOwnerTurnId;

        public bool TryDuckPlayback(long ownerTurnId, float gain, float originalGain)
        {
            if (!_playbackCancellation.TryDuckActivePlayback(ownerTurnId, gain, originalGain))
                return false;
            _logger.LogDebug("playback-ducked: ownerTurnId={TurnId}, gain={Gain:F2}, originalGain={OriginalGain:F2}.",
                ownerTurnId, gain, originalGain);
            return true;
        }

        public bool TryRestorePlayback(long ownerTurnId)
        {
            if (!_playbackCancellation.TryRestoreActivePlayback(ownerTurnId))
                return false;
            _logger.LogDebug("playback-gain-restored: ownerTurnId={TurnId}.", ownerTurnId);
            return true;
        }

        PlaybackActivitySnapshot IPlaybackActivityView.GetActivitySnapshot()
            => new(IsPlaybackActive);
    }
}
