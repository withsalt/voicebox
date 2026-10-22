using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Modules.Playback.Contracts;

namespace VoiceBox.Assistant.Modules.Playback.Controllers;

internal sealed class PlaybackDrainWaiter
{
    private readonly IBoundedPlaybackPort _playback;
    private readonly ILogger _logger;
    private readonly int _tailBufferMs;

    public PlaybackDrainWaiter(
        IBoundedPlaybackPort playback,
        ILogger logger,
        int tailBufferMs)
    {
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _tailBufferMs = tailBufferMs;
    }

    public async Task WaitAsync(long totalSamples, int sampleRate, CancellationToken cancellationToken)
    {
        double expectedMs = sampleRate > 0 && totalSamples > 0
            ? totalSamples * 1000.0 / sampleRate
            : 0;
        double safetyTimeoutMs = expectedMs * 2 + 5000;
        const int pollIntervalMs = 50;
        const int requiredIdleSamples = 3;
        int idleStreak = 0;
        double waited = 0;

        while (!cancellationToken.IsCancellationRequested && waited < safetyTimeoutMs)
        {
            bool idle = _playback.CurrentQueueSize == 0 && !_playback.IsPlaying;
            if (idle)
            {
                if (++idleStreak >= requiredIdleSamples)
                    break;
            }
            else
            {
                idleStreak = 0;
            }

            try
            {
                await Task.Delay(pollIntervalMs, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            waited += pollIntervalMs;
        }

        if (waited >= safetyTimeoutMs)
        {
            _logger.LogWarning(
                "播放排空轮询超时 ({Waited:F0}ms / {Safety:F0}ms), 可能存在播放堆积。",
                waited,
                safetyTimeoutMs);
        }

        if (_tailBufferMs > 0)
        {
            try
            {
                await Task.Delay(_tailBufferMs, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
            }
        }
    }
}
