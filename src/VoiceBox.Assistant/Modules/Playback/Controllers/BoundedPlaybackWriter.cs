using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Modules.Playback.Contracts;

namespace VoiceBox.Assistant.Modules.Playback.Controllers;

internal sealed class BoundedPlaybackWriter
{
    private static readonly TimeSpan EnqueueStallTimeout = TimeSpan.FromSeconds(30);
    private const int EnqueueRetryDelayMs = 20;

    private readonly IBoundedPlaybackPort _playback;
    private readonly ILogger _logger;
    private readonly long? _ownerTurnId;
    private readonly IPlaybackStartedSink _playbackStarted;
    private int _playbackStartedNotified;

    public BoundedPlaybackWriter(
        IBoundedPlaybackPort playback,
        ILogger logger,
        long? ownerTurnId,
        IPlaybackStartedSink playbackStarted)
    {
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _ownerTurnId = ownerTurnId;
        _playbackStarted = playbackStarted ?? throw new ArgumentNullException(nameof(playbackStarted));
    }

    public async ValueTask EnqueueAsync(float[] buffer, int count, CancellationToken cancellationToken)
    {
        long firstFailureTicks = 0;
        long lastLogTicks = 0;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_playback.TryEnqueue(buffer.AsSpan(0, count), out int droppedSamples))
            {
                if (_ownerTurnId is { } turnId &&
                    Interlocked.Exchange(ref _playbackStartedNotified, 1) == 0)
                {
                    _playbackStarted.OnPlaybackStarted(turnId);
                }
                return;
            }

            if (!_playback.IsActive)
            {
                throw new InvalidOperationException(
                    $"播放服务当前状态为 {_playback.StateName}, 无法入队 {count} 个样本。");
            }

            long nowTicks = Stopwatch.GetTimestamp();
            if (firstFailureTicks == 0)
            {
                firstFailureTicks = nowTicks;
                lastLogTicks = nowTicks;
            }
            else
            {
                TimeSpan stalled = Stopwatch.GetElapsedTime(firstFailureTicks, nowTicks);
                if (stalled >= EnqueueStallTimeout)
                {
                    throw new TimeoutException(
                        $"播放队列连续 {stalled.TotalSeconds:F1}s 未能接收音频块, 最近一次拒绝 {droppedSamples} 个样本。");
                }

                if (Stopwatch.GetElapsedTime(lastLogTicks, nowTicks) >= TimeSpan.FromSeconds(1))
                {
                    lastLogTicks = nowTicks;
                    _logger.LogWarning(
                        "播放队列已满, 等待消费后重试入队 ({QueueSize} queued, droppedAttempt={DroppedSamples}).",
                        _playback.CurrentQueueSize,
                        droppedSamples);
                }
            }

            await Task.Delay(EnqueueRetryDelayMs, cancellationToken).ConfigureAwait(false);
        }
    }
}
