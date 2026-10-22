using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Modules.Playback.Controllers
{
    /// <summary>
    /// Shields the first moments after TTS playback starts while AEC has not yet stabilized on the render reference.
    /// </summary>
    internal sealed class PlaybackStartupGuard : IPlaybackStartupGuardView
    {
        private readonly TimeProvider _timeProvider;
        private long _suppressionUntilTicks;

        public PlaybackStartupGuard(TimeProvider? timeProvider = null)
        {
            _timeProvider = timeProvider ?? TimeProvider.System;
        }

        public bool IsActive => GetRemainingMilliseconds() > 0;

        public int RemainingMilliseconds => GetRemainingMilliseconds();

        public bool IsActiveAt(long timestampTicks)
            => GetRemainingMillisecondsAt(
                timestampTicks > 0 ? timestampTicks : _timeProvider.GetTimestamp()) > 0;

        public void Start(int durationMs)
        {
            if (durationMs <= 0)
                return;

            long durationTicks = checked((long)Math.Ceiling(
                durationMs * _timeProvider.TimestampFrequency / 1000.0));
            long now = _timeProvider.GetTimestamp();
            long until = now > long.MaxValue - durationTicks
                ? long.MaxValue
                : now + durationTicks;
            Interlocked.Exchange(ref _suppressionUntilTicks, until);
        }

        public int GetRemainingMilliseconds()
            => GetRemainingMillisecondsAt(_timeProvider.GetTimestamp());

        private int GetRemainingMillisecondsAt(long timestampTicks)
        {
            long until = Volatile.Read(ref _suppressionUntilTicks);
            if (until <= timestampTicks)
                return 0;

            double remainingMs =
                (until - timestampTicks) * 1000.0 / _timeProvider.TimestampFrequency;
            if (remainingMs >= int.MaxValue)
                return int.MaxValue;
            return Math.Max(1, (int)Math.Ceiling(remainingMs));
        }

        public void Clear()
            => Interlocked.Exchange(ref _suppressionUntilTicks, 0);
    }
}
