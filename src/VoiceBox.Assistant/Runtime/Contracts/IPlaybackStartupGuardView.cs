namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IPlaybackStartupGuardView
{
    bool IsActive { get; }

    int RemainingMilliseconds { get; }

    bool IsActiveAt(long timestampTicks);
}
