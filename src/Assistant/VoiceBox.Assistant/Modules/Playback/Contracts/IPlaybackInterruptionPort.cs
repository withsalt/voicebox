namespace VoiceBox.Assistant.Modules.Playback.Contracts;

/// <summary>
/// Playback-side effects of an interruption decision, used by the barge-in and wakeup owners.
/// </summary>
/// <remarks>
/// Deliberately kept apart from <c>IActiveNearEndPlaybackPort</c>: that port serves the near-end
/// data plane and exposes only playback activity, while this one serves interruption owners on the
/// control plane. Neither exposes gain changes: a pending or denied identity candidate leaves the
/// interrupted reply and its gain untouched.
/// </remarks>
internal interface IPlaybackInterruptionPort
{
    /// <summary>
    /// Cancels the response of <paramref name="expectedOwnerTurnId"/> and removes its device PCM,
    /// but only while that turn still owns the active playback; otherwise it is a no-op and returns
    /// false. The cancellation source is borrowed; ownership stays with the turn that created it.
    /// </summary>
    bool InterruptActiveResponse(long expectedOwnerTurnId, CancellationTokenSource? responseCts);
}
