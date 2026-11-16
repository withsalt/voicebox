using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

/// <summary>
/// Playback effect family. Both effects are owner-scoped: they act only on device audio that still
/// belongs to <c>effect.TurnId</c> and are no-ops for any other owner, so a late decision can never
/// stop a newer reply. There is no duck/restore: a pending or denied candidate changes nothing.
/// </summary>
internal interface IPlaybackEffectHandler
{
    ValueTask ClearPlaybackQueueAsync(ClearPlaybackQueueEffect effect, CancellationToken cancellationToken);
    ValueTask StopPlaybackAsync(StopPlaybackEffect effect, CancellationToken cancellationToken);
}
