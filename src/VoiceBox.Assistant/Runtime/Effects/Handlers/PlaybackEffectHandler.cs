using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Runtime.Effects.Contracts;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class PlaybackEffectHandler : IPlaybackEffectHandler
{
    private readonly PlaybackInterruptionController _interruption;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;

    public PlaybackEffectHandler(
        PlaybackInterruptionController interruption,
        VoiceAssistantBenchmarkProbe? benchmarkProbe = null)
    {
        _interruption = interruption ?? throw new ArgumentNullException(nameof(interruption));
        _benchmarkProbe = benchmarkProbe;
    }

    public ValueTask ClearPlaybackQueueAsync(ClearPlaybackQueueEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _benchmarkProbe?.ObserveControlTrace("effect.clear-playback", effect.TurnId, effect.OperationId);
        _interruption.StopOwnedPlayback(effect.TurnId);
        return ValueTask.CompletedTask;
    }

    public ValueTask StopPlaybackAsync(StopPlaybackEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _benchmarkProbe?.ObserveControlTrace("effect.stop-playback", effect.TurnId, effect.OperationId);
        _interruption.StopOwnedPlayback(effect.TurnId);
        return ValueTask.CompletedTask;
    }
}
