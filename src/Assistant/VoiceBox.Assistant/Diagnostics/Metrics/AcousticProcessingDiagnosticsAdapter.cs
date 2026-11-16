using VoiceBox.Assistant.Diagnostics.Metrics.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.Assistant.Diagnostics.Metrics;

/// <summary>
/// Anti-corruption adapter that exposes a provider metrics source as the Assistant-owned acoustic
/// diagnostics port. Runtime modules receive only normalized immutable snapshots.
/// </summary>
internal sealed class AcousticProcessingDiagnosticsAdapter : IAcousticProcessingDiagnostics
{
    private readonly IAcousticProcessingMetricsSource _source;

    public AcousticProcessingDiagnosticsAdapter(IAcousticProcessingMetricsSource source)
        => _source = source ?? throw new ArgumentNullException(nameof(source));

    public bool TryGetLatestMetrics(out AcousticProcessingMetricsSnapshot metrics)
    {
        if (!_source.TryGetLatestMetrics(out AcousticProcessingMetrics provider))
        {
            metrics = default;
            return false;
        }

        metrics = new AcousticProcessingMetricsSnapshot(
            provider.TimestampTicks,
            provider.HasDelayMs,
            provider.DelayMs,
            provider.HasDelayMedianMs,
            provider.DelayMedianMs,
            provider.HasResidualEchoLikelihood,
            provider.ResidualEchoLikelihood);
        return true;
    }
}
