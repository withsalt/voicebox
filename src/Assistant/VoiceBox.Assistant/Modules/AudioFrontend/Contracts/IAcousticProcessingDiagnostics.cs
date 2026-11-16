using VoiceBox.Assistant.Diagnostics.Metrics.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>Optional immutable acoustic diagnostics exposed by a capable audio frontend.</summary>
internal interface IAcousticProcessingDiagnostics
{
    bool TryGetLatestMetrics(out AcousticProcessingMetricsSnapshot metrics);
}
