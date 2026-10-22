using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Provides the detector's normalized low-frequency decision.</summary>
internal interface IActiveNearEndDecisionSource
{
    BargeInDecision GetRecentDecision(TimeSpan window, int minDurationMs, float minRms);
}
