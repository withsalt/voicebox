using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Isolates the two business rules that differ between acoustic-only and identity-confirmed
/// barge-in: whether a candidate may open and what may authorize it. Implementations are pure
/// predicates and never mutate shared state.
/// </summary>
internal interface INearEndAuthorizationPolicy
{
    /// <summary>True when this composition requires the current candidate's speaker Match.</summary>
    bool RequiresSpeaker { get; }

    /// <summary>True when early near-end evidence may temporarily lower playback gain.</summary>
    bool AllowsPlaybackDucking { get; }

    /// <summary>Whether a fresh candidate may open on this detector decision.</summary>
    bool ShouldOpenCandidate(in BargeInDecision decision);

    /// <summary>Whether the decision alone authorizes the interruption without speaker identity.</summary>
    bool IsAcousticallyAuthorized(in BargeInDecision decision);

    /// <summary>Strong upstream evidence flag recorded on a speaker candidate.</summary>
    bool IsStrongNearEndDecision(in BargeInDecision decision);
}
