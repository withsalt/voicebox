using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies;

/// <summary>
/// Identity confirmation disabled: the DTD Interrupt is the whole decision, early near-end
/// evidence may duck playback, and no speaker is ever queried.
/// </summary>
internal sealed class AcousticAuthorizationPolicy : INearEndAuthorizationPolicy
{
    public static AcousticAuthorizationPolicy Instance { get; } = new();

    private AcousticAuthorizationPolicy()
    {
    }

    public bool RequiresSpeaker => false;

    public bool AllowsPlaybackDucking => true;

    public bool ShouldOpenCandidate(in BargeInDecision decision)
        => decision.Action == BargeInAction.Interrupt;

    public bool IsAcousticallyAuthorized(in BargeInDecision decision) => true;

    public bool IsStrongNearEndDecision(in BargeInDecision decision)
        => NearEndDecisionRules.IsStrongNearEndDecision(in decision);
}
