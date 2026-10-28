using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies;

/// <summary>
/// Identity confirmation enabled: the candidate opens on valid near-end speech evidence and only
/// the current candidate's speaker Match authorizes it. A DTD Interrupt never authorizes on its
/// own and never lowers playback gain.
/// </summary>
internal sealed class SpeakerAuthorizationPolicy : INearEndAuthorizationPolicy
{
    public static SpeakerAuthorizationPolicy Instance { get; } = new();

    private SpeakerAuthorizationPolicy()
    {
    }

    public bool RequiresSpeaker => true;

    public bool AllowsPlaybackDucking => false;

    public bool ShouldOpenCandidate(in BargeInDecision decision)
        => NearEndDecisionRules.HasNearEndSpeechEvidence(in decision);

    public bool IsAcousticallyAuthorized(in BargeInDecision decision) => false;

    public bool IsStrongNearEndDecision(in BargeInDecision decision)
        => NearEndDecisionRules.IsStrongNearEndDecision(in decision);
}
