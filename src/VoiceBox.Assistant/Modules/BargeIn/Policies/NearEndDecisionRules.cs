using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies;

/// <summary>Pure predicates shared by the near-end authorization policies.</summary>
internal static class NearEndDecisionRules
{
    public static bool HasUsableTimeline(in BargeInDecision decision)
        => !decision.HasDiscontinuity &&
           decision.StreamEpoch > 0 &&
           decision.StartSample >= 0 &&
           decision.EndSampleExclusive > decision.StartSample;

    /// <summary>
    /// Valid near-end speech on the shared timeline: the DTD accepted the latest frame and has
    /// causal near-end evidence. This is the candidate trigger, not an authorization.
    /// </summary>
    public static bool HasNearEndSpeechEvidence(in BargeInDecision decision)
        => HasUsableTimeline(in decision) &&
           decision.LatestFrameAccepted &&
           decision.AcceptedFrameCount > 0 &&
           decision.AcceptedDurationMs > 0;

    public static bool IsStrongNearEndDecision(in BargeInDecision decision)
        => decision.Action == BargeInAction.Interrupt &&
            !decision.HasDiscontinuity &&
            decision.LatestFrameAccepted &&
            decision.EchoLikeFrameCount == 0 &&
            decision.AcceptedDurationMs >= decision.RequiredDurationMs;
}
