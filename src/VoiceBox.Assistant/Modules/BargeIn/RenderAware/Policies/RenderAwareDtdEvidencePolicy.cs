using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Policies;

/// <summary>
/// Converts DTD output plus runtime acoustic evidence into one frame decision. Constants
/// describe signal relationships only; the learned envelope remains stream-local.
/// </summary>
internal sealed class RenderAwareDtdEvidencePolicy
{
    private const double BootstrapEchoResidualRatio = 0.20;
    private const double MinimumEchoCorrelation = 0.20;
    private const double MinimumLearningCorrelation = 0.25;
    private const double MinimumAcousticPathChangeCorrelation = 0.45;
    private const double TrustedEchoResidualCeiling = 0.65;
    private const double LearnedPostAecMargin = 1.60;
    private const double LearnedCaptureMargin = 1.80;
    private const double LearnedResidualMargin = 1.50;
    private const double PreservedPostAecMargin = 2.25;
    private const double PreservedCaptureMargin = 1.50;
    private const float PureEchoModelMargin = 0.12f;
    private const double MinimumTrustedAlignmentConfidence = 0.60;

    private readonly float _decisionThreshold;

    public RenderAwareDtdEvidencePolicy(float decisionThreshold)
    {
        if (!float.IsFinite(decisionThreshold) || decisionThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(decisionThreshold));
        _decisionThreshold = decisionThreshold;
    }

    public RenderAwareDtdEvidenceDecision Evaluate(
        in RenderAwareDtdFrameEvidence evidence,
        in RuntimeEchoProfileSnapshot profile)
    {
        bool hasUsablePostAec = evidence.HasPostAecEvidence &&
            double.IsFinite(evidence.PostAecRms) && evidence.PostAecRms >= 0 &&
            double.IsFinite(evidence.PostAecResidualRatio) && evidence.PostAecResidualRatio >= 0;
        double inputCorrelation = Math.Abs(evidence.RenderCorrelation);
        double outputCorrelation = double.IsFinite(evidence.PostAecRenderCorrelation)
            ? Math.Abs(evidence.PostAecRenderCorrelation)
            : 0;
        RenderAwareDtdCausalEvidence causal = ComputeCausalEvidence(
            in evidence,
            in profile,
            hasUsablePostAec,
            inputCorrelation,
            outputCorrelation);
        bool preservedNearEnd = causal.PreservedSupport >= 0.50;
        bool echoLike = causal.EchoSupport >= 0.55 &&
            causal.EchoSupport > causal.ModelSupport &&
            !preservedNearEnd;
        bool accepted = evidence.DecisionEnabled &&
            evidence.IsNearEndSpeech &&
            causal.Score >= 0.50;

        string reason = accepted
            ? preservedNearEnd
                ? "render-aware-dtd-causal-preserved-nearend"
                : "render-aware-dtd-causal"
            : echoLike
                ? "render-aware-dtd-causal-echo-veto"
                : hasUsablePostAec
                    ? "render-aware-dtd-causal-rejected"
                    : "render-aware-dtd-causal-no-post-aec-evidence";
        return new RenderAwareDtdEvidenceDecision(
            accepted,
            echoLike,
            preservedNearEnd,
            causal.TrustedEcho,
            causal.AcousticPathChange,
            reason);
    }

    private RenderAwareDtdCausalEvidence ComputeCausalEvidence(
        in RenderAwareDtdFrameEvidence evidence,
        in RuntimeEchoProfileSnapshot profile,
        bool hasUsablePostAec,
        double inputCorrelation,
        double outputCorrelation)
    {
        // Every branch below contributes to one causal score. The profile is read before
        // it is updated, so trusted echo can only affect later frames.
        double modelSupport = NormalizeModelSupport(evidence.Probability, evidence.SmoothedProbability);
        double correlationSupport = Math.Clamp(
            Math.Max(inputCorrelation, outputCorrelation) / 0.90,
            0,
            1);
        double echoSupport = 0;
        if (hasUsablePostAec && evidence.CaptureRms > 1e-4 &&
            Math.Max(inputCorrelation, outputCorrelation) >= MinimumEchoCorrelation)
        {
            double residualSupport = evidence.PostAecResidualRatio < BootstrapEchoResidualRatio
                ? 1.0
                : 0.0;
            echoSupport = residualSupport * (0.55 + 0.45 * correlationSupport);

            if (profile.IsReliable)
            {
                double postAecUpper = profile.PredictedPostAecRms * LearnedPostAecMargin;
                double captureUpper = profile.PredictedCaptureRms * LearnedCaptureMargin;
                double residualUpper = Math.Max(
                    BootstrapEchoResidualRatio,
                    profile.PredictedResidualRatio * LearnedResidualMargin);
                float learnedSmoothedUpper = Math.Max(
                    _decisionThreshold + PureEchoModelMargin,
                    profile.PureEchoSmoothedProbabilityUpper + PureEchoModelMargin);
                bool learnedEcho = evidence.PostAecRms <= postAecUpper &&
                    evidence.CaptureRms <= captureUpper &&
                    evidence.PostAecResidualRatio >= TrustedEchoResidualCeiling &&
                    evidence.PostAecResidualRatio <= residualUpper &&
                    evidence.SmoothedProbability <= Math.Min(1, learnedSmoothedUpper);
                if (learnedEcho)
                    echoSupport = Math.Max(echoSupport, 0.90);
            }
        }

        double preservedSupport = 0;
        bool acousticPathChange = false;
        if (evidence.IsNearEndSpeech && profile.IsReliable && hasUsablePostAec)
        {
            bool exceedsPostAecEnvelope = evidence.PostAecRms >
                profile.PredictedPostAecRms * PreservedPostAecMargin;
            bool exceedsCaptureEnvelope = evidence.CaptureRms >
                profile.PredictedCaptureRms * PreservedCaptureMargin;
            bool exceedsResidualEnvelope = evidence.PostAecResidualRatio >
                profile.PredictedResidualRatio * LearnedResidualMargin;
            bool envelopeBreak = modelSupport >= 0.20 &&
                exceedsPostAecEnvelope &&
                (exceedsCaptureEnvelope || exceedsResidualEnvelope);
            acousticPathChange = envelopeBreak &&
                Math.Max(inputCorrelation, outputCorrelation) >= MinimumAcousticPathChangeCorrelation;
            if (envelopeBreak && !acousticPathChange)
                preservedSupport = 0.90;
        }

        double score = modelSupport * (1.0 - 0.85 * echoSupport) +
            preservedSupport * (1.0 - modelSupport * 0.25);
        bool trustedEcho = evidence.EchoLearningEnabled &&
            hasUsablePostAec &&
            !evidence.IsNearEndSpeech &&
            evidence.AlignmentConfidence >= MinimumTrustedAlignmentConfidence &&
            evidence.RenderRms > 1e-5 &&
            evidence.CaptureRms > 1e-4 &&
            evidence.PostAecResidualRatio <= TrustedEchoResidualCeiling &&
            modelSupport <= 0.40 &&
            Math.Max(inputCorrelation, outputCorrelation) >= MinimumLearningCorrelation;

        return new RenderAwareDtdCausalEvidence(
            modelSupport,
            echoSupport,
            preservedSupport,
            Math.Clamp(score, 0, 1),
            trustedEcho,
            acousticPathChange);
    }

    private double NormalizeModelSupport(float probability, float smoothedProbability)
    {
        if (!float.IsFinite(probability) || !float.IsFinite(smoothedProbability))
            return 0;

        double model = 0.35 * probability + 0.65 * smoothedProbability;
        // The model's boolean near-end output is already thresholded. Keep a small
        // probability floor here so moderate, causal onset frames are not discarded
        // merely because smoothing has not reached the final plateau yet.
        double floor = _decisionThreshold * 0.20;
        return Math.Clamp((model - floor) / Math.Max(1e-6, 1.0 - floor), 0, 1);
    }
}
