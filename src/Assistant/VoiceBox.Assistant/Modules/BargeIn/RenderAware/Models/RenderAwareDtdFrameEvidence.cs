namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderAwareDtdFrameEvidence(
    bool DecisionEnabled,
    bool EchoLearningEnabled,
    bool IsNearEndSpeech,
    float Probability,
    float SmoothedProbability,
    double CaptureRms,
    double RenderRms,
    double RenderCorrelation,
    double AlignmentConfidence,
    bool HasPostAecEvidence,
    double PostAecRms,
    double PostAecResidualRatio,
    double PostAecRenderCorrelation);
