using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal sealed record SpeakerBankBuildResult(
    SpeakerPrototypeBank Bank,
    IReadOnlyList<SpeakerEnrollmentSegmentEmbedding> Retained,
    IReadOnlyList<SpeakerEnrollmentSegmentEmbedding> Discarded,
    double PairwiseMedian,
    double PairwiseP10,
    double WithinSpeakerSpread);
