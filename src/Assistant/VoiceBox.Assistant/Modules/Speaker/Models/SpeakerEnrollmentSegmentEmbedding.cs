using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>One accepted enrollment segment reduced to a normalized embedding plus its statistics.</summary>
internal sealed record SpeakerEnrollmentSegmentEmbedding(
    SpeakerEnrollmentCondition Condition,
    int SegmentIndex,
    float[] Embedding,
    float[][] WindowEmbeddings,
    SpeakerEnrollmentSegmentQuality Quality);
