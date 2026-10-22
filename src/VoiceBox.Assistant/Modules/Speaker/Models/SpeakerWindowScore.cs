using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Per-window comparison of one embedding against one prototype bank.</summary>
internal readonly record struct SpeakerWindowScore(
    float CentroidScore,
    float BestPrototypeScore,
    float Top2MeanScore)
{
    /// <summary>Reported score: the stronger of the two agreement measures.</summary>
    public float Reported => Math.Max(Top2MeanScore, CentroidScore);

    public bool IsFinite =>
        float.IsFinite(CentroidScore) && float.IsFinite(BestPrototypeScore) && float.IsFinite(Top2MeanScore);
}
