namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Robust centroid plus a bounded set of representative prototypes for one condition.</summary>
internal sealed record SpeakerPrototypeBank(
    float[] Centroid,
    IReadOnlyList<SpeakerPrototype> Prototypes)
{
    public const int MinPrototypes = 2;

    public const int MaxPrototypes = 8;

    /// <summary>A bank with fewer than two prototypes cannot support the top-2 rule and is unusable.</summary>
    public bool IsUsable => Prototypes.Count >= MinPrototypes && Centroid.Length > 0;
}
