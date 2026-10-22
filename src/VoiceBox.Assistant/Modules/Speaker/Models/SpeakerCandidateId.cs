namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// Identity of one speaker verification candidate. Every inference job, cache entry and verdict
/// is correlated by it so late results can be rejected without touching another candidate.
/// </summary>
internal readonly record struct SpeakerCandidateId(long Value)
{
    public static SpeakerCandidateId None => default;

    public bool IsNone => Value == 0;

    public override string ToString() => Value.ToString();
}
