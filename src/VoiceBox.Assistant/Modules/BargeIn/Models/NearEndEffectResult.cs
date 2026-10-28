namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>Result of executing one typed near-end effect.</summary>
internal readonly record struct NearEndEffectResult(
    bool Succeeded,
    bool HasAttachedSession = false)
{
    public static NearEndEffectResult Ok => new(true);
}
