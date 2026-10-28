namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>One reserved ASR feed range owned by a capture turn.</summary>
internal readonly record struct NearEndFeedReservation(
    long TurnId,
    long StartSample,
    long EndSampleExclusive)
{
    public bool IsValid => EndSampleExclusive > StartSample;
}
