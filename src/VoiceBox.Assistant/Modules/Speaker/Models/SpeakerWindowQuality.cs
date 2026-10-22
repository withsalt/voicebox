namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Per-window quality statistics that gate whether a window may reach the embedding model.</summary>
internal readonly record struct SpeakerWindowQuality(
    bool AllFinite,
    double Rms,
    double Peak,
    double ClippingRatio,
    double DcOffset,
    double SpeechRatio,
    int SpeechSamples,
    int SampleCount)
{
    public static SpeakerWindowQuality Invalid => new(false, 0, 0, 0, 0, 0, 0, 0);
}
