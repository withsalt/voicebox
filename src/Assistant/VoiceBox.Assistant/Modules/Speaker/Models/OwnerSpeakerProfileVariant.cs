namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>One acoustic front-end variant of the owner.</summary>
internal sealed record OwnerSpeakerProfileVariant(
    string VariantId,
    string FrontendFingerprint,
    string TapId,
    DateTimeOffset CreatedAt,
    IReadOnlyDictionary<SpeakerPurposeBank, SpeakerPrototypeBank> Banks,
    OwnerSpeakerProfileQuality Quality)
{
    public bool TryGetBank(SpeakerPurposeBank bank, out SpeakerPrototypeBank value)
        => Banks.TryGetValue(bank, out value!);
}
