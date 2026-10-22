namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// owner-speaker-profile-v2: one owner, one speaker provider geometry, one variant per acoustic
/// front-end fingerprint, and per-purpose prototype banks inside each variant.
/// </summary>
internal sealed record OwnerSpeakerProfile(
    string SpeakerId,
    DateTimeOffset CreatedAt,
    SpeakerProfileProvider Provider,
    IReadOnlyList<OwnerSpeakerProfileVariant> Variants)
{
    public const string OwnerSpeakerId = "owner";

    public const int MaxVariants = 8;

    /// <summary>Finds the variant whose fingerprint equals the current tap fingerprint exactly.</summary>
    public OwnerSpeakerProfileVariant? FindVariant(string frontendFingerprint)
    {
        foreach (OwnerSpeakerProfileVariant variant in Variants)
        {
            if (string.Equals(variant.FrontendFingerprint, frontendFingerprint, StringComparison.Ordinal))
                return variant;
        }

        return null;
    }

    /// <summary>
    /// Returns a copy where the variant carrying <paramref name="variant"/>'s fingerprint is replaced
    /// and every other variant is kept untouched.
    /// </summary>
    public OwnerSpeakerProfile WithVariant(OwnerSpeakerProfileVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        var variants = new List<OwnerSpeakerProfileVariant>(Variants.Count + 1);
        bool replaced = false;
        foreach (OwnerSpeakerProfileVariant existing in Variants)
        {
            if (string.Equals(existing.FrontendFingerprint, variant.FrontendFingerprint, StringComparison.Ordinal))
            {
                variants.Add(variant);
                replaced = true;
            }
            else
            {
                variants.Add(existing);
            }
        }

        if (!replaced)
            variants.Add(variant);
        if (variants.Count > MaxVariants)
        {
            throw new InvalidOperationException(
                $"Owner speaker profile cannot hold more than {MaxVariants} front-end variants.");
        }

        return this with { Variants = variants.AsReadOnly() };
    }
}
