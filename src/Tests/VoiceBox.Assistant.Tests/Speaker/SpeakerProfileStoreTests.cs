using System.Text.Json;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerProfileStoreTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "voicebox-speaker-store", Guid.NewGuid().ToString("N"));
    private readonly ScriptedSpeakerService _speaker = new();

    public SpeakerProfileStoreTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public async Task SaveAndLoad_RoundTripsVersion6WithVariantsAndBanks()
    {
        string path = Path.Combine(_dir, "owner.json");
        SpeakerCaptureDescriptor descriptor = SpeakerTestProfiles.Descriptor();
        await SpeakerProfileStore.SaveAsync(_speaker, path, SpeakerTestProfiles.Profile(_speaker, descriptor));

        string json = await File.ReadAllTextAsync(path);
        Assert.Contains("\"formatVersion\": 6", json, StringComparison.Ordinal);
        Assert.Contains("owner-speaker-profile-v2", json, StringComparison.Ordinal);
        Assert.Contains("cleanNearEnd", json, StringComparison.Ordinal);
        Assert.Contains("doubleTalkNearEnd", json, StringComparison.Ordinal);
        Assert.DoesNotContain("processed-capture-v1", json, StringComparison.Ordinal);

        OwnerSpeakerProfile loaded = SpeakerProfileStore.Load(_speaker, path);
        OwnerSpeakerProfileVariant variant = Assert.Single(loaded.Variants);
        Assert.Equal(descriptor.FrontendFingerprint, variant.FrontendFingerprint);
        Assert.Equal(SpeakerCaptureDescriptor.PostCaptureProcessorTapId, variant.TapId);
        Assert.True(variant.TryGetBank(SpeakerPurposeBank.CleanNearEnd, out SpeakerPrototypeBank clean));
        Assert.Equal(2, clean.Prototypes.Count);
        Assert.Equal(ScriptedSpeakerService.OwnerAxis(), clean.Centroid);
        Assert.False(Directory.EnumerateFiles(_dir, ".*.tmp").Any());
    }

    [Fact]
    public async Task Load_RejectsRetiredFormat5ProcessedCaptureProfile()
    {
        string path = Path.Combine(_dir, "legacy.json");
        var legacy = new
        {
            formatVersion = 5,
            speakerId = "owner",
            serviceType = _speaker.GetType().FullName,
            sampleRate = 16000,
            embeddingSize = 8,
            streamingWindowSamples = 8000,
            streamingHopSamples = 1600,
            audioTap = "processed-capture-v1",
            createdAt = DateTimeOffset.UtcNow,
            embedding = SpeakerProfileStore.EncodeEmbedding(ScriptedSpeakerService.OwnerAxis()),
        };
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(legacy));

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => SpeakerProfileStore.Load(_speaker, path));
        Assert.Contains("format version 5", error.Message, StringComparison.Ordinal);
        Assert.Contains("processed-capture-v1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RequireVariant_SelectsExactFingerprintAndFailsClosedOtherwise()
    {
        string path = Path.Combine(_dir, "owner.json");
        SpeakerCaptureDescriptor noAec = SpeakerTestProfiles.Descriptor(aec: false);
        SpeakerCaptureDescriptor withAec = SpeakerTestProfiles.Descriptor(aec: true);
        await SpeakerProfileStore.SaveAsync(_speaker, path, SpeakerTestProfiles.Profile(_speaker, noAec, includeDoubleTalk: false));
        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, path);

        OwnerSpeakerProfileVariant variant = SpeakerProfileStore.RequireVariant(
            profile,
            noAec,
            [SpeakerPurposeBank.CleanNearEnd],
            path);
        Assert.Equal(noAec.FrontendFingerprint, variant.FrontendFingerprint);

        InvalidOperationException missingVariant = Assert.Throws<InvalidOperationException>(() =>
            SpeakerProfileStore.RequireVariant(profile, withAec, [SpeakerPurposeBank.CleanNearEnd], path));
        Assert.Contains("no variant", missingVariant.Message, StringComparison.Ordinal);

        InvalidOperationException missingBank = Assert.Throws<InvalidOperationException>(() =>
            SpeakerProfileStore.RequireVariant(profile, noAec, [SpeakerPurposeBank.DoubleTalkNearEnd], path));
        Assert.Contains("doubleTalkNearEnd", missingBank.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task WithVariant_ReplacesSameFingerprintAndKeepsOtherFrontends()
    {
        string path = Path.Combine(_dir, "owner.json");
        SpeakerCaptureDescriptor noAec = SpeakerTestProfiles.Descriptor(aec: false);
        SpeakerCaptureDescriptor withAec = SpeakerTestProfiles.Descriptor(aec: true);
        OwnerSpeakerProfile profile = SpeakerTestProfiles.Profile(_speaker, noAec)
            .WithVariant(SpeakerTestProfiles.Variant(withAec) with { VariantId = "aec-variant" });
        await SpeakerProfileStore.SaveAsync(_speaker, path, profile);

        OwnerSpeakerProfile replaced = SpeakerProfileStore.Load(_speaker, path)
            .WithVariant(SpeakerTestProfiles.Variant(noAec) with { VariantId = "re-enrolled" });
        await SpeakerProfileStore.SaveAsync(_speaker, path, replaced);
        OwnerSpeakerProfile loaded = SpeakerProfileStore.Load(_speaker, path);

        Assert.Equal(2, loaded.Variants.Count);
        Assert.Equal("re-enrolled", loaded.FindVariant(noAec.FrontendFingerprint)!.VariantId);
        Assert.Equal("aec-variant", loaded.FindVariant(withAec.FrontendFingerprint)!.VariantId);
    }

    [Fact]
    public async Task Load_RejectsGeometryMismatchAndCorruptEmbeddings()
    {
        string path = Path.Combine(_dir, "owner.json");
        SpeakerCaptureDescriptor descriptor = SpeakerTestProfiles.Descriptor();
        await SpeakerProfileStore.SaveAsync(_speaker, path, SpeakerTestProfiles.Profile(_speaker, descriptor));

        var otherGeometry = new ScriptedSpeakerService(windowSamples: 16000, hopSamples: 1600);
        Assert.Throws<InvalidDataException>(() => SpeakerProfileStore.Load(otherGeometry, path));

        string json = await File.ReadAllTextAsync(path);
        string corrupt = json.Replace(SpeakerProfileStore.EncodeEmbedding(ScriptedSpeakerService.OwnerAxis()), "AAAA", StringComparison.Ordinal);
        await File.WriteAllTextAsync(path, corrupt);
        Assert.Throws<InvalidDataException>(() => SpeakerProfileStore.Load(_speaker, path));
    }

    [Fact]
    public async Task Save_RejectsBanksOutsidePrototypeBounds()
    {
        string path = Path.Combine(_dir, "owner.json");
        SpeakerCaptureDescriptor descriptor = SpeakerTestProfiles.Descriptor();
        var single = new SpeakerPrototypeBank(
            ScriptedSpeakerService.OwnerAxis(),
            [new SpeakerPrototype("only", ScriptedSpeakerService.OwnerAxis(), 1f)]);
        var variant = new OwnerSpeakerProfileVariant(
            "v",
            descriptor.FrontendFingerprint,
            descriptor.TapId,
            DateTimeOffset.UtcNow,
            new Dictionary<SpeakerPurposeBank, SpeakerPrototypeBank> { [SpeakerPurposeBank.CleanNearEnd] = single },
            OwnerSpeakerProfileQuality.Empty);
        var profile = new OwnerSpeakerProfile(
            OwnerSpeakerProfile.OwnerSpeakerId,
            DateTimeOffset.UtcNow,
            SpeakerProfileStore.DescribeProvider(_speaker),
            [variant]);

        await Assert.ThrowsAsync<InvalidDataException>(() => SpeakerProfileStore.SaveAsync(_speaker, path, profile));
        Assert.False(File.Exists(path));
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }
}
