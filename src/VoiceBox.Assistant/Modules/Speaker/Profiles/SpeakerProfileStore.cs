using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Serialization;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Profiles;

/// <summary>
/// owner-speaker-profile-v2 persistence. Only format 6 is readable; every earlier format, and
/// in particular the retired <c>processed-capture-v1</c> tap, is rejected so identity-tap audio is
/// never compared against a profile enrolled on the final processed audio path.
/// </summary>
internal static class SpeakerProfileStore
{
    public const int CurrentFormatVersion = 6;
    public const string CurrentSchema = "owner-speaker-profile-v2";
    public const string RetiredProcessedCaptureTap = "processed-capture-v1";
    private const int MaxProfileBytes = 512 * 1024;
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static OwnerSpeakerProfile Load(ISpeakerService service, VoiceAssistantSpeakerOptions options)
    {
        ArgumentNullException.ThrowIfNull(service);
        ArgumentNullException.ThrowIfNull(options);
        return Load(service, options.ResolveProfilePath());
    }

    public static OwnerSpeakerProfile Load(ISpeakerService service, string path)
    {
        ArgumentNullException.ThrowIfNull(service);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        return Decode(ReadDocument(path), service, path);
    }

    /// <summary>
    /// Loads the profile if present, or returns null when no file exists. Any other problem is
    /// surfaced so a corrupt or outdated profile is never silently treated as "not enrolled".
    /// </summary>
    public static OwnerSpeakerProfile? TryLoadExisting(ISpeakerService service, string path)
    {
        if (!File.Exists(path))
            return null;
        return Load(service, path);
    }

    public static async Task SaveAsync(
        ISpeakerService service,
        VoiceAssistantSpeakerOptions options,
        OwnerSpeakerProfile profile,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        await SaveAsync(service, options.ResolveProfilePath(), profile, ct).ConfigureAwait(false);
    }

    public static async Task SaveAsync(
        ISpeakerService service,
        string path,
        OwnerSpeakerProfile profile,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(service);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(profile);
        ValidateProfileForProvider(profile, service, path);

        string? directory = Path.GetDirectoryName(path);
        if (string.IsNullOrEmpty(directory))
            throw new InvalidOperationException($"Cannot resolve speaker profile directory from '{path}'.");

        Directory.CreateDirectory(directory);
        string tempPath = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        ProfileDocument document = Encode(profile);
        try
        {
            await using (var stream = new FileStream(
                tempPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4096,
                FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await JsonSerializer.SerializeAsync(stream, document, JsonOptions, ct).ConfigureAwait(false);
                await stream.FlushAsync(ct).ConfigureAwait(false);
            }

            File.Move(tempPath, path, overwrite: true);
        }
        finally
        {
            try { File.Delete(tempPath); }
            catch { }
        }
    }

    public static bool IsUsable(ISpeakerService service, VoiceAssistantSpeakerOptions options)
    {
        try
        {
            _ = Load(service, options);
            return true;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Selects the exact front-end variant and verifies every bank the requested purposes need.
    /// No nearest-variant or clean-bank fallback exists: a missing variant or bank fails closed.
    /// </summary>
    public static OwnerSpeakerProfileVariant RequireVariant(
        OwnerSpeakerProfile profile,
        SpeakerCaptureDescriptor descriptor,
        IReadOnlyCollection<SpeakerPurposeBank> requiredBanks,
        string source)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(descriptor);
        OwnerSpeakerProfileVariant? variant = profile.FindVariant(descriptor.FrontendFingerprint);
        if (variant is null)
        {
            throw new InvalidOperationException(
                $"Speaker profile '{source}' has no variant for the current acoustic front end " +
                $"'{descriptor.FrontendFingerprint}'. Available: " +
                $"{string.Join("; ", profile.Variants.Select(static v => v.FrontendFingerprint))}. " +
                "Re-enroll the owner on the current AEC/APM configuration.");
        }
        if (!string.Equals(variant.TapId, descriptor.TapId, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Speaker profile '{source}' variant uses tap '{variant.TapId}', current tap is '{descriptor.TapId}'.");
        }

        foreach (SpeakerPurposeBank bank in requiredBanks)
        {
            if (!variant.TryGetBank(bank, out SpeakerPrototypeBank value) || !value.IsUsable)
            {
                throw new InvalidOperationException(
                    $"Speaker profile '{source}' variant '{variant.VariantId}' has no usable '{bank}' bank. " +
                    "Complete the corresponding enrollment before enabling this purpose.");
            }
        }

        return variant;
    }

    public static SpeakerProfileProvider DescribeProvider(ISpeakerService service)
        => new(
            ServiceId: service.GetType().Name,
            ServiceType: service.GetType().FullName ?? service.GetType().Name,
            ModelDigest: SpeakerCaptureDescriptor.Unknown,
            SampleRate: service.SampleRate,
            EmbeddingSize: service.EmbeddingSize,
            WindowSamples: service.StreamingWindowSamples,
            HopSamples: service.StreamingHopSamples);

    private static void ValidateProfileForProvider(OwnerSpeakerProfile profile, ISpeakerService service, string path)
    {
        if (!string.Equals(profile.SpeakerId, OwnerSpeakerProfile.OwnerSpeakerId, StringComparison.Ordinal))
            throw new InvalidDataException($"Speaker profile '{path}' does not contain the owner profile.");

        SpeakerProfileProvider provider = profile.Provider;
        string serviceType = service.GetType().FullName ?? service.GetType().Name;
        if (!string.Equals(provider.ServiceType, serviceType, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Speaker profile '{path}' was created for '{provider.ServiceType}', current service is '{serviceType}'.");
        }
        if (provider.SampleRate != service.SampleRate)
            throw new InvalidDataException(
                $"Speaker profile '{path}' sample rate {provider.SampleRate}Hz does not match {service.SampleRate}Hz.");
        if (provider.EmbeddingSize != service.EmbeddingSize)
            throw new InvalidDataException(
                $"Speaker profile '{path}' embedding size {provider.EmbeddingSize} does not match {service.EmbeddingSize}.");
        if (provider.WindowSamples != service.StreamingWindowSamples ||
            provider.HopSamples != service.StreamingHopSamples)
        {
            throw new InvalidDataException(
                $"Speaker profile '{path}' was created for {provider.WindowSamples}/{provider.HopSamples} " +
                $"window/hop samples; current service is {service.StreamingWindowSamples}/{service.StreamingHopSamples}.");
        }
        if (profile.Variants.Count == 0)
            throw new InvalidDataException($"Speaker profile '{path}' has no front-end variants.");
        if (profile.Variants.Count > OwnerSpeakerProfile.MaxVariants)
            throw new InvalidDataException($"Speaker profile '{path}' has more than {OwnerSpeakerProfile.MaxVariants} variants.");

        var fingerprints = new HashSet<string>(StringComparer.Ordinal);
        foreach (OwnerSpeakerProfileVariant variant in profile.Variants)
        {
            if (string.IsNullOrWhiteSpace(variant.FrontendFingerprint) ||
                string.IsNullOrWhiteSpace(variant.TapId) ||
                string.IsNullOrWhiteSpace(variant.VariantId))
            {
                throw new InvalidDataException($"Speaker profile '{path}' has a variant without identity fields.");
            }
            if (string.Equals(variant.TapId, RetiredProcessedCaptureTap, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Speaker profile '{path}' uses the retired audio tap '{RetiredProcessedCaptureTap}'. Re-enroll on the identity tap.");
            }
            if (!fingerprints.Add(variant.FrontendFingerprint))
                throw new InvalidDataException($"Speaker profile '{path}' has duplicate variant '{variant.FrontendFingerprint}'.");
            if (variant.Banks.Count == 0)
                throw new InvalidDataException($"Speaker profile '{path}' variant '{variant.VariantId}' has no banks.");
            foreach ((SpeakerPurposeBank bankKind, SpeakerPrototypeBank bank) in variant.Banks)
            {
                ValidateEmbedding(bank.Centroid, provider.EmbeddingSize, $"{path}:{variant.VariantId}:{bankKind}:centroid");
                if (bank.Prototypes.Count < SpeakerPrototypeBank.MinPrototypes ||
                    bank.Prototypes.Count > SpeakerPrototypeBank.MaxPrototypes)
                {
                    throw new InvalidDataException(
                        $"Speaker profile '{path}' bank '{bankKind}' must hold {SpeakerPrototypeBank.MinPrototypes}..{SpeakerPrototypeBank.MaxPrototypes} prototypes.");
                }
                foreach (SpeakerPrototype prototype in bank.Prototypes)
                    ValidateEmbedding(prototype.Embedding, provider.EmbeddingSize, $"{path}:{variant.VariantId}:{bankKind}:{prototype.Id}");
            }
        }
    }

    private static ProfileDocument ReadDocument(string path)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (stream.Length > MaxProfileBytes)
                throw new InvalidDataException(
                    $"Speaker profile '{path}' is unexpectedly large ({stream.Length} bytes).");
            return JsonSerializer.Deserialize<ProfileDocument>(stream, JsonOptions)
                ?? throw new InvalidDataException($"Speaker profile '{path}' is empty.");
        }
        catch (FileNotFoundException ex)
        {
            throw new InvalidOperationException(
                $"未找到本人声纹 profile '{path}'。请在启动 VoiceAssistant 前完成说话人注册。", ex);
        }
        catch (DirectoryNotFoundException ex)
        {
            throw new InvalidOperationException(
                $"未找到本人声纹 profile '{path}'。请在启动 VoiceAssistant 前完成说话人注册。", ex);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"Speaker profile '{path}' is not valid JSON.", ex);
        }
    }

    private static OwnerSpeakerProfile Decode(ProfileDocument document, ISpeakerService service, string path)
    {
        if (document.FormatVersion != CurrentFormatVersion ||
            !string.Equals(document.Schema, CurrentSchema, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Speaker profile '{path}' uses unsupported format version {document.FormatVersion} " +
                $"('{document.Schema ?? document.AudioTap ?? "unknown"}'). Only {CurrentSchema} (format {CurrentFormatVersion}) " +
                "is accepted; please re-enroll the owner.");
        }
        if (document.Provider is null)
            throw new InvalidDataException($"Speaker profile '{path}' has no provider section.");

        var provider = new SpeakerProfileProvider(
            document.Provider.ServiceId ?? string.Empty,
            document.Provider.ServiceType ?? string.Empty,
            string.IsNullOrWhiteSpace(document.Provider.ModelDigest)
                ? SpeakerCaptureDescriptor.Unknown
                : document.Provider.ModelDigest,
            document.Provider.SampleRate,
            document.Provider.EmbeddingSize,
            document.Provider.WindowSamples,
            document.Provider.HopSamples);

        var variants = new List<OwnerSpeakerProfileVariant>();
        foreach (VariantDocument variant in document.Variants ?? [])
        {
            var banks = new Dictionary<SpeakerPurposeBank, SpeakerPrototypeBank>();
            if (variant.Banks is not null)
            {
                foreach ((string name, BankDocument bank) in variant.Banks)
                {
                    SpeakerPurposeBank kind = ParseBank(name, path);
                    if (bank.Centroid is null)
                        throw new InvalidDataException($"Speaker profile '{path}' bank '{name}' has no centroid.");
                    var prototypes = new List<SpeakerPrototype>();
                    foreach (PrototypeDocument prototype in bank.Prototypes ?? [])
                    {
                        prototypes.Add(new SpeakerPrototype(
                            prototype.Id ?? string.Empty,
                            DecodeEmbedding(prototype.Embedding ?? string.Empty, provider.EmbeddingSize, $"{path}:{name}:{prototype.Id}"),
                            prototype.Quality));
                    }
                    banks[kind] = new SpeakerPrototypeBank(
                        DecodeEmbedding(bank.Centroid, provider.EmbeddingSize, $"{path}:{name}:centroid"),
                        prototypes.AsReadOnly());
                }
            }

            variants.Add(new OwnerSpeakerProfileVariant(
                variant.VariantId ?? string.Empty,
                variant.FrontendFingerprint ?? string.Empty,
                variant.TapId ?? string.Empty,
                variant.CreatedAt,
                banks,
                variant.Quality is null
                    ? OwnerSpeakerProfileQuality.Empty
                    : new OwnerSpeakerProfileQuality(
                        variant.Quality.SegmentCount,
                        variant.Quality.ConfirmationCount,
                        variant.Quality.PairwiseP10,
                        variant.Quality.PairwiseMedian,
                        variant.Quality.ConfirmationMinScore,
                        variant.Quality.WithinSpeakerSpread)));
        }

        var profile = new OwnerSpeakerProfile(
            document.SpeakerId ?? string.Empty,
            document.CreatedAt,
            provider,
            variants.AsReadOnly());
        ValidateProfileForProvider(profile, service, path);
        return profile;
    }

    private static ProfileDocument Encode(OwnerSpeakerProfile profile)
        => new()
        {
            FormatVersion = CurrentFormatVersion,
            Schema = CurrentSchema,
            SpeakerId = profile.SpeakerId,
            CreatedAt = profile.CreatedAt,
            Provider = new ProviderDocument
            {
                ServiceId = profile.Provider.ServiceId,
                ServiceType = profile.Provider.ServiceType,
                ModelDigest = profile.Provider.ModelDigest,
                SampleRate = profile.Provider.SampleRate,
                EmbeddingSize = profile.Provider.EmbeddingSize,
                WindowSamples = profile.Provider.WindowSamples,
                HopSamples = profile.Provider.HopSamples,
            },
            Variants = profile.Variants.Select(static variant => new VariantDocument
            {
                VariantId = variant.VariantId,
                FrontendFingerprint = variant.FrontendFingerprint,
                TapId = variant.TapId,
                CreatedAt = variant.CreatedAt,
                Banks = variant.Banks.ToDictionary(
                    static pair => BankName(pair.Key),
                    static pair => new BankDocument
                    {
                        Centroid = EncodeEmbedding(pair.Value.Centroid),
                        Prototypes = pair.Value.Prototypes.Select(static prototype => new PrototypeDocument
                        {
                            Id = prototype.Id,
                            Embedding = EncodeEmbedding(prototype.Embedding),
                            Quality = prototype.Quality,
                        }).ToList(),
                    },
                    StringComparer.Ordinal),
                Quality = new QualityDocument
                {
                    SegmentCount = variant.Quality.SegmentCount,
                    ConfirmationCount = variant.Quality.ConfirmationCount,
                    PairwiseP10 = variant.Quality.PairwiseP10,
                    PairwiseMedian = variant.Quality.PairwiseMedian,
                    ConfirmationMinScore = variant.Quality.ConfirmationMinScore,
                    WithinSpeakerSpread = variant.Quality.WithinSpeakerSpread,
                },
            }).ToList(),
        };

    internal static string BankName(SpeakerPurposeBank bank)
        => bank switch
        {
            SpeakerPurposeBank.CleanNearEnd => "cleanNearEnd",
            SpeakerPurposeBank.DoubleTalkNearEnd => "doubleTalkNearEnd",
            _ => throw new ArgumentOutOfRangeException(nameof(bank), bank, null),
        };

    private static SpeakerPurposeBank ParseBank(string name, string path)
        => name switch
        {
            "cleanNearEnd" => SpeakerPurposeBank.CleanNearEnd,
            "doubleTalkNearEnd" => SpeakerPurposeBank.DoubleTalkNearEnd,
            _ => throw new InvalidDataException($"Speaker profile '{path}' contains unknown bank '{name}'."),
        };

    internal static string EncodeEmbedding(ReadOnlySpan<float> embedding)
    {
        var bytes = new byte[embedding.Length * sizeof(float)];
        for (int i = 0; i < embedding.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(i * sizeof(float), sizeof(float)),
                BitConverter.SingleToInt32Bits(embedding[i]));
        }
        return Convert.ToBase64String(bytes);
    }

    internal static float[] DecodeEmbedding(string encoded, int embeddingSize, string source)
    {
        byte[] bytes;
        try { bytes = Convert.FromBase64String(encoded); }
        catch (FormatException ex)
        {
            throw new InvalidDataException($"Speaker embedding from '{source}' is not valid Base64.", ex);
        }

        if (bytes.Length != embeddingSize * sizeof(float))
        {
            throw new InvalidDataException(
                $"Speaker embedding from '{source}' byte length {bytes.Length} is invalid for {embeddingSize} dimensions.");
        }

        var embedding = new float[embeddingSize];
        for (int i = 0; i < embedding.Length; i++)
        {
            int bits = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * sizeof(float), sizeof(float)));
            embedding[i] = BitConverter.Int32BitsToSingle(bits);
        }
        ValidateEmbedding(embedding, embeddingSize, source);
        return embedding;
    }

    internal static void ValidateEmbedding(ReadOnlySpan<float> embedding, int embeddingSize, string source)
    {
        if (embedding.Length != embeddingSize)
        {
            throw new InvalidDataException(
                $"Speaker embedding from '{source}' has {embedding.Length} dimensions; expected {embeddingSize}.");
        }

        double sumSq = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            float value = embedding[i];
            if (!float.IsFinite(value))
                throw new InvalidDataException($"Speaker embedding from '{source}' contains non-finite values.");
            sumSq += (double)value * value;
        }
        if (!(sumSq > 0) || !double.IsFinite(sumSq))
            throw new InvalidDataException($"Speaker embedding from '{source}' is empty or invalid.");
    }

    private sealed class ProfileDocument
    {
        public int FormatVersion { get; set; }
        public string? Schema { get; set; }
        public string? SpeakerId { get; set; }
        public DateTimeOffset CreatedAt { get; set; }
        public ProviderDocument? Provider { get; set; }
        public List<VariantDocument>? Variants { get; set; }

        /// <summary>Present only in retired format-5 files; read solely to produce a precise error.</summary>
        public string? AudioTap { get; set; }
    }

    private sealed class ProviderDocument
    {
        public string? ServiceId { get; set; }
        public string? ServiceType { get; set; }
        public string? ModelDigest { get; set; }
        public int SampleRate { get; set; }
        public int EmbeddingSize { get; set; }
        public int WindowSamples { get; set; }
        public int HopSamples { get; set; }
    }

    private sealed class VariantDocument
    {
        public string? VariantId { get; set; }
        public string? FrontendFingerprint { get; set; }
        public string? TapId { get; set; }
        public DateTimeOffset CreatedAt { get; set; }
        public Dictionary<string, BankDocument>? Banks { get; set; }
        public QualityDocument? Quality { get; set; }
    }

    private sealed class BankDocument
    {
        public string? Centroid { get; set; }
        public List<PrototypeDocument>? Prototypes { get; set; }
    }

    private sealed class PrototypeDocument
    {
        public string? Id { get; set; }
        public string? Embedding { get; set; }
        public float Quality { get; set; }
    }

    private sealed class QualityDocument
    {
        public int SegmentCount { get; set; }
        public int ConfirmationCount { get; set; }
        public double PairwiseP10 { get; set; }
        public double PairwiseMedian { get; set; }
        public double ConfirmationMinScore { get; set; }
        public double WithinSpeakerSpread { get; set; }
    }
}
