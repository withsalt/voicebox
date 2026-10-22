using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed record EvaluationDatasetManifest
{
    public const string FileName = "manifest.json";
    public const string CurrentSchemaVersion = "2.0";
    public const string LegacySchemaVersion = "1.0";

    public required string SchemaVersion { get; init; }
    public required string DatasetId { get; init; }
    public required DateTimeOffset CreatedUtc { get; init; }
    public required int Seed { get; init; }
    public required string Materialization { get; init; }
    public required IReadOnlyDictionary<string, string> SourceRoots { get; init; }
    public DatasetGeneratorMetadata? Generator { get; init; }
    public required IReadOnlyList<PreparedAcousticCase> AcousticCases { get; init; }
    public required IReadOnlyList<PreparedSpeechAsset> SpeechAssets { get; init; }
    public required IReadOnlyList<PreparedRirAsset> RirAssets { get; init; }
    public required IReadOnlyList<PreparedNoiseAsset> NoiseAssets { get; init; }
    public IReadOnlyList<PreparedLibriSpeechAsset> LibriSpeechAssets { get; init; } = [];
    public required IReadOnlyList<PreparedEndToEndCase> EndToEndCases { get; init; }

    public static EvaluationDatasetManifest Load(string root)
    {
        string path = Path.Combine(Path.GetFullPath(root), FileName);
        if (!File.Exists(path))
            throw new FileNotFoundException("Prepared evaluation dataset manifest not found.", path);

        EvaluationDatasetManifest manifest =
            JsonSerializer.Deserialize<EvaluationDatasetManifest>(
                File.ReadAllText(path),
                JsonOptions)
            ?? throw new InvalidDataException($"Invalid prepared dataset manifest: {path}");
        if (manifest.SchemaVersion is not (CurrentSchemaVersion or LegacySchemaVersion))
        {
            throw new InvalidDataException(
                $"Unsupported prepared dataset schema '{manifest.SchemaVersion}', " +
                $"expected '{LegacySchemaVersion}' or '{CurrentSchemaVersion}'.");
        }
        if (manifest.SchemaVersion == CurrentSchemaVersion)
            manifest.ValidateV2();
        return manifest;
    }

    public void Save(string root)
    {
        if (SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidOperationException(
                $"New manifests must use schema '{CurrentSchemaVersion}'.");
        }
        ValidateV2();
        string path = Path.Combine(Path.GetFullPath(root), FileName);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));
    }

    public static string ResolvePath(string root, string relativePath)
    {
        string fullRoot = Path.GetFullPath(root);
        string fullPath = Path.GetFullPath(
            relativePath.Replace('/', Path.DirectorySeparatorChar),
            fullRoot);
        string boundary = fullRoot.EndsWith(Path.DirectorySeparatorChar)
            ? fullRoot
            : fullRoot + Path.DirectorySeparatorChar;
        if (!fullPath.StartsWith(boundary, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Manifest path escapes the dataset root: {relativePath}");
        return fullPath;
    }

    private void ValidateV2()
    {
        if (Generator is null ||
            string.IsNullOrWhiteSpace(Generator.Commit) ||
            Generator.Options.Count == 0)
        {
            throw new InvalidDataException(
                "Schema 2.0 requires generator commit and options metadata.");
        }

        foreach (PreparedAcousticCase item in AcousticCases)
        {
            if (string.IsNullOrWhiteSpace(item.Split) ||
                string.IsNullOrWhiteSpace(item.MicrophoneSha256) ||
                string.IsNullOrWhiteSpace(item.RenderSha256) ||
                (item.NearEndReferencePath is not null &&
                 string.IsNullOrWhiteSpace(item.NearEndReferenceSha256)))
            {
                throw new InvalidDataException(
                    $"Acoustic case '{item.Id}' is missing v2 split or source hashes.");
            }
        }

        foreach (PreparedSpeechAsset item in SpeechAssets)
        {
            if (string.IsNullOrWhiteSpace(item.Split) ||
                string.IsNullOrWhiteSpace(item.AudioSha256) ||
                item.DurationMs <= 0)
            {
                throw new InvalidDataException(
                    $"Speech asset '{item.Id}' is missing v2 split, duration, or source hash.");
            }
        }

        foreach (PreparedRirAsset item in RirAssets)
        {
            if (string.IsNullOrWhiteSpace(item.GeometryId) ||
                string.IsNullOrWhiteSpace(item.MicrophoneSetupId) ||
                string.IsNullOrWhiteSpace(item.MicrophoneChannelId) ||
                string.IsNullOrWhiteSpace(item.SpeakerSetupId) ||
                string.IsNullOrWhiteSpace(item.AudioSha256))
            {
                throw new InvalidDataException(
                    $"RIR asset '{item.Id}' is missing v2 geometry or source hash.");
            }
        }

        foreach (PreparedNoiseAsset item in NoiseAssets)
        {
            if (string.IsNullOrWhiteSpace(item.AudioSha256))
            {
                throw new InvalidDataException(
                    $"Noise asset '{item.Id}' is missing its v2 source hash.");
            }
        }

        foreach (PreparedLibriSpeechAsset item in LibriSpeechAssets)
        {
            if (string.IsNullOrWhiteSpace(item.AudioSha256))
            {
                throw new InvalidDataException(
                    $"BUT LibriSpeech asset '{item.Id}' is missing its v2 source hash.");
            }
        }

        foreach (PreparedEndToEndCase item in EndToEndCases)
        {
            if (item.ExpectHeard is null ||
                item.ExpectDuck is null ||
                item.ExpectAccepted is null ||
                item.ExpectPlaybackStop is null ||
                item.ExpectPlaybackResume is null ||
                string.IsNullOrWhiteSpace(item.Scenario) ||
                string.IsNullOrWhiteSpace(item.Split) ||
                item.Suites.Count == 0 ||
                string.IsNullOrWhiteSpace(item.GeometryId) ||
                string.IsNullOrWhiteSpace(item.SpeakerPolicy))
            {
                throw new InvalidDataException(
                    $"End-to-end case '{item.Id}' is missing v2 layered truth or grouping metadata.");
            }
        }
    }

    private static JsonSerializerOptions JsonOptions { get; } = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}

internal sealed record DatasetGeneratorMetadata
{
    public required string Commit { get; init; }
    public required bool Dirty { get; init; }
    public required IReadOnlyDictionary<string, string> Options { get; init; }
}

internal readonly record struct OracleSpeechInterval(
    double StartMs,
    double EndMs);

internal sealed record PreparedAcousticCase
{
    public required string Id { get; init; }
    public required string Corpus { get; init; }
    public required string Scenario { get; init; }
    public required bool ExpectedNearEnd { get; init; }
    public required string MicrophonePath { get; init; }
    public required string RenderPath { get; init; }
    public string? NearEndReferencePath { get; init; }
    public double NearEndScale { get; init; } = 1;
    public double? SignalToEchoRatioDb { get; init; }
    public bool? FarEndNonlinear { get; init; }
    public bool? FarEndNoisy { get; init; }
    public bool? NearEndNoisy { get; init; }
    public string? DeviceClass { get; init; }
    public string? DeviceSession { get; init; }
    public required string ClusterId { get; init; }
    public string Split { get; init; } = "test";
    public IReadOnlyList<string> Suites { get; init; } = ["release-acoustic"];
    public string? GeometryId { get; init; }
    public string? MicrophoneSha256 { get; init; }
    public string? RenderSha256 { get; init; }
    public string? NearEndReferenceSha256 { get; init; }
}

internal sealed record PreparedSpeechAsset
{
    public required string Id { get; init; }
    public required string Corpus { get; init; }
    public required string SpeakerId { get; init; }
    public required string UtteranceId { get; init; }
    public required string ClusterId { get; init; }
    public required string Text { get; init; }
    public required string AudioPath { get; init; }
    public required bool Noisy { get; init; }
    public string Split { get; init; } = "test";
    public double DurationMs { get; init; }
    public string? AudioSha256 { get; init; }
}

internal sealed record PreparedRirAsset
{
    public required string Id { get; init; }
    public required string RoomId { get; init; }
    public required string Path { get; init; }
    public string GeometryId { get; init; } = string.Empty;
    public string MicrophoneSetupId { get; init; } = string.Empty;
    public string MicrophoneChannelId { get; init; } = string.Empty;
    public string SpeakerSetupId { get; init; } = string.Empty;
    public string? AudioSha256 { get; init; }
}

internal sealed record PreparedNoiseAsset
{
    public required string Id { get; init; }
    public required string RoomId { get; init; }
    public required string Path { get; init; }
    public string Corpus { get; init; } = "but-reverbdb";
    public string Kind { get; init; } = "room-ambience";
    public string? GeometryId { get; init; }
    public string? AudioSha256 { get; init; }
}

internal sealed record PreparedLibriSpeechAsset
{
    public required string Id { get; init; }
    public required string RoomId { get; init; }
    public required string UtteranceId { get; init; }
    public required string Path { get; init; }
    public string? AudioSha256 { get; init; }
}

internal sealed record PreparedEndToEndCase
{
    public required string Id { get; init; }
    public required bool ExpectedNearEnd { get; init; }
    public string? SpeechAssetId { get; init; }
    public required string PromptText { get; init; }
    public required string EchoRirId { get; init; }
    public string? NearRirId { get; init; }
    public string? NoiseAssetId { get; init; }
    public required int NearEndOnsetMs { get; init; }
    public required double SignalToEchoRatioDb { get; init; }
    public required double SignalToNoiseRatioDb { get; init; }
    public required int EchoDelayMs { get; init; }
    public required int ClockDriftPpm { get; init; }
    public required bool FarEndNonlinear { get; init; }

    /// <summary>
    /// Echo return loss in dB: how much the loudspeaker-to-microphone path attenuates the played
    /// signal. It separates loudspeaker drive level from the echo level observed at the microphone,
    /// which is what decides how much the loudspeaker nonlinearity distorts. Optional: when absent
    /// the renderer applies <see cref="AcousticSceneRenderer.NominalEchoReturnLossDb"/>, so existing
    /// datasets stay loadable without regeneration.
    /// </summary>
    public double? EchoReturnLossDb { get; init; }

    public required int CallbackSamples { get; init; }
    public required string ClusterId { get; init; }
    public string Scenario { get; init; } = "valid-command";
    public string Split { get; init; } = "test";
    public IReadOnlyList<string> Suites { get; init; } = ["release-e2e"];
    public string GeometryId { get; init; } = string.Empty;
    public string SpeakerPolicy { get; init; } = "any-speaker";
    public string? ReferenceText { get; init; }
    public bool? ExpectHeard { get; init; }
    public bool? ExpectDuck { get; init; }
    public bool? ExpectAccepted { get; init; }
    public bool? ExpectPlaybackStop { get; init; }
    public bool? ExpectPlaybackResume { get; init; }
    public IReadOnlyList<OracleSpeechInterval> OracleSpeechIntervals { get; init; } = [];
    public int? RenderGapStartMs { get; init; }
    public int RenderGapDurationMs { get; init; }

    [JsonIgnore]
    public bool OracleExpectHeard => ExpectHeard ?? ExpectedNearEnd;

    [JsonIgnore]
    public bool OracleExpectDuck => ExpectDuck ?? ExpectedNearEnd;

    [JsonIgnore]
    public bool OracleExpectAccepted => ExpectAccepted ?? ExpectedNearEnd;

    [JsonIgnore]
    public bool OracleExpectPlaybackStop => ExpectPlaybackStop ?? ExpectedNearEnd;

    [JsonIgnore]
    public bool OracleExpectPlaybackResume => ExpectPlaybackResume ?? false;
}
