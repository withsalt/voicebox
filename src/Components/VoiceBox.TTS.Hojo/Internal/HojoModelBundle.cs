using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.TTS.Hojo.Internal;

internal sealed class HojoModelBundle
{
    public const int HiddenSize = 512;
    public int VocabularySize { get; }
    public int Layers { get; }
    public int SpeakerDimension { get; }
    public bool UsesSpeakerSlots { get; }
    public const int MaxPositions = 4096;
    public float[] Embeddings { get; }
    private readonly float[] _speakers;
    private readonly float[] _speakerVectors;
    private readonly HojoVoicePrompt[]? _prompts;
    private readonly Dictionary<string, int> _voiceIndices;
    public IReadOnlyList<TtsVoice> Voices { get; }
    public string DirectoryPath { get; }

    public HojoModelBundle(string directory)
    {
        DirectoryPath = directory;
        string manifestPath = Path.Combine(directory, "hojo.json");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("Prepare the Hojo bundle with scripts/hojo-tts-prepare/prepare.py first.", manifestPath);
        using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var root = document.RootElement;
        (UsesSpeakerSlots, VocabularySize, Layers, SpeakerDimension) = root.GetProperty("architecture").GetString() switch
        {
            "hojo-40m-v2" => (true, 17685, 10, 192),
            "hojo-80m" => (false, 17667, 17, 2048),
            _ => throw new InvalidDataException("Unsupported Hojo architecture."),
        };
        if (root.GetProperty("formatVersion").GetInt32() != 1
            || root.GetProperty("sampleRate").GetInt32() != 24000 || root.GetProperty("hiddenSize").GetInt32() != HiddenSize
            || root.GetProperty("vocabSize").GetInt32() != VocabularySize || root.GetProperty("layers").GetInt32() != Layers
            || root.GetProperty("maxPositions").GetInt32() != MaxPositions || root.GetProperty("fineHistory").GetInt32() != 60
            || root.GetProperty("lookaheadFrames").GetInt32() != 10)
            throw new InvalidDataException("Unsupported Hojo bundle architecture or streaming context.");
        string[] names = ["lm.onnx", "fine.onnx", "decoder.onnx", "embeddings.f32", "speaker_vectors.f32", "tokenizer.json",
            UsesSpeakerSlots ? "speakers.f32" : "voice_prompts.json"];
        foreach (string name in names)
        {
            using var stream = File.OpenRead(Path.Combine(directory, name));
            string actual = Convert.ToHexString(SHA256.HashData(stream));
            if (!string.Equals(actual, root.GetProperty("files").GetProperty(name).GetString(), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Hojo bundle integrity check failed: {name}.");
        }
        string[] ids = root.GetProperty("voices").EnumerateArray().Select(v => v.GetString() ?? "").ToArray();
        if (ids.Length is < 1 or > 128 || ids.Any(string.IsNullOrWhiteSpace) || ids.Distinct(StringComparer.Ordinal).Count() != ids.Length)
            throw new InvalidDataException("Invalid voice list.");
        _voiceIndices = ids.Select((name, index) => (name, index)).ToDictionary(x => x.name, x => x.index, StringComparer.Ordinal);
        Voices = Array.AsReadOnly(ids.Select(id => new TtsVoice
        {
            Name = id, Language = id.StartsWith("hojo_zh_", StringComparison.Ordinal) ? "zh" : "en",
            Description = root.TryGetProperty("voiceDescriptions", out var descriptions) && descriptions.TryGetProperty(id, out var description)
                ? description.GetString() : null,
        }).ToArray());
        Embeddings = ReadFloats("embeddings.f32", VocabularySize * HiddenSize);
        _speakers = UsesSpeakerSlots ? ReadFloats("speakers.f32", ids.Length * 16 * HiddenSize) : [];
        _speakerVectors = ReadFloats("speaker_vectors.f32", ids.Length * SpeakerDimension);
        if (!UsesSpeakerSlots)
        {
            using var prompts = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "voice_prompts.json")));
            _prompts = ids.Select(id =>
            {
                var entry = prompts.RootElement.GetProperty(id);
                string text = entry.GetProperty("text").GetString() ?? "";
                int[] codes = entry.GetProperty("codes").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                if (text.Length > 2048 || codes.Length is < 1 or > 1500 || codes.Any(code => code is < 0 or >= 16384))
                    throw new InvalidDataException("Invalid fixed voice reference prompt.");
                HojoTokenizer.ValidateText(text);
                return new HojoVoicePrompt(text, codes);
            }).ToArray();
        }
    }

    public int VoiceIndex(string voice) => _voiceIndices.TryGetValue(voice, out int index)
        ? index : throw new ArgumentException($"Unknown Hojo voice: {voice}.", nameof(voice));
    public ReadOnlySpan<float> Speaker(int index) => _speakers.AsSpan(index * 16 * HiddenSize, 16 * HiddenSize);
    public float[] SpeakerVector(int index) => _speakerVectors.AsSpan(index * SpeakerDimension, SpeakerDimension).ToArray();
    public HojoVoicePrompt? VoicePrompt(int index) => _prompts?[index];
    public ReadOnlySpan<float> TokenEmbedding(int token) => Embeddings.AsSpan(token * HiddenSize, HiddenSize);

    private float[] ReadFloats(string name, int count)
    {
        using var stream = File.OpenRead(Path.Combine(DirectoryPath, name));
        if (stream.Length != count * sizeof(float)) throw new InvalidDataException($"Unexpected tensor length: {name}.");
        var result = new float[count];
        stream.ReadExactly(MemoryMarshal.AsBytes(result.AsSpan()));
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Hojo bundles require little-endian IEEE float32.");
        if (result.Any(value => !float.IsFinite(value))) throw new InvalidDataException($"Non-finite model tensor: {name}.");
        return result;
    }
}

internal sealed record HojoVoicePrompt(string Text, int[] Codes);
