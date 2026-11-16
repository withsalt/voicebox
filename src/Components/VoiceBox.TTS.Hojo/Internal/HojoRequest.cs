using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.TTS.Hojo.Internal;

internal sealed record HojoRequest(string Voice, int MaxFrames, int? Seed, bool Sample,
    float Temperature, float TopP, float Penalty, int TopK, string Language)
{
    public static HojoRequest Create(HojoTTSOptions defaults, TtsSynthesisOptions? options)
    {
        float temperature = options?.AudioTemperature ?? defaults.Temperature;
        float topP = options?.AudioTopP ?? defaults.TopP;
        float penalty = options?.AudioRepetitionPenalty ?? defaults.RepetitionPenalty;
        int topK = options?.AudioTopK ?? 0;
        ValidateSampling(temperature, topP, penalty, topK);
        if (options?.MaxNewFrames is < 1) throw new ArgumentOutOfRangeException(nameof(options.MaxNewFrames));
        if (options?.Speed is { } speed && speed != 1f)
            throw new NotSupportedException("Hojo does not support changing speaking speed.");
        if (!string.IsNullOrWhiteSpace(options?.Instruct))
            throw new NotSupportedException("Hojo preset voices do not support style instructions.");
        if (options?.TextTemperature is not null || options?.TextTopP is not null || options?.TextTopK is not null)
            throw new NotSupportedException("Use AudioTemperature/AudioTopP/AudioTopK for Hojo audio-token sampling.");
        string language = options?.Language?.ToLowerInvariant() switch
        {
            null or "" or "auto" => "auto", "zh" or "chinese" => "zh", "en" or "english" => "en",
            _ => throw new NotSupportedException("Hojo supports auto, Chinese (zh), and English (en).")
        };
        return new(string.IsNullOrWhiteSpace(options?.Voice) ? defaults.DefaultVoice : options.Voice,
            Math.Min(defaults.MaxNewFrames, options?.MaxNewFrames ?? defaults.MaxNewFrames),
            options?.Seed ?? defaults.DefaultSeed, options?.DoSample ?? true, temperature, topP, penalty, topK, language);
    }

    public static void ValidateSampling(float temperature, float topP, float penalty, int topK)
    {
        if (!float.IsFinite(temperature) || temperature is < 0 or > 5) throw new ArgumentOutOfRangeException(nameof(temperature));
        if (!float.IsFinite(topP) || topP is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(topP));
        if (!float.IsFinite(penalty) || penalty is <= 0 or > 10) throw new ArgumentOutOfRangeException(nameof(penalty));
        if (topK is < 0 or > 17685) throw new ArgumentOutOfRangeException(nameof(topK));
    }
}
