using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.Hojo;

/// <summary>Hojo 80M / 40M V2 CPU options. Prepare original or LM INT8 bundles with scripts/hojo-tts-prepare.</summary>
public sealed class HojoTTSOptions : OnnxExecutionOptions
{
    public string ModelPath { get; set; } = "";
    public string DefaultVoice { get; set; } = "hojo_zh_f_01";
    public override int ThreadsNum { get; set; } = 2;
    /// <summary>Output block size in 20ms frames. The first block also needs ten lookahead frames.</summary>
    public int ChunkFrames { get; set; } = 40;
    /// <summary>Hard limit per utterance, including non-audio generated tokens. Reaching it throws.</summary>
    public int MaxNewFrames { get; set; } = 1500;
    public int MaxTextLength { get; set; } = 2048;
    /// <summary>Active and waiting requests combined; overflow fails explicitly.</summary>
    public int MaxPendingRequests { get; set; } = 8;
    /// <summary>Bounded PCM queue; a slow consumer pauses generation.</summary>
    public int BufferedChunks { get; set; } = 2;
    public bool NormalizeText { get; set; } = true;
    public int? DefaultSeed { get; set; } = 42;
    public float Temperature { get; set; } = 0.8f;
    public float TopP { get; set; } = 0.95f;
    public float RepetitionPenalty { get; set; } = 1.1f;

    internal HojoTTSOptions Snapshot()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(DefaultVoice);
        if (ThreadsNum is < 1 or > 128) throw new ArgumentOutOfRangeException(nameof(ThreadsNum));
        if (ChunkFrames is < 5 or > 100) throw new ArgumentOutOfRangeException(nameof(ChunkFrames));
        if (MaxNewFrames is < 1 or > 4000) throw new ArgumentOutOfRangeException(nameof(MaxNewFrames));
        if (MaxTextLength is < 1 or > 8192) throw new ArgumentOutOfRangeException(nameof(MaxTextLength));
        if (MaxPendingRequests is < 1 or > 128) throw new ArgumentOutOfRangeException(nameof(MaxPendingRequests));
        if (BufferedChunks is < 1 or > 16) throw new ArgumentOutOfRangeException(nameof(BufferedChunks));
        if (ExecutionProvider != OnnxExecutionProvider.Cpu)
            throw new NotSupportedException("Hojo deployment is validated for CPU. GPU providers are not supported.");
        Internal.HojoRequest.ValidateSampling(Temperature, TopP, RepetitionPenalty, 0);
        var copy = (HojoTTSOptions)MemberwiseClone();
        copy.ModelPath = Path.GetFullPath(ModelPath);
        return copy;
    }
}
