using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Confucius4;

/// <summary>16 kHz mono PCM streaming options. One native inference runs at a time per service.</summary>
public sealed class Confucius4AsrOptions : OnnxExecutionOptions
{
    public Confucius4AsrOptions()
    {
        CudaTuning = new CudaExecutionTuning
        {
            ArenaExtendStrategy = "kSameAsRequested",
        };
    }

    public override int ThreadsNum { get; set; } = Math.Clamp(Environment.ProcessorCount / 2, 2, 8);
    public string ModelPath { get; set; } = "";
    public string? Language { get; set; } = "Chinese";
    public string Context { get; set; } = "";
    public int ChunkMilliseconds { get; set; } = 160;
    public int LookaheadMilliseconds { get; set; } = 160;
    public int RollbackTokens { get; set; } = 1;
    public int MaxNewTokens { get; set; } = 4;
    public int FinalMaxNewTokens { get; set; } = 256;
    /// <summary>Bounded utterance window; a full window is finalized before the next starts.</summary>
    public int MaxWindowMilliseconds { get; set; } = 16000;
    /// <summary>Maximum queued PCM duration per session. Overflow throws; samples are never silently dropped.</summary>
    public int MaxBufferedMilliseconds { get; set; } = 30000;
    public int MaxConcurrentSessions { get; set; } = 4;
    public bool EnableIoBinding { get; set; } = true;
    /// <summary>Keep expected ORT EP-assignment, Memcpy and ScatterND warnings quiet. Set false with ORT_LOG_VERBOSE=1 when diagnosing placement.</summary>
    public bool SuppressOrtWarnings { get; set; } = true;

    internal Confucius4AsrOptions Snapshot()
    {
        var copy = (Confucius4AsrOptions)MemberwiseClone();
        if (string.IsNullOrWhiteSpace(ModelPath)) throw new ArgumentException("ModelPath is required.");
        if (ChunkMilliseconds is < 80 or > 2000) throw new ArgumentOutOfRangeException(nameof(ChunkMilliseconds));
        if (LookaheadMilliseconds is < 0 or > 2000) throw new ArgumentOutOfRangeException(nameof(LookaheadMilliseconds));
        if (MaxWindowMilliseconds < ChunkMilliseconds + LookaheadMilliseconds || MaxWindowMilliseconds > 30000)
            throw new ArgumentOutOfRangeException(nameof(MaxWindowMilliseconds));
        if (RollbackTokens < 0 || RollbackTokens > 32) throw new ArgumentOutOfRangeException(nameof(RollbackTokens));
        if (MaxNewTokens is < 1 or > 512 || FinalMaxNewTokens is < 1 or > 2048)
            throw new ArgumentOutOfRangeException(nameof(MaxNewTokens));
        if (MaxBufferedMilliseconds is < 1 or > 300000) throw new ArgumentOutOfRangeException(nameof(MaxBufferedMilliseconds));
        if (MaxConcurrentSessions is < 1 or > 64) throw new ArgumentOutOfRangeException(nameof(MaxConcurrentSessions));
        if (!Enum.IsDefined(ExecutionProvider)) throw new ArgumentOutOfRangeException(nameof(ExecutionProvider));
        if (GpuDeviceId < 0) throw new ArgumentOutOfRangeException(nameof(GpuDeviceId));
        if (ExecutionProvider == OnnxExecutionProvider.DirectML)
            throw new NotSupportedException("Confucius4 packages are validated for CPU and CUDA.");
        copy.ModelPath = Path.GetFullPath(ModelPath);
        copy.Language = string.IsNullOrWhiteSpace(Language) ? null : Language.Trim();
        copy.Context = Context ?? "";
        if (CudaTuning is { } tuning)
            copy.CudaTuning = new CudaExecutionTuning
            {
                CudnnConvAlgoSearch = tuning.CudnnConvAlgoSearch,
                ArenaExtendStrategy = tuning.ArenaExtendStrategy,
                CudnnConvUseMaxWorkspace = tuning.CudnnConvUseMaxWorkspace,
                DoCopyInDefaultStream = tuning.DoCopyInDefaultStream,
                GpuMemLimitBytes = tuning.GpuMemLimitBytes,
                AdditionalOptions = tuning.AdditionalOptions is null ? null : new(tuning.AdditionalOptions),
            };
        return copy;
    }
}
