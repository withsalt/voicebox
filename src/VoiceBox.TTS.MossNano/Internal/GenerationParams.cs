using System.Collections.Generic;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 单次 voice-clone 推理请求的输入: 完整的 prompt token 序列 (每行宽 n_vq+1)。
    /// 与 Python `build_voice_clone_request_rows` 返回值 1:1 对齐。
    /// </summary>
    internal readonly record struct VoiceCloneRequestRows(int[][] InputIds, int[] AttentionMask);

    /// <summary>
    /// 单步生成的 frame 回调。on_frame(generatedFrames, stepIndex, frame)。
    /// </summary>
    internal delegate void OnFrameCallback(IReadOnlyList<int[]> generatedFrames, int stepIndex, int[] frame);

    /// <summary>
    /// 生成参数。复制自 manifest.generation_defaults 并可由 TtsSynthesisOptions 覆盖。
    /// </summary>
    internal sealed class GenerationParams
    {
        public int MaxNewFrames { get; set; }
        public MossNanoSampleMode SampleMode { get; set; }
        public bool DoSample { get; set; }
        public float TextTemperature { get; set; }
        public float TextTopP { get; set; }
        public int TextTopK { get; set; }
        public float AudioTemperature { get; set; }
        public float AudioTopP { get; set; }
        public int AudioTopK { get; set; }
        public float AudioRepetitionPenalty { get; set; }

        public static GenerationParams FromManifest(ManifestGenerationDefaults d, MossNanoSampleMode normalizedSampleMode)
            => new()
            {
                MaxNewFrames = d.MaxNewFrames,
                DoSample = normalizedSampleMode != MossNanoSampleMode.Greedy,
                SampleMode = normalizedSampleMode,
                TextTemperature = d.TextTemperature,
                TextTopP = d.TextTopP,
                TextTopK = d.TextTopK,
                AudioTemperature = d.AudioTemperature,
                AudioTopP = d.AudioTopP,
                AudioTopK = d.AudioTopK,
                AudioRepetitionPenalty = d.AudioRepetitionPenalty,
            };

        public GenerationParams Clone() => (GenerationParams)MemberwiseClone();
    }

    internal static class SampleModeUtil
    {
        /// <summary>
        /// 与 Python `_normalize_sample_mode` 1:1 对齐。
        /// </summary>
        public static MossNanoSampleMode Normalize(string? raw, bool doSample)
        {
            string n = (raw ?? string.Empty).Trim().ToLowerInvariant();
            switch (n)
            {
                case "greedy": return MossNanoSampleMode.Greedy;
                case "fixed": return MossNanoSampleMode.Fixed;
                case "full": return MossNanoSampleMode.Full;
                case "mixed3": return doSample ? MossNanoSampleMode.Fixed : MossNanoSampleMode.Greedy;
                default: return doSample ? MossNanoSampleMode.Fixed : MossNanoSampleMode.Greedy;
            }
        }
    }
}
