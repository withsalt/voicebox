using System.Diagnostics;
using System.Text.Json.Serialization;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 单条音频的识别结果与对比指标。所有时长字段单位均为秒。
    /// </summary>
    internal sealed class SampleResult
    {
        [JsonPropertyName("index")] public int Index { get; set; }
        [JsonPropertyName("audio_path")] public string AudioPath { get; set; } = "";
        [JsonPropertyName("reference")] public string Reference { get; set; } = "";
        [JsonPropertyName("hypothesis")] public string Hypothesis { get; set; } = "";

        [JsonPropertyName("audio_seconds")] public double AudioSeconds { get; set; }
        [JsonPropertyName("inference_seconds")] public double InferenceSeconds { get; set; }
        [JsonPropertyName("preproc_seconds")] public double PreprocSeconds { get; set; }
        [JsonPropertyName("rtf")] public double Rtf { get; set; }

        [JsonPropertyName("ref_length")] public int RefLength { get; set; }
        [JsonPropertyName("hyp_length")] public int HypLength { get; set; }
        [JsonPropertyName("substitutions")] public int Substitutions { get; set; }
        [JsonPropertyName("insertions")] public int Insertions { get; set; }
        [JsonPropertyName("deletions")] public int Deletions { get; set; }
        [JsonPropertyName("cer")] public double Cer { get; set; }
        [JsonPropertyName("exact_match")] public bool ExactMatch { get; set; }

        [JsonPropertyName("error")] public string? Error { get; set; }
    }

    internal sealed class BenchmarkSummary
    {
        [JsonPropertyName("timestamp")] public string Timestamp { get; set; } = "";
        [JsonPropertyName("dataset_csv")] public string DatasetCsv { get; set; } = "";
        [JsonPropertyName("model_dir")] public string ModelDir { get; set; } = "";
        [JsonPropertyName("execution_provider")] public string ExecutionProvider { get; set; } = "";

        /// <summary>前处理配置标签: none / aec / dnr / both。</summary>
        [JsonPropertyName("preproc")] public string Preproc { get; set; } = "none";

        /// <summary>DNR 后端 (dpdfnet / rnnoise)。仅当 preproc 含 dnr 时有意义。</summary>
        [JsonPropertyName("dnr_backend")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? DnrBackend { get; set; }

        /// <summary>叠加噪声的目标 SNR (dB)。null = 不叠噪 (干净集)。</summary>
        [JsonPropertyName("add_noise_snr_db")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public double? AddNoiseSnrDb { get; set; }

        /// <summary>真实噪声源 (wav 文件或目录)。null/缺省 = 白噪声 (当 add_noise_snr_db 非空时)。</summary>
        [JsonPropertyName("noise_file")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? NoiseFile { get; set; }
        [JsonPropertyName("chunk_size_sec")] public double ChunkSizeSec { get; set; }
        [JsonPropertyName("max_new_tokens")] public int MaxNewTokens { get; set; }
        [JsonPropertyName("sample_count")] public int SampleCount { get; set; }
        [JsonPropertyName("failed_count")] public int FailedCount { get; set; }
        [JsonPropertyName("seed")] public int Seed { get; set; }

        [JsonPropertyName("total_ref_chars")] public int TotalRefChars { get; set; }
        [JsonPropertyName("total_substitutions")] public int TotalSubstitutions { get; set; }
        [JsonPropertyName("total_insertions")] public int TotalInsertions { get; set; }
        [JsonPropertyName("total_deletions")] public int TotalDeletions { get; set; }
        [JsonPropertyName("aggregate_cer")] public double AggregateCer { get; set; }
        [JsonPropertyName("exact_match_rate")] public double ExactMatchRate { get; set; }

        [JsonPropertyName("total_audio_seconds")] public double TotalAudioSeconds { get; set; }
        [JsonPropertyName("total_inference_seconds")] public double TotalInferenceSeconds { get; set; }
        [JsonPropertyName("aggregate_rtf")] public double AggregateRtf { get; set; }

        /// <summary>所有样本前处理 (AEC/DNR) 累计耗时。独立于 ASR 推理, 不并入 RTF。</summary>
        [JsonPropertyName("total_preproc_seconds")] public double TotalPreprocSeconds { get; set; }
        [JsonPropertyName("warmup_seconds")] public double WarmupSeconds { get; set; }
        [JsonPropertyName("wall_seconds")] public double WallSeconds { get; set; }

        [JsonPropertyName("samples")] public List<SampleResult> Samples { get; set; } = new();

        /// <summary>
        /// 仅在使用 --compare &lt;baseline.json&gt; 时非空。包含与 baseline 的聚合差值及回归样本 top-N。
        /// </summary>
        [JsonPropertyName("comparison")]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ComparisonResult? Comparison { get; set; }
    }

    /// <summary>
    /// 当前一次跑与 baseline JSON 的对比结果。Δ 值统一为 current − baseline:
    /// CER/RTF 负数，exact_match_rate 正数。
    /// </summary>
    internal sealed class ComparisonResult
    {
        [JsonPropertyName("baseline_path")] public string BaselinePath { get; set; } = "";
        [JsonPropertyName("baseline_timestamp")] public string BaselineTimestamp { get; set; } = "";
        [JsonPropertyName("baseline_sample_count")] public int BaselineSampleCount { get; set; }
        [JsonPropertyName("matched_sample_count")] public int MatchedSampleCount { get; set; }

        [JsonPropertyName("baseline_aggregate_cer")] public double BaselineAggregateCer { get; set; }
        [JsonPropertyName("current_aggregate_cer")] public double CurrentAggregateCer { get; set; }
        [JsonPropertyName("delta_aggregate_cer")] public double DeltaAggregateCer { get; set; }

        [JsonPropertyName("baseline_aggregate_rtf")] public double BaselineAggregateRtf { get; set; }
        [JsonPropertyName("current_aggregate_rtf")] public double CurrentAggregateRtf { get; set; }
        [JsonPropertyName("delta_aggregate_rtf")] public double DeltaAggregateRtf { get; set; }

        [JsonPropertyName("baseline_exact_match_rate")] public double BaselineExactMatchRate { get; set; }
        [JsonPropertyName("current_exact_match_rate")] public double CurrentExactMatchRate { get; set; }
        [JsonPropertyName("delta_exact_match_rate")] public double DeltaExactMatchRate { get; set; }

        [JsonPropertyName("regressed_samples")] public List<RegressedSample> RegressedSamples { get; set; } = new();
    }

    /// <summary>
    /// 单条回归样本: baseline 命中 / 当前未命中, 或 CER 上升。按 (exact-flip 优先, 然后 delta_cer 降序) 排序后取 top-N。
    /// </summary>
    internal sealed class RegressedSample
    {
        [JsonPropertyName("audio_path")] public string AudioPath { get; set; } = "";
        [JsonPropertyName("reference")] public string Reference { get; set; } = "";
        [JsonPropertyName("baseline_hypothesis")] public string BaselineHypothesis { get; set; } = "";
        [JsonPropertyName("current_hypothesis")] public string CurrentHypothesis { get; set; } = "";

        [JsonPropertyName("baseline_cer")] public double BaselineCer { get; set; }
        [JsonPropertyName("current_cer")] public double CurrentCer { get; set; }
        [JsonPropertyName("delta_cer")] public double DeltaCer { get; set; }

        [JsonPropertyName("baseline_exact_match")] public bool BaselineExactMatch { get; set; }
        [JsonPropertyName("current_exact_match")] public bool CurrentExactMatch { get; set; }
    }

    internal sealed class BenchmarkConfig
    {
        public string CsvPath { get; init; } = "";
        public string ModelDir { get; init; } = "";
        public string ExecutionProvider { get; init; } = "Cpu";
        public int SampleCount { get; init; } = 500;
        public int Seed { get; init; } = 42;
        public double ChunkSizeSec { get; init; } = 30.0;
        public int MaxNewTokens { get; init; } = 256;
        public bool Verbose { get; init; }

        /// <summary>本次跑使用的前处理配置 (none/aec/dnr/both)。</summary>
        public PreprocMode Preproc { get; init; } = PreprocMode.None;

        /// <summary>AEC/DNR 的旋钮与执行后端。</summary>
        public PreprocParams PreprocParams { get; init; } = new();

        /// <summary>叠加噪声的目标 SNR (dB)。null = 不叠噪。在前处理之前叠加, 用于量化 DNR 收益侧。</summary>
        public double? AddNoiseSnrDb { get; init; }

        /// <summary>真实噪声源 (wav 文件或目录, 递归)。null = 白噪声 (当 AddNoiseSnrDb 非空时)。</summary>
        public string? NoiseFile { get; init; }
    }
}
