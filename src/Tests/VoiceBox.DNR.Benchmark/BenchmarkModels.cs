using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>一次运行的输入配置 (各后端共享)。</summary>
    internal sealed class BenchmarkConfig
    {
        public required string CleanPath { get; init; }
        public string? NoisePath { get; init; }            // null => 白噪声
        public required double[] SnrLevels { get; init; }
        public int SampleCount { get; init; }              // -1 = 全部
        public int Seed { get; init; } = 42;
        public required string ExecutionProvider { get; init; }
        public required string NsLevel { get; init; }
        public bool MeasureCleanCost { get; init; } = true;
        public string? DumpDir { get; init; }
        public int DumpCount { get; init; } = 3;
        public bool Verbose { get; init; }
    }

    /// <summary>逐 (utterance, SNR) 的评分。NaN 指标 (静音参考等) 存 null。</summary>
    internal sealed class PerSampleResult
    {
        public int Index { get; set; }
        public string AudioPath { get; set; } = string.Empty;
        public double SnrDb { get; set; }
        public double? InSiSdr { get; set; }
        public double? OutSiSdr { get; set; }
        public double? SiSdrI { get; set; }
        public double? SegSnrI { get; set; }
        public double AudioSeconds { get; set; }
        public double ProcSeconds { get; set; }
        public double Rtf { get; set; }
        public string? Error { get; set; }
    }

    /// <summary>干净语音保真 (代价侧): 干净→DNR→vs 干净 的 SI-SDR, 每条一项。</summary>
    internal sealed class CleanCostSample
    {
        public string AudioPath { get; set; } = string.Empty;
        public double? SiSdr { get; set; }
    }

    /// <summary>单个 SNR 档的聚合。</summary>
    internal sealed class SnrAggregate
    {
        public double SnrDb { get; set; }
        public int Count { get; set; }
        public double? MeanInSiSdr { get; set; }
        public double? MeanOutSiSdr { get; set; }
        public double? MeanSiSdrI { get; set; }
        public double? MeanSegSnrI { get; set; }
    }

    /// <summary>单个后端跑完一批的完整结果 (落 JSON + 进横评表)。</summary>
    internal sealed class BenchmarkSummary
    {
        public string Backend { get; set; } = string.Empty;
        public string? Error { get; set; }
        public string Timestamp { get; set; } = string.Empty;
        public string CleanPath { get; set; } = string.Empty;
        public string? NoisePath { get; set; }
        public string NoiseType { get; set; } = "white";
        public string ExecutionProvider { get; set; } = string.Empty;
        public double[] SnrLevels { get; set; } = Array.Empty<double>();

        public int SampleCount { get; set; }     // 成功处理的 utterance 数
        public int FailedCount { get; set; }
        public int PairCount { get; set; }        // utterance × SNR 评分对数

        // 质量聚合 (对有效样本取均值, 越高越好)。
        public double? MeanSiSdrI { get; set; }
        public double? MeanSegSnrI { get; set; }
        public double? MeanCleanFidelity { get; set; }
        public List<SnrAggregate> PerSnr { get; set; } = new();

        public double AggregateRtf { get; set; }
        public double LatencyP50Ms { get; set; }
        public double LatencyP90Ms { get; set; }
        public double LatencyP99Ms { get; set; }
        public double AlgorithmicLatencyMs { get; set; }   // 实测固定流水线延迟 (对齐滞后 → ms)
        public double ThroughputXRealtime { get; set; }    // = 1 / RTF
        public double WarmupSeconds { get; set; }
        public double WallSeconds { get; set; }
        public double TotalAudioSeconds { get; set; }
        public double TotalProcSeconds { get; set; }

        public List<CleanCostSample> CleanCost { get; set; } = new();
        public List<PerSampleResult> Samples { get; set; } = new();
    }

    internal static class BenchmarkJson
    {
        private static readonly JsonSerializerOptions Options = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        // 不写 BOM: File.WriteAllText(..., Encoding.UTF8) 会加 UTF-8 BOM, 令 jq / json.load 解析失败。
        private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

        public static void Write(BenchmarkSummary summary, string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(summary, Options), Utf8NoBom);
        }
    }
}
