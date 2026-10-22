using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Logging;
using VoiceBox.Core.Onnx;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 单后端评测器: 对同一批干净 utterance × 各 SNR 档, 混噪 → 跑 DNR → 时间对齐 → 算信号域指标,
    /// 并记录 RTF、逐帧延迟、算法延迟和预热时间。每条之间复位, 互不串扰。
    /// </summary>
    internal sealed class DnrRunner
    {
        private const int SampleRate = WavReader.ExpectedSampleRate; // 16000
        private const int RefineRadius = 320; // 逐对细化搜索半径 (20ms), 覆盖各后端固定延迟+HPF相移+内容抖动

        private readonly DnrSpec _spec;
        private readonly BenchmarkConfig _config;
        private readonly IReadOnlyList<string> _cleanFiles;
        private readonly RealNoiseSource? _realNoise;
        private readonly ILogger _log;

        public DnrRunner(DnrSpec spec, BenchmarkConfig config, IReadOnlyList<string> cleanFiles,
            RealNoiseSource? realNoise, ILoggerFactory loggerFactory)
        {
            _spec = spec;
            _config = config;
            _cleanFiles = cleanFiles;
            _realNoise = realNoise;
            _log = loggerFactory.CreateLogger($"DNR.{spec.Label}");
        }

        public BenchmarkSummary Run(CancellationToken ct)
        {
            var ep = ParseProvider(_config.ExecutionProvider);
            using DnrEngine engine = DnrEngineFactory.Create(_spec, SampleRate, ep);

            int maxLag = Math.Clamp(engine.FrameSize * 8, 1500, 8000);

            // 预热 + 估计该后端的粗粒度标称滞后 (逐对细化的搜索中心)。
            var warmupSw = Stopwatch.StartNew();
            int nominalLag = WarmupAndEstimateLag(engine, maxLag, ct);
            warmupSw.Stop();

            var summary = new BenchmarkSummary
            {
                Backend = _spec.Label,
                Timestamp = DateTime.Now.ToString("s", CultureInfo.InvariantCulture),
                CleanPath = _config.CleanPath,
                NoisePath = _config.NoisePath,
                NoiseType = _realNoise is null ? "white" : "real",
                ExecutionProvider = _config.ExecutionProvider,
                SnrLevels = _config.SnrLevels,
                WarmupSeconds = warmupSw.Elapsed.TotalSeconds,
            };

            var siSdrI = new List<double>();
            var segSnrI = new List<double>();
            var cleanFid = new List<double>();
            var alignLags = new List<int>(_cleanFiles.Count * _config.SnrLevels.Length);
            var latencies = new List<double>(_cleanFiles.Count * _config.SnrLevels.Length * 64);
            var perSnr = _config.SnrLevels.ToDictionary(s => s, _ => new SnrBucket());

            double totalAudio = 0, totalProc = 0;
            int processed = 0, failed = 0, dumped = 0;
            var wall = Stopwatch.StartNew();

            for (int fi = 0; fi < _cleanFiles.Count; fi++)
            {
                ct.ThrowIfCancellationRequested();
                string path = _cleanFiles[fi];

                float[] clean;
                try
                {
                    clean = WavReader.ReadMonoFloat32(path, out _, out _);
                }
                catch (Exception ex)
                {
                    failed++;
                    if (_config.Verbose) _log.LogWarning("跳过 {Path}: {Msg}", path, ex.Message);
                    continue;
                }
                if (clean.Length < engine.FrameSize) { failed++; continue; }
                processed++;
                double audioSec = clean.Length / (double)SampleRate;

                // 代价侧: 干净 → DNR → vs 干净 的 SI-SDR (无噪时是否损伤语音)。
                if (_config.MeasureCleanCost)
                {
                    engine.Reset();
                    float[] enhClean = engine.Run(clean, null);
                    int lagClean = AudioAlign.AlignBestLag(clean, enhClean, nominalLag, RefineRadius);
                    var (rc, ec) = AudioAlign.TrimToAligned(clean, enhClean, lagClean);
                    double fid = QualityMetrics.SiSdr(rc, ec);
                    summary.CleanCost.Add(new CleanCostSample { AudioPath = path, SiSdr = Clean(fid) });
                    if (!double.IsNaN(fid)) cleanFid.Add(fid);
                }

                bool dumpThis = _config.DumpDir is not null && dumped < _config.DumpCount;

                foreach (double snr in _config.SnrLevels)
                {
                    ct.ThrowIfCancellationRequested();
                    string key = $"{path}|snr={snr.ToString(CultureInfo.InvariantCulture)}";
                    float[] noisy = _realNoise is not null
                        ? _realNoise.AddNoise(clean, snr, key, _config.Seed)
                        : NoiseMixer.AddWhiteNoise(clean, snr, key, _config.Seed);

                    engine.Reset();
                    var perFrame = new List<double>(clean.Length / engine.FrameSize + 4);
                    var procSw = Stopwatch.StartNew();
                    float[] enhanced = engine.Run(noisy, perFrame);
                    procSw.Stop();

                    double procSec = procSw.Elapsed.TotalSeconds;
                    double inSiSdr = QualityMetrics.SiSdr(clean, noisy);
                    int lagNoisy = AudioAlign.AlignBestLag(clean, enhanced, nominalLag, RefineRadius);
                    alignLags.Add(lagNoisy);
                    var (rA, eA) = AudioAlign.TrimToAligned(clean, enhanced, lagNoisy);
                    double outSiSdr = QualityMetrics.SiSdr(rA, eA);
                    double di = outSiSdr - inSiSdr;

                    double segIn = QualityMetrics.SegmentalSnr(clean, noisy);
                    double segOut = QualityMetrics.SegmentalSnr(rA, eA);
                    double dseg = segOut - segIn;

                    totalAudio += audioSec;
                    totalProc += procSec;
                    latencies.AddRange(perFrame);

                    if (IsReal(inSiSdr) && IsReal(outSiSdr)) siSdrI.Add(di);
                    if (IsReal(segIn) && IsReal(segOut)) segSnrI.Add(dseg);

                    var bucket = perSnr[snr];
                    bucket.Count++;
                    if (IsReal(inSiSdr)) bucket.InSiSdr.Add(inSiSdr);
                    if (IsReal(outSiSdr)) bucket.OutSiSdr.Add(outSiSdr);
                    if (IsReal(inSiSdr) && IsReal(outSiSdr)) bucket.SiSdrI.Add(di);
                    if (IsReal(segIn) && IsReal(segOut)) bucket.SegSnrI.Add(dseg);

                    summary.Samples.Add(new PerSampleResult
                    {
                        Index = summary.Samples.Count + 1,
                        AudioPath = path,
                        SnrDb = snr,
                        InSiSdr = Clean(inSiSdr),
                        OutSiSdr = Clean(outSiSdr),
                        SiSdrI = (IsReal(inSiSdr) && IsReal(outSiSdr)) ? di : null,
                        SegSnrI = (IsReal(segIn) && IsReal(segOut)) ? dseg : null,
                        AudioSeconds = audioSec,
                        ProcSeconds = procSec,
                        Rtf = audioSec > 0 ? procSec / audioSec : 0,
                    });

                    if (dumpThis && Math.Abs(snr - _config.SnrLevels[0]) < 1e-9)
                        DumpAudition(path, snr, clean, noisy, enhanced);
                }

                if (dumpThis) dumped++;

                if (_config.Verbose || (processed % 25 == 0))
                    _log.LogInformation("[{Backend}] {Done}/{Total} 已处理",
                        _spec.Label, processed, _cleanFiles.Count);
            }
            wall.Stop();

            summary.SampleCount = processed;
            summary.FailedCount = failed;
            summary.PairCount = summary.Samples.Count;
            summary.MeanSiSdrI = Mean(siSdrI);
            summary.MeanSegSnrI = Mean(segSnrI);
            summary.MeanCleanFidelity = _config.MeasureCleanCost ? Mean(cleanFid) : null;
            summary.PerSnr = _config.SnrLevels.Select(s => perSnr[s].ToAggregate(s)).ToList();
            summary.TotalAudioSeconds = totalAudio;
            summary.TotalProcSeconds = totalProc;
            summary.AggregateRtf = totalAudio > 0 ? totalProc / totalAudio : 0;
            summary.ThroughputXRealtime = totalProc > 0 ? totalAudio / totalProc : 0;
            summary.WallSeconds = wall.Elapsed.TotalSeconds;
            summary.AlgorithmicLatencyMs = MedianLag(alignLags, nominalLag) * 1000.0 / SampleRate;
            latencies.Sort();
            summary.LatencyP50Ms = Percentile(latencies, 0.50);
            summary.LatencyP90Ms = Percentile(latencies, 0.90);
            summary.LatencyP99Ms = Percentile(latencies, 0.99);

            return summary;
        }

        /// <summary>用第一条可读的干净 clip 预热引擎并估计固定滞后。全失败则滞后取 0。</summary>
        private int WarmupAndEstimateLag(DnrEngine engine, int maxLag, CancellationToken ct)
        {
            foreach (string path in _cleanFiles)
            {
                ct.ThrowIfCancellationRequested();
                float[] clean;
                try { clean = WavReader.ReadMonoFloat32(path, out _, out _); }
                catch { continue; }
                if (clean.Length < engine.FrameSize * 4) continue;

                engine.Reset();
                float[] enhanced = engine.Run(clean, null);
                engine.Reset();
                return AudioAlign.EstimateNominalLag(clean, enhanced, maxLag);
            }
            return 0;
        }

        /// <summary>观测滞后的中位数 (整数); 无观测时回退标称值。作算法延迟上报。</summary>
        private static int MedianLag(List<int> lags, int fallback)
        {
            if (lags.Count == 0) return fallback;
            var sorted = lags.ToArray();
            Array.Sort(sorted);
            return sorted[sorted.Length / 2];
        }

        private void DumpAudition(string path, double snr, float[] clean, float[] noisy, float[] enhanced)
        {
            string stem = Path.GetFileNameWithoutExtension(path);
            string dir = _config.DumpDir!;
            string snrTag = snr.ToString("0", CultureInfo.InvariantCulture);
            string prefix = $"{_spec.Label}__{stem}";
            WavWriter.WriteMono16(Path.Combine(dir, $"{prefix}__clean.wav"), clean, SampleRate);
            WavWriter.WriteMono16(Path.Combine(dir, $"{prefix}__noisy_snr{snrTag}.wav"), noisy, SampleRate);
            WavWriter.WriteMono16(Path.Combine(dir, $"{prefix}__enh_snr{snrTag}.wav"), enhanced, SampleRate);
        }

        private static bool IsReal(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
        private static double? Clean(double v) => IsReal(v) ? v : null;

        private static double? Mean(List<double> xs)
        {
            if (xs.Count == 0) return null;
            double s = 0;
            foreach (var x in xs) s += x;
            return s / xs.Count;
        }

        private static double Percentile(List<double> sorted, double p)
        {
            if (sorted.Count == 0) return 0;
            int idx = (int)Math.Round(p * (sorted.Count - 1), MidpointRounding.AwayFromZero);
            idx = Math.Clamp(idx, 0, sorted.Count - 1);
            return sorted[idx];
        }

        private static OnnxExecutionProvider ParseProvider(string s) =>
            s.Trim().ToLowerInvariant() switch
            {
                "cpu" => OnnxExecutionProvider.Cpu,
                "cuda" => OnnxExecutionProvider.Cuda,
                "directml" or "dml" => OnnxExecutionProvider.DirectML,
                _ => throw new ArgumentException($"Unknown execution provider: {s} (cpu|cuda|directml)"),
            };

        private sealed class SnrBucket
        {
            public int Count;
            public readonly List<double> InSiSdr = new();
            public readonly List<double> OutSiSdr = new();
            public readonly List<double> SiSdrI = new();
            public readonly List<double> SegSnrI = new();

            public SnrAggregate ToAggregate(double snr) => new()
            {
                SnrDb = snr,
                Count = Count,
                MeanInSiSdr = Mean(InSiSdr),
                MeanOutSiSdr = Mean(OutSiSdr),
                MeanSiSdrI = Mean(SiSdrI),
                MeanSegSnrI = Mean(SegSnrI),
            };
        }
    }
}
