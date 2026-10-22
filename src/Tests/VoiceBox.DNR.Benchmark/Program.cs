using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using VoiceBox.APM.WebRtc.Models;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// DNR 降噪效果评测 CLI 入口。
    /// 用法示例:
    ///   VoiceBox.DNR.Benchmark.exe --count 30 --snr 10 --backends rnnoise
    ///   VoiceBox.DNR.Benchmark.exe --count 100 --snr 5,10,15 --noise E:\ML\Audio\noise\DEMAND
    ///   VoiceBox.DNR.Benchmark.exe --backends deepfilternet,dpdfnet-all,hush,rnnoise,webrtc --ep cuda
    /// </summary>
    internal static class Program
    {
        // 默认干净语料复用 ASR benchmark 的 AIShell 测试集常量; 机器不同请用 --clean 覆盖。
        private const string DefaultCleanPath =
            "E:\\Projects\\kws-train\\datasets\\speech_asr_aishell1_trainsets\\speech_asr_aishell_testsets.csv";

        static int Main(string[] args)
        {
            try { Console.OutputEncoding = Encoding.UTF8; } catch { /* 某些 CI 无 console buffer */ }

            ParsedArgs parsed;
            try { parsed = ParseArgs(args); }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"[ERROR] {ex.Message}");
                return 2;
            }

            if (parsed.ShowHelp) { PrintHelp(); return 0; }

            // 路径 / 参数提前校验, 避免跑到一半才挂。
            if (!File.Exists(parsed.CleanPath) && !Directory.Exists(parsed.CleanPath))
            {
                Console.Error.WriteLine($"[ERROR] Clean speech path not found (csv or dir): {parsed.CleanPath}");
                return 2;
            }
            if (parsed.NoisePath is not null && !File.Exists(parsed.NoisePath) && !Directory.Exists(parsed.NoisePath))
            {
                Console.Error.WriteLine($"[ERROR] Noise path not found: {parsed.NoisePath}");
                return 2;
            }

            List<DnrSpec> specs;
            try { specs = BackendParser.Parse(parsed.Backends, parsed.NsLevel); }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"[ERROR] {ex.Message}");
                return 2;
            }
            if (specs.Count == 0)
            {
                Console.Error.WriteLine("[ERROR] 没有可评测的后端 (--backends 为空)。");
                return 2;
            }

            IReadOnlyList<string> cleanFiles;
            try
            {
                cleanFiles = CleanSource.LoadAndSample(parsed.CleanPath, parsed.Full ? -1 : parsed.Count, parsed.Seed);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] 加载干净语料失败: {ex.Message}");
                return 2;
            }

            RealNoiseSource? realNoise = null;
            if (parsed.NoisePath is not null)
            {
                try { realNoise = new RealNoiseSource(parsed.NoisePath); }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] 加载噪声源失败: {ex.Message}");
                    return 2;
                }
            }

            var config = new BenchmarkConfig
            {
                CleanPath = parsed.CleanPath,
                NoisePath = parsed.NoisePath,
                SnrLevels = parsed.SnrLevels,
                SampleCount = parsed.Full ? -1 : parsed.Count,
                Seed = parsed.Seed,
                ExecutionProvider = parsed.ExecutionProvider,
                NsLevel = parsed.NsLevel.ToString().ToLowerInvariant(),
                MeasureCleanCost = parsed.MeasureCleanCost,
                DumpDir = parsed.DumpDir,
                Verbose = parsed.Verbose,
            };

            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Information);
                if (!parsed.Verbose)
                {
                    // DNR/APM 服务在 Information 打初始化/单次日志, 100+ 样本会刷屏; 非 verbose 压到 Warning。
                    b.AddFilter("VoiceBox.DNR", LogLevel.Warning);
                    b.AddFilter("VoiceBox.APM", LogLevel.Warning);
                }
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
                Console.WriteLine("[INFO] 收到 Ctrl-C, 正在收尾...");
            };

            PrintHeader(config, cleanFiles.Count, specs, realNoise);

            var rows = new List<MultiBackendTable.Row>(specs.Count);
            for (int i = 0; i < specs.Count; i++)
            {
                var spec = specs[i];
                Console.WriteLine();
                Console.WriteLine($"---------------- [{i + 1}/{specs.Count}] {spec.Label} ----------------");

                BenchmarkSummary summary;
                try
                {
                    summary = new DnrRunner(spec, config, cleanFiles, realNoise, loggerFactory).Run(cts.Token);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[INFO] 已取消。");
                    return 130;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] 后端 {spec.Label} 跑挂了: {ex.Message}");
                    rows.Add(new MultiBackendTable.Row(
                        new BenchmarkSummary { Backend = spec.Label, Error = ex.Message },
                        string.Empty, isError: true, error: ex.Message));
                    continue;
                }

                string outputPath = ResolveOutputPath(parsed.OutputPath, spec.Label);
                BenchmarkJson.Write(summary, outputPath);
                PrintSummary(summary, outputPath);
                rows.Add(new MultiBackendTable.Row(summary, outputPath, isError: false, error: null));
            }

            MultiBackendTable.Print(rows);
            return 0;
        }

        private static void PrintHeader(BenchmarkConfig c, int sampleCount, List<DnrSpec> specs, RealNoiseSource? noise)
        {
            Console.WriteLine();
            Console.WriteLine("================ DNR Benchmark ================");
            Console.WriteLine($"Clean source      : {c.CleanPath}");
            Console.WriteLine($"Noise source      : {(noise is null ? "white noise (synthetic)" : $"{c.NoisePath}  ({noise.FileCount} file(s))")}");
            Console.WriteLine($"SNR levels (dB)   : {string.Join(", ", c.SnrLevels.Select(s => s.ToString("0.#", CultureInfo.InvariantCulture)))}");
            Console.WriteLine($"Samples           : {sampleCount} (seed={c.Seed})");
            Console.WriteLine($"Execution provider: {c.ExecutionProvider}  (RNNoise/WebRTC 为 native, 忽略)");
            Console.WriteLine($"Clean-cost side   : {(c.MeasureCleanCost ? "on" : "off")}");
            if (c.DumpDir is not null)
                Console.WriteLine($"Dump audition     : {c.DumpDir}  (first {c.DumpCount} clips)");
            Console.WriteLine($"Backends ({specs.Count})     : {string.Join(", ", specs.Select(s => s.Label))}");
            Console.WriteLine("==============================================");
        }

        private static void PrintSummary(BenchmarkSummary s, string outputPath)
        {
            Console.WriteLine();
            Console.WriteLine($"==== {s.Backend} ====");
            Console.WriteLine($"Samples / pairs    : {s.SampleCount} utt (failed {s.FailedCount}) × {s.SnrLevels.Length} SNR = {s.PairCount}");
            Console.WriteLine($"Mean SI-SDRi       : {MultiBackendTable.Db(s.MeanSiSdrI)} dB   (越高越好)");
            Console.WriteLine($"Mean segSNRi       : {MultiBackendTable.Db(s.MeanSegSnrI)} dB   (越高越好)");
            if (s.MeanCleanFidelity is not null)
                Console.WriteLine($"Clean fidelity     : {MultiBackendTable.Db(s.MeanCleanFidelity)} dB SI-SDR (干净语音保真, 越高越好)");
            foreach (var a in s.PerSnr)
                Console.WriteLine($"  SNR {a.SnrDb,4:0.#} dB     : SI-SDRi {MultiBackendTable.Db(a.MeanSiSdrI)} dB, segSNRi {MultiBackendTable.Db(a.MeanSegSnrI)} dB  (in {MultiBackendTable.Db(a.MeanInSiSdr)}→out {MultiBackendTable.Db(a.MeanOutSiSdr)}, n={a.Count})");
            Console.WriteLine($"Aggregate RTF      : {s.AggregateRtf.ToString("F3", CultureInfo.InvariantCulture)}  ({s.ThroughputXRealtime.ToString("F1", CultureInfo.InvariantCulture)}x realtime)");
            Console.WriteLine($"Frame latency (ms) : p50 {s.LatencyP50Ms:F2} / p90 {s.LatencyP90Ms:F2} / p99 {s.LatencyP99Ms:F2}");
            Console.WriteLine($"Algorithmic delay  : {s.AlgorithmicLatencyMs:F1} ms  (固定流水线延迟)");
            Console.WriteLine($"Warmup / wall      : {s.WarmupSeconds:F2}s / {s.WallSeconds:F1}s");
            Console.WriteLine($"Output             : {outputPath}");
        }

        private static string ResolveOutputPath(string? userPath, string suffix)
        {
            if (!string.IsNullOrWhiteSpace(userPath))
            {
                string full = Path.GetFullPath(userPath);
                string dir = Path.GetDirectoryName(full) ?? string.Empty;
                string stem = Path.GetFileNameWithoutExtension(full);
                string ext = Path.GetExtension(full);
                return Path.Combine(dir, $"{stem}-{suffix}{ext}");
            }

            string ts = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            string fileName = $"dnr-benchmark-{suffix}-{ts}.json";

            // 从 dll 位置与 CWD 两处向上爬, 找仓库内的 TestResults 目录 (兼顾 --no-build 与从仓库根运行)。
            foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
            {
                string? cursor = start;
                for (int i = 0; i < 8 && cursor is not null; i++)
                {
                    string c1 = Path.Combine(cursor, "TestResults");
                    if (Directory.Exists(c1)) return Path.Combine(c1, fileName);
                    string c2 = Path.Combine(cursor, "src", "TestResults");
                    if (Directory.Exists(c2)) return Path.Combine(c2, fileName);
                    cursor = Path.GetDirectoryName(cursor);
                }
            }
            return Path.Combine(Environment.CurrentDirectory, fileName);
        }

        private sealed class ParsedArgs
        {
            public string CleanPath { get; set; } = DefaultCleanPath;
            public string? NoisePath { get; set; }
            public double[] SnrLevels { get; set; } = { 5, 10, 15 };
            public int Count { get; set; } = 200;
            public bool Full { get; set; }
            public int Seed { get; set; } = 42;
            public List<string> Backends { get; } = new();
            public string ExecutionProvider { get; set; } = "cpu";
            public NoiseSuppressionLevel NsLevel { get; set; } = NoiseSuppressionLevel.High;
            public bool MeasureCleanCost { get; set; } = true;
            public string? DumpDir { get; set; }
            public string? OutputPath { get; set; }
            public bool Verbose { get; set; }
            public bool ShowHelp { get; set; }
        }

        private static ParsedArgs ParseArgs(string[] args)
        {
            var p = new ParsedArgs();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "-h": case "--help": p.ShowHelp = true; break;
                    case "--clean": p.CleanPath = Req(args, ref i, a); break;
                    case "--noise": p.NoisePath = Req(args, ref i, a); break;
                    case "--snr": p.SnrLevels = ParseSnr(Req(args, ref i, a)); break;
                    case "--count": p.Count = int.Parse(Req(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--full": p.Full = true; break;
                    case "--seed": p.Seed = int.Parse(Req(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--backends":
                        foreach (var part in Req(args, ref i, a).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                            p.Backends.Add(part);
                        break;
                    case "--ep": case "--execution-provider": p.ExecutionProvider = Req(args, ref i, a); break;
                    case "--ns-level": p.NsLevel = ParseNs(Req(args, ref i, a)); break;
                    case "--no-clean-cost": p.MeasureCleanCost = false; break;
                    case "--dump-dir": p.DumpDir = Req(args, ref i, a); break;
                    case "--output": case "-o": p.OutputPath = Req(args, ref i, a); break;
                    case "--verbose": case "-v": p.Verbose = true; break;
                    default: throw new ArgumentException($"Unknown argument: {a}. Use --help for usage.");
                }
            }
            if (p.Backends.Count == 0) p.Backends.AddRange(BackendParser.DefaultTokens);
            return p;
        }

        private static string Req(string[] args, ref int i, string flag)
        {
            if (i + 1 >= args.Length) throw new ArgumentException($"Missing value for {flag}.");
            return args[++i];
        }

        private static double[] ParseSnr(string s)
        {
            var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length == 0) throw new ArgumentException($"--snr expects one or more dB values, got '{s}'.");
            var result = new double[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out result[i]))
                    throw new ArgumentException($"--snr value not a number: '{parts[i]}'.");
            return result;
        }

        private static NoiseSuppressionLevel ParseNs(string s) => s.Trim().ToLowerInvariant() switch
        {
            "low" => NoiseSuppressionLevel.Low,
            "moderate" or "mid" => NoiseSuppressionLevel.Moderate,
            "high" => NoiseSuppressionLevel.High,
            "veryhigh" or "very-high" or "vhigh" => NoiseSuppressionLevel.VeryHigh,
            _ => throw new ArgumentException($"Unknown --ns-level: {s} (low|moderate|high|veryhigh)"),
        };

        private static void PrintHelp()
        {
            Console.WriteLine(@"VoiceBox.DNR Benchmark — 降噪效果 / 性能横评

Usage:
  VoiceBox.DNR.Benchmark [options]

评测思路: 干净语音 + 噪声@SNR → 各 DNR → 时间对齐 → 对比干净参考算信号域指标。

Options:
  --clean <csv|dir>        干净语音源: AIShell 风格 CSV 或含 *.wav 的目录 (默认见源码常量)
  --noise <wav|dir>        真实噪声源 (DEMAND/DNS 等); 缺省时回退合成白噪声
  --snr <dB[,dB,...]>      测试信噪比档 (默认 5,10,15); 越小噪声越强
  --count <N>              抽样 utterance 数 (默认 200)
  --full                   跑全部 (覆盖 --count)
  --seed <N>               抽样 / 噪声种子 (默认 42, 保证各后端看到相同含噪信号)
  --backends <a,b,...>     选后端 (默认 deepfilternet,dpdfnet,hush,rnnoise,webrtc):
                             deepfilternet | hush | rnnoise | webrtc
                             dpdfnet (=dpdfnet8) | dpdfnet:baseline|2|4|8 | dpdfnet-all
  --ep <cpu|cuda|directml> ONNX 后端 EP (默认 cpu; RNNoise/WebRTC 为 native, 忽略)
  --ns-level <lvl>         WebRTC 噪声抑制强度: low|moderate|high|veryhigh (默认 high)
  --no-clean-cost          关闭'干净语音保真'测量 (默认开; 关掉可省一半推理)
  --dump-dir <dir>         导出前若干条 clean/noisy/enhanced wav 供试听
  --output, -o <path>      JSON 输出路径 (默认 <repo>/src/TestResults/dnr-benchmark-<backend>-<ts>.json;
                           后端短名会插到扩展名前)
  --verbose, -v            打印每条进度 / 服务日志
  --help, -h               显示帮助

Examples:
  # Smoke (快, native, 免 GPU)
  dotnet run --project src/Tests/VoiceBox.DNR.Benchmark -- --count 10 --snr 10 --backends rnnoise
  # 真实噪声全后端横评
  dotnet run --project src/Tests/VoiceBox.DNR.Benchmark -- --count 100 --snr 5,10,15 \
    --noise E:\ML\Audio\noise\DEMAND --backends deepfilternet,dpdfnet-all,hush,rnnoise,webrtc

指标: SI-SDRi / segSNRi 越高越好(降噪效果); CleanFid 越高越好(干净保真);
      RTF / p90 延迟 / 算法延迟 越低越好(性能)。退出码: 0 正常 / 2 参数或路径错 / 130 Ctrl-C。");
        }
    }
}
