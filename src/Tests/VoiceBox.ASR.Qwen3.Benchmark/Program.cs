using System.Globalization;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using VoiceBox.Core.Onnx;
using VoiceBox.DNR.DPDFNet.Models;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 基准测试 CLI 入口。
    /// 用法示例:
    ///   VoiceBox.ASR.Qwen3.Benchmark.exe
    ///   VoiceBox.ASR.Qwen3.Benchmark.exe --count 100 --ep cuda
    ///   VoiceBox.ASR.Qwen3.Benchmark.exe --full --output result.json
    /// </summary>
    internal static class Program
    {
        // 默认值与 RecordASRDemo 保持一致, 便于直接复用同一份模型与 ORT。
        private const string DefaultModelDir = "E:\\ML\\Audio\\models\\v20260616\\asr\\qwen3\\qwen3-asr-1.7B-onnx-fp16";
        private const string DefaultDatasetCsv = "E:\\Projects\\kws-train\\datasets\\speech_asr_aishell1_trainsets\\speech_asr_aishell_testsets.csv";
        private const string DefaultExecutionProvider = "cuda";

        static async Task<int> Main(string[] args)
        {
            try
            {
                Console.OutputEncoding = Encoding.UTF8;
            }
            catch { /* 某些 CI 没 console buffer, 忽略 */ }

            ParsedArgs parsed;
            try
            {
                parsed = ParseArgs(args);
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"[ERROR] {ex.Message}");
                return 2;
            }

            if (parsed.ShowHelp)
            {
                PrintHelp();
                return 0;
            }

            // 互斥: baseline 是单模型概念, 与 batch 一起用语义不清。
            if (parsed.Models.Count > 0 && !string.IsNullOrWhiteSpace(parsed.ComparePath))
            {
                Console.Error.WriteLine("[ERROR] --models 与 --compare 不能同时使用 (baseline 仅对应单模型)。");
                return 2;
            }

            // 互斥: --matrix 自带四路对照并以 none 为内置 baseline, 与 --models / --compare / --preproc 冲突。
            if (parsed.Matrix && parsed.Models.Count > 0)
            {
                Console.Error.WriteLine("[ERROR] --matrix 与 --models 不能同时使用。");
                return 2;
            }
            if (parsed.Matrix && !string.IsNullOrWhiteSpace(parsed.ComparePath))
            {
                Console.Error.WriteLine("[ERROR] --matrix 自带 vs-none 对照, 不要再叠加 --compare。");
                return 2;
            }
            if (parsed.Matrix && parsed.Preproc is not null)
            {
                Console.Error.WriteLine("[ERROR] --matrix 会遍历所有前处理配置, 不要再指定 --preproc。");
                return 2;
            }

            // --noise-file 需配 --add-noise 提供 SNR; 路径提前校验, 避免跑到一半才挂。
            if (parsed.NoiseFile is not null)
            {
                if (parsed.AddNoiseSnrDb is null)
                {
                    Console.Error.WriteLine("[ERROR] --noise-file 需同时用 --add-noise <snr_db> 指定信噪比。");
                    return 2;
                }
                if (!File.Exists(parsed.NoiseFile) && !Directory.Exists(parsed.NoiseFile))
                {
                    Console.Error.WriteLine($"[ERROR] Noise path not found: {parsed.NoiseFile}");
                    return 2;
                }
            }

            // 提前校验所有模型目录存在, 避免跑到一半才挂。
            var modelDirs = parsed.Models.Count > 0
                ? parsed.Models
                : new List<string> { parsed.ModelDir };
            foreach (var dir in modelDirs)
            {
                if (!Directory.Exists(dir))
                {
                    Console.Error.WriteLine($"[ERROR] Model directory not found: {dir}");
                    return 2;
                }
            }

            //OnnxRuntimeNativeLoader.Configure(parsed.OnnxNativeDir, ParseProvider(parsed.ExecutionProvider));

            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Information);
                // Qwen3-ASR runtime logs "单次识别完成" at Information for every call —
                // useful for one-off demos, noisy for a 100+ sample benchmark. Suppress
                // unless --verbose so the runner's progress lines stay readable.
                if (!parsed.Verbose)
                    b.AddFilter("VoiceBox.ASR.Qwen3", LogLevel.Warning);
                b.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss ";
                });
            });

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
                Console.WriteLine("[INFO] 收到 Ctrl-C, 正在收尾...");
            };

            if (parsed.Matrix)
                return await RunMatrixAsync(parsed, loggerFactory, cts.Token).ConfigureAwait(false);
            else if (parsed.Models.Count > 0)
                return await RunBatchAsync(parsed, modelDirs, loggerFactory, cts.Token).ConfigureAwait(false);
            else
                return await RunSingleAsync(parsed, loggerFactory, cts.Token).ConfigureAwait(false);
        }

        private static async Task<int> RunSingleAsync(ParsedArgs parsed, ILoggerFactory loggerFactory, CancellationToken ct)
        {
            var config = BuildConfig(parsed, parsed.ModelDir, parsed.Preproc ?? PreprocMode.None);
            PrintHeader(config, parsed, modelLabel: null);

            var runner = new BenchmarkRunner(config, loggerFactory);
            BenchmarkSummary summary;
            try
            {
                summary = await runner.RunAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("[INFO] 已取消。");
                return 130;
            }

            if (!string.IsNullOrWhiteSpace(parsed.ComparePath))
            {
                try
                {
                    summary.Comparison = BenchmarkComparer.Compare(summary, parsed.ComparePath!);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[WARN] --compare 失败: {ex.Message}");
                }
            }

            string outputPath = ResolveOutputPath(parsed.OutputPath, suffix: null);
            WriteJson(summary, outputPath);
            PrintSummary(summary, outputPath);
            if (summary.Comparison is not null)
                BenchmarkComparer.PrintToConsole(summary.Comparison);
            return 0;
        }

        private static async Task<int> RunBatchAsync(ParsedArgs parsed, IReadOnlyList<string> modelDirs, ILoggerFactory loggerFactory, CancellationToken ct)
        {
            Console.WriteLine();
            Console.WriteLine($"================ Qwen3-ASR Benchmark (batch x {modelDirs.Count}) ================");
            for (int i = 0; i < modelDirs.Count; i++)
                Console.WriteLine($"  [{i + 1}] {modelDirs[i]}");
            Console.WriteLine("=================================================================");
            Console.WriteLine();

            var rows = new List<MultiModelTable.Row>(modelDirs.Count);
            for (int i = 0; i < modelDirs.Count; i++)
            {
                string dir = modelDirs[i];
                string modelName = new DirectoryInfo(dir).Name;
                Console.WriteLine();
                Console.WriteLine($"---------------- [{i + 1}/{modelDirs.Count}] {modelName} ----------------");

                var config = BuildConfig(parsed, dir, parsed.Preproc ?? PreprocMode.None);
                PrintHeader(config, parsed, modelLabel: modelName);

                var runner = new BenchmarkRunner(config, loggerFactory);
                BenchmarkSummary summary;
                try
                {
                    summary = await runner.RunAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[INFO] 已取消。");
                    return 130;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] 模型 {modelName} 跑挂了: {ex.Message}");
                    rows.Add(new MultiModelTable.Row(modelName, new BenchmarkSummary(), string.Empty, isError: true, error: ex.Message));
                    continue;
                }

                string outputPath = ResolveOutputPath(parsed.OutputPath, suffix: modelName);
                WriteJson(summary, outputPath);
                PrintSummary(summary, outputPath);
                rows.Add(new MultiModelTable.Row(modelName, summary, outputPath, isError: false, error: null));
            }

            MultiModelTable.Print(rows);
            return 0;
        }

        /// <summary>
        /// 前处理矩阵: 同一批样本顺序跑 none / aec / dnr / both 四种前处理, 复用 MultiModelTable
        /// 出横向对照, 并以 none 为 baseline 打印每路的 ΔCER / Δexact / 回归样本。
        /// 这是回答"AEC/DNR 是否在干净语音上掉点"最直接的形态。
        /// </summary>
        private static async Task<int> RunMatrixAsync(ParsedArgs parsed, ILoggerFactory loggerFactory, CancellationToken ct)
        {
            var modes = new[] { PreprocMode.None, PreprocMode.Aec, PreprocMode.Dnr, PreprocMode.Both };

            Console.WriteLine();
            Console.WriteLine("================ Qwen3-ASR Preprocessing Matrix ================");
            bool dpdfnet = parsed.DnrBackend == DnrBackend.DpdfNet;
            Console.WriteLine($"  Model : {parsed.ModelDir}");
            Console.WriteLine($"  Preproc EP : {parsed.PreprocEp}   (ASR EP: {parsed.ExecutionProvider})");
            Console.WriteLine($"  AEC MaskFloorDb : {(parsed.MaskFloorDb is { } mf ? mf.ToString(CultureInfo.InvariantCulture) : "off")}");
            Console.WriteLine($"  DNR backend : {AudioPreprocessor.BackendLabel(parsed.DnrBackend)}"
                + (dpdfnet
                    ? $"  (model={parsed.DnrModel}, AttenuationLimitDb={(parsed.AttenuationLimitDb is { } al ? al.ToString(CultureInfo.InvariantCulture) : "off")}, mode={parsed.AttenMode})"
                    : "  (48k native GRU, no tunables)"));
            if (parsed.AddNoiseSnrDb is double snr)
                Console.WriteLine($"  Add noise : {(parsed.NoiseFile is null ? "white" : "real")} @ {snr.ToString("0.#", CultureInfo.InvariantCulture)} dB SNR (before preproc) — measures DNR benefit"
                    + (parsed.NoiseFile is null ? "" : $"  [{parsed.NoiseFile}]"));
            Console.WriteLine($"  Configs : none, aec, dnr, both  (baseline = none)");
            Console.WriteLine("================================================================");

            var rows = new List<MultiModelTable.Row>(modes.Length);
            var summaries = new Dictionary<PreprocMode, BenchmarkSummary>();

            foreach (var mode in modes)
            {
                string label = AudioPreprocessor.ModeLabel(mode);
                Console.WriteLine();
                Console.WriteLine($"---------------- preproc = {label} ----------------");

                var config = BuildConfig(parsed, parsed.ModelDir, mode);
                var runner = new BenchmarkRunner(config, loggerFactory);
                BenchmarkSummary summary;
                try
                {
                    summary = await runner.RunAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[INFO] 已取消。");
                    return 130;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[ERROR] 前处理配置 {label} 跑挂了: {ex.Message}");
                    rows.Add(new MultiModelTable.Row(label, new BenchmarkSummary(), string.Empty, isError: true, error: ex.Message));
                    continue;
                }

                string outputPath = ResolveOutputPath(parsed.OutputPath, suffix: label);
                WriteJson(summary, outputPath);
                PrintSummary(summary, outputPath);
                summaries[mode] = summary;
                rows.Add(new MultiModelTable.Row(label, summary, outputPath, isError: false, error: null));
            }

            MultiModelTable.Print(rows);

            // 以 none 为 baseline, 对 aec / dnr / both 各打印一段对照 (ΔCER / Δexact / 回归 top-N)。
            if (summaries.TryGetValue(PreprocMode.None, out var baseline))
            {
                foreach (var mode in new[] { PreprocMode.Aec, PreprocMode.Dnr, PreprocMode.Both })
                {
                    if (!summaries.TryGetValue(mode, out var cur)) continue;
                    Console.WriteLine();
                    Console.WriteLine($"### {AudioPreprocessor.ModeLabel(mode)} vs none ###");
                    var cmp = BenchmarkComparer.Compare(cur, baseline, baselineLabel: "preproc=none (in-memory)");
                    BenchmarkComparer.PrintToConsole(cmp);
                }
            }
            else
            {
                Console.Error.WriteLine("[WARN] none 路未成功产出, 无法做 vs-none 对照。");
            }

            return 0;
        }

        private static BenchmarkConfig BuildConfig(ParsedArgs parsed, string modelDir, PreprocMode preproc) => new()
        {
            CsvPath = parsed.CsvPath,
            ModelDir = modelDir,
            ExecutionProvider = parsed.ExecutionProvider,
            SampleCount = parsed.Full ? -1 : parsed.Count,
            Seed = parsed.Seed,
            ChunkSizeSec = parsed.ChunkSizeSec,
            MaxNewTokens = parsed.MaxNewTokens,
            Verbose = parsed.Verbose,
            Preproc = preproc,
            AddNoiseSnrDb = parsed.AddNoiseSnrDb,
            NoiseFile = parsed.NoiseFile,
            PreprocParams = new PreprocParams
            {
                ExecutionProvider = ParseProvider(parsed.PreprocEp),
                MaskFloorDb = parsed.MaskFloorDb,
                DnrBackend = parsed.DnrBackend,
                DnrModel = parsed.DnrModel,
                AttenuationLimitDb = parsed.AttenuationLimitDb,
                AttenuationMode = parsed.AttenMode,
            },
        };

        private static void PrintHeader(BenchmarkConfig config, ParsedArgs parsed, string? modelLabel)
        {
            Console.WriteLine();
            Console.WriteLine("================ Qwen3-ASR Benchmark ================");
            if (modelLabel is not null)
                Console.WriteLine($"Model label       : {modelLabel}");
            Console.WriteLine($"Dataset CSV       : {config.CsvPath}");
            Console.WriteLine($"Model dir         : {config.ModelDir}");
            Console.WriteLine($"Execution provider: {config.ExecutionProvider}");
            Console.WriteLine($"Preproc           : {AudioPreprocessor.ModeLabel(config.Preproc)}"
                + (config.Preproc == PreprocMode.None ? "" : $" (EP={config.PreprocParams.ExecutionProvider})"));
            if (config.AddNoiseSnrDb is double snr)
                Console.WriteLine($"Add noise         : {(config.NoiseFile is null ? "white" : "real")} @ {snr.ToString("0.#", CultureInfo.InvariantCulture)} dB SNR"
                    + (config.NoiseFile is null ? "" : $"  [{config.NoiseFile}]"));
            Console.WriteLine($"Samples           : {(parsed.Full ? "ALL" : config.SampleCount.ToString())} (seed={config.Seed})");
            Console.WriteLine($"Chunk size (sec)  : {config.ChunkSizeSec}");
            Console.WriteLine($"Max new tokens    : {config.MaxNewTokens}");
            Console.WriteLine("=====================================================");
            Console.WriteLine();
        }

        /// <summary>
        /// 解析输出路径。batch 模式传 suffix=模型短名, 写出形如 qwen3-asr-benchmark-{model}-{ts}.json;
        /// 当用户显式给了 --output 时, suffix 会拼到文件名 stem 之后, 保持目录与扩展名不变。
        /// </summary>
        private static string ResolveOutputPath(string? userPath, string? suffix)
        {
            if (!string.IsNullOrWhiteSpace(userPath))
            {
                string full = Path.GetFullPath(userPath);
                if (string.IsNullOrEmpty(suffix)) return full;
                string dir = Path.GetDirectoryName(full) ?? string.Empty;
                string stem = Path.GetFileNameWithoutExtension(full);
                string ext = Path.GetExtension(full);
                return Path.Combine(dir, $"{stem}-{suffix}{ext}");
            }

            string ts = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            string fileName = string.IsNullOrEmpty(suffix)
                ? $"qwen3-asr-benchmark-{ts}.json"
                : $"qwen3-asr-benchmark-{suffix}-{ts}.json";

            // 从两个起点向上爬, 找仓库内的 TestResults 目录:
            //   1) bin\Debug\net10.0\ —— 从 dll 实际位置反推, 与 CWD 无关 (--no-build 时也准)
            //   2) Environment.CurrentDirectory —— 用户从仓库根或 src/ 跑时, 这条更直接
            // 每级既看 <cursor>/TestResults, 也看 <cursor>/src/TestResults, 覆盖两种布局。
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
            // 兜底: 输出到当前工作目录。
            return Path.Combine(Environment.CurrentDirectory, fileName);
        }

        private static void WriteJson(BenchmarkSummary summary, string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            };
            File.WriteAllText(path, JsonSerializer.Serialize(summary, options), Encoding.UTF8);
        }

        private static void PrintSummary(BenchmarkSummary s, string outputPath)
        {
            Console.WriteLine();
            Console.WriteLine("==================== Result Summary ====================");
            Console.WriteLine($"Samples            : {s.SampleCount} (failed: {s.FailedCount})");
            Console.WriteLine($"Aggregate CER      : {CerCalculator.FormatPercent(s.AggregateCer)}  (S={s.TotalSubstitutions}, I={s.TotalInsertions}, D={s.TotalDeletions}, ref chars={s.TotalRefChars})");
            Console.WriteLine($"Exact-match rate   : {CerCalculator.FormatPercent(s.ExactMatchRate)}");
            Console.WriteLine($"Total audio        : {s.TotalAudioSeconds:F1}s");
            Console.WriteLine($"Total inference    : {s.TotalInferenceSeconds:F1}s");
            if (!string.Equals(s.Preproc, "none", StringComparison.OrdinalIgnoreCase))
                Console.WriteLine($"Total preproc      : {s.TotalPreprocSeconds:F1}s  (preproc={s.Preproc}, not in RTF)");
            Console.WriteLine($"Aggregate RTF      : {s.AggregateRtf.ToString("F3", CultureInfo.InvariantCulture)}");
            Console.WriteLine($"Warmup time        : {s.WarmupSeconds:F1}s");
            Console.WriteLine($"Wall time          : {s.WallSeconds:F1}s");
            Console.WriteLine($"Output             : {outputPath}");
            Console.WriteLine("========================================================");
        }

        private static OnnxExecutionProvider ParseProvider(string s) =>
            s.Trim().ToLowerInvariant() switch
            {
                "cpu" => OnnxExecutionProvider.Cpu,
                "cuda" => OnnxExecutionProvider.Cuda,
                "directml" or "dml" => OnnxExecutionProvider.DirectML,
                _ => throw new ArgumentException($"Unknown execution provider: {s} (cpu|cuda|directml)"),
            };

        private sealed class ParsedArgs
        {
            public string CsvPath { get; set; } = DefaultDatasetCsv;
            public string ModelDir { get; set; } = DefaultModelDir;
            public List<string> Models { get; } = new();
            public string ExecutionProvider { get; set; } = DefaultExecutionProvider;
            public int Count { get; set; } = 500;
            public bool Full { get; set; }
            public int Seed { get; set; } = 42;
            public double ChunkSizeSec { get; set; } = 30.0;
            public int MaxNewTokens { get; set; } = 256;
            public string? OutputPath { get; set; }
            public string? ComparePath { get; set; }
            public bool Verbose { get; set; }
            public bool ShowHelp { get; set; }

            // 前处理矩阵 / 单配置。
            public bool Matrix { get; set; }
            public PreprocMode? Preproc { get; set; }      // null = none (单跑时)
            public string PreprocEp { get; set; } = "cpu"; // AEC/DNR 的 EP, 默认 CPU 避免与 ASR 抢 GPU
            public DnrBackend DnrBackend { get; set; } = DnrBackend.DpdfNet; // DNR 后端: dpdfnet(默认) / rnnoise
            public float? MaskFloorDb { get; set; } = 60f;             // 对齐 Demo; "off" => null
            public DPDFNetModel DnrModel { get; set; } = DPDFNetModel.DpDfNet8;
            public float? AttenuationLimitDb { get; set; } = 6f;       // 对齐 Demo; "off" => null
            public DPDFNetAttenuationMode AttenMode { get; set; } = DPDFNetAttenuationMode.ComplexBlend;
            public double? AddNoiseSnrDb { get; set; }                 // null = 不叠噪
            public string? NoiseFile { get; set; }                     // 真实噪声源 (wav/dir); null = 白噪声
        }

        private static ParsedArgs ParseArgs(string[] args)
        {
            var p = new ParsedArgs();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "-h":
                    case "--help":
                        p.ShowHelp = true; break;
                    case "--csv":
                        p.CsvPath = RequireValue(args, ref i, a); break;
                    case "--model-dir":
                        p.ModelDir = RequireValue(args, ref i, a); break;
                    case "--models":
                        foreach (var part in RequireValue(args, ref i, a).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                            p.Models.Add(part);
                        break;
                    case "--compare":
                        p.ComparePath = RequireValue(args, ref i, a); break;
                    case "--ep":
                    case "--execution-provider":
                        p.ExecutionProvider = RequireValue(args, ref i, a); break;
                    case "--count":
                        p.Count = int.Parse(RequireValue(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--full":
                        p.Full = true; break;
                    case "--seed":
                        p.Seed = int.Parse(RequireValue(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--chunk-size":
                        p.ChunkSizeSec = double.Parse(RequireValue(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--max-new-tokens":
                        p.MaxNewTokens = int.Parse(RequireValue(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--output":
                    case "-o":
                        p.OutputPath = RequireValue(args, ref i, a); break;
                    case "--verbose":
                    case "-v":
                        p.Verbose = true; break;
                    case "--matrix":
                        p.Matrix = true; break;
                    case "--preproc":
                        p.Preproc = AudioPreprocessor.ParseMode(RequireValue(args, ref i, a)); break;
                    case "--preproc-ep":
                        p.PreprocEp = RequireValue(args, ref i, a); break;
                    case "--dnr-backend":
                        p.DnrBackend = AudioPreprocessor.ParseBackend(RequireValue(args, ref i, a)); break;
                    case "--mask-floor-db":
                        p.MaskFloorDb = ParseNullableFloat(RequireValue(args, ref i, a), a); break;
                    case "--dnr-model":
                        p.DnrModel = ParseDnrModel(RequireValue(args, ref i, a)); break;
                    case "--atten-limit-db":
                        p.AttenuationLimitDb = ParseNullableFloat(RequireValue(args, ref i, a), a); break;
                    case "--atten-mode":
                        p.AttenMode = ParseAttenMode(RequireValue(args, ref i, a)); break;
                    case "--add-noise":
                        p.AddNoiseSnrDb = double.Parse(RequireValue(args, ref i, a), CultureInfo.InvariantCulture); break;
                    case "--noise-file":
                        p.NoiseFile = RequireValue(args, ref i, a); break;
                    default:
                        throw new ArgumentException($"Unknown argument: {a}. Use --help for usage.");
                }
            }
            return p;
        }

        private static string RequireValue(string[] args, ref int i, string flag)
        {
            if (i + 1 >= args.Length)
                throw new ArgumentException($"Missing value for {flag}.");
            return args[++i];
        }

        /// <summary>解析 "off"/"none"/"null" → null (不设限), 否则按非负 float 解析。</summary>
        private static float? ParseNullableFloat(string s, string flag)
        {
            string t = s.Trim();
            if (t.Equals("off", StringComparison.OrdinalIgnoreCase) ||
                t.Equals("none", StringComparison.OrdinalIgnoreCase) ||
                t.Equals("null", StringComparison.OrdinalIgnoreCase))
                return null;
            if (!float.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) || v < 0f)
                throw new ArgumentException($"{flag} expects a non-negative number or 'off', got '{s}'.");
            return v;
        }

        private static DPDFNetModel ParseDnrModel(string s) =>
            s.Trim().ToLowerInvariant() switch
            {
                "baseline" => DPDFNetModel.Baseline,
                "dpdfnet2" => DPDFNetModel.DpDfNet2,
                "dpdfnet4" => DPDFNetModel.DpDfNet4,
                "dpdfnet8" => DPDFNetModel.DpDfNet8,
                _ => throw new ArgumentException($"Unknown DNR model: {s} (baseline|dpdfnet2|dpdfnet4|dpdfnet8)"),
            };

        private static DPDFNetAttenuationMode ParseAttenMode(string s) =>
            s.Trim().ToLowerInvariant() switch
            {
                "complex" or "complexblend" or "blend" => DPDFNetAttenuationMode.ComplexBlend,
                "magfloor" or "magnitudefloor" or "magnitude" => DPDFNetAttenuationMode.MagnitudeFloor,
                _ => throw new ArgumentException($"Unknown atten mode: {s} (complex|magfloor)"),
            };

        private static void PrintHelp()
        {
            Console.WriteLine(@"VoiceBox.ASR.Qwen3 Benchmark

Usage:
  VoiceBox.ASR.Qwen3.Benchmark [options]

Options:
  --csv <path>             AIShell-1 test CSV path (default: " + DefaultDatasetCsv + @")
  --model-dir <path>       Qwen3-ASR ONNX model dir (default: " + DefaultModelDir + @")
  --models <a,b,c>         Batch mode: run multiple model dirs sequentially.
                           Use commas, or pass --models repeatedly. Each model produces
                           its own JSON; a comparison table is printed at the end.
                           Mutually exclusive with --compare.
  --compare <baseline.json> After the run, diff against a baseline JSON. Aggregate Δ
                           (CER/RTF/exact-match) and the top regressed samples are
                           printed and embedded in the output JSON as `comparison`.
                           Mutually exclusive with --models.
  --ep <cpu|cuda|directml> ONNX execution provider (default: " + DefaultExecutionProvider + @")
  --count <N>              Number of samples to benchmark (default: 500)
  --full                   Run all rows in the CSV (overrides --count)
  --seed <N>               Sampling seed for reproducibility (default: 42)
  --chunk-size <sec>       Qwen3 streaming chunk size (default: 30.0)
                           Set large so AIShell short clips finish in one pass.
  --max-new-tokens <N>     Decoder max new tokens (default: 256)
  --output, -o <path>      JSON output path (default: <repo>/TestResults/qwen3-asr-benchmark-<ts>.json)
                           In batch mode the model short name is inserted before the extension.

Preprocessing (AEC/DNR) — investigate ASR accuracy under DFSMN-AEC / DPDFNet-DNR:
  --matrix                 Run none/aec/dnr/both over the SAME samples, print a comparison
                           table + each config's ΔCER vs none. Best way to see if AEC/DNR
                           degrade clean speech. Mutually exclusive with --models/--compare/--preproc.
  --preproc <mode>         Single run with one preproc config: none|aec|dnr|both (default: none).
  --preproc-ep <ep>        EP for AEC/DNR ONNX: cpu|cuda|directml (default: cpu, avoids GPU
                           contention with the ASR session). Ignored by the rnnoise backend.
  --dnr-backend <name>     DNR implementation: dpdfnet (default, DPDFNet ONNX) | deepfilternet
                           (DFN3-LL ONNX, 48k native) | hush (DFN3-SE ONNX, 16k native) | rnnoise
                           (classic 48k native GRU) | webrtc (WebRTC APM noise suppressor).
                           Only dpdfnet honours --dnr-model/--atten-limit-db/--atten-mode; others ignore them.
  --mask-floor-db <v|off>  DFSMN-AEC neural-mask attenuation floor in dB (default: 60, matches
                           VoiceAssistantDemo). 'off' = unclamped model mask. Lower it (e.g. 18)
                           to test whether a gentler mask recovers clean-speech accuracy.
  --dnr-model <name>       DPDFNet variant: baseline|dpdfnet2|dpdfnet4|dpdfnet8 (default: dpdfnet8).
  --atten-limit-db <v|off> DPDFNet max per-frame suppression cap in dB (default: 6, matches Demo).
                           'off' = fully use the enhanced spectrum.
  --atten-mode <name>      How AttenuationLimitDb is applied: complex (default, historical
                           complex-spectrum blend) | magfloor (magnitude-domain gain floor that
                           preserves the noisy phase — avoids the complex-blend phase artifact,
                           much friendlier to ASR). dpdfnet backend only.
  --add-noise <snr_db>     Mix deterministic noise into each clip at this SNR (dB) BEFORE
                           preprocessing. Lets the matrix measure the DNR *benefit* on noisy
                           input (dnr vs none should now improve). Lower SNR = harsher noise.
                           Per-clip seed is stable (path hash), so all configs see identical noise.
                           White noise by default; pass --noise-file for real recorded noise.
  --noise-file <wav|dir>   Use REAL recorded noise instead of white noise (requires --add-noise).
                           A single .wav or a directory (recursed for *.wav). Per clip a noise file
                           + offset are picked deterministically (stable path hash), so none/dnr
                           see identical noisy audio. 16k/mono/16-bit expected; each file cached
                           capped to 30s. Real noise is in-distribution for SE models (DEMAND/DNS),
                           unlike white noise — the proper test of whether DNR helps ASR.
  --verbose, -v            Log every sample's hypothesis/reference
  --help, -h               Show this help
");
        }
    }
}
