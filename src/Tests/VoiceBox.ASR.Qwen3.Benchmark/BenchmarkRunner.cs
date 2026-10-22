using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using VoiceBox.ASR.Qwen3;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.ASRService;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 协调器: 加载数据集 → 启动 ASR 服务 → 逐条跑 → 汇总 CER / RTF。
    /// 单线程顺序执行: Qwen3 ASR 内部 decode 较重, 串行更能反映真实 RTF; 也避免 GPU 排队抖动干扰精度。
    /// </summary>
    internal sealed class BenchmarkRunner
    {
        private readonly BenchmarkConfig _config;
        private readonly ILoggerFactory _loggerFactory;
        private readonly ILogger _logger;

        public BenchmarkRunner(BenchmarkConfig config, ILoggerFactory loggerFactory)
        {
            _config = config;
            _loggerFactory = loggerFactory;
            _logger = loggerFactory.CreateLogger("Benchmark");
        }

        public async Task<BenchmarkSummary> RunAsync(CancellationToken ct)
        {
            var dataset = DatasetLoader.LoadAndSample(_config.CsvPath, _config.SampleCount, _config.Seed);
            _logger.LogInformation("数据集加载完毕: 共 {Count} 条 (CSV: {Csv})", dataset.Count, _config.CsvPath);

            var services = new ServiceCollection();
            services.AddSingleton(_loggerFactory);
            services.AddLogging();
            services.AddQwen3ASRService(o =>
            {
                o.ModelPath = _config.ModelDir;
                o.ExecutionProvider = ParseProvider(_config.ExecutionProvider);
                o.ChunkSizeSec = _config.ChunkSizeSec;
                o.MaxNewTokens = _config.MaxNewTokens;
                o.MaxConcurrentInferences = 1;
                o.MaxConcurrentSessions = 1;
            });

            var warmupSw = Stopwatch.StartNew();
            await using var provider = services.BuildServiceProvider();
            var asr = provider.GetRequiredService<IStreamASRService>();
            warmupSw.Stop();
            _logger.LogInformation("ASR 服务就绪 (含 warmup): {Ms:F0}ms", warmupSw.Elapsed.TotalMilliseconds);

            var summary = new BenchmarkSummary
            {
                Timestamp = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture),
                DatasetCsv = _config.CsvPath,
                ModelDir = _config.ModelDir,
                ExecutionProvider = _config.ExecutionProvider,
                Preproc = AudioPreprocessor.ModeLabel(_config.Preproc),
                DnrBackend = _config.Preproc is PreprocMode.Dnr or PreprocMode.Both
                    ? AudioPreprocessor.BackendLabel(_config.PreprocParams.DnrBackend)
                    : null,
                AddNoiseSnrDb = _config.AddNoiseSnrDb,
                NoiseFile = _config.NoiseFile,
                ChunkSizeSec = _config.ChunkSizeSec,
                MaxNewTokens = _config.MaxNewTokens,
                SampleCount = dataset.Count,
                Seed = _config.Seed,
                WarmupSeconds = warmupSw.Elapsed.TotalSeconds,
            };

            // 前处理器整段复用 (ONNX 模型只加载一次), 逐条 utterance 内部复位。None 模式为轻量直通。
            using var preprocessor = new AudioPreprocessor(_config.Preproc, _config.PreprocParams);

            // 真实噪声源 (若指定 --noise-file): 整批复用一个实例, 内部按文件路径缓存噪声数组。
            RealNoiseSource? realNoise = !string.IsNullOrWhiteSpace(_config.NoiseFile)
                ? new RealNoiseSource(_config.NoiseFile!)
                : null;
            if (realNoise is not null)
                _logger.LogInformation("真实噪声源就绪: {Count} 个 wav (root: {Root})", realNoise.FileCount, realNoise.Root);

            var wallSw = Stopwatch.StartNew();
            int index = 0;
            foreach (var entry in dataset)
            {
                ct.ThrowIfCancellationRequested();
                index++;
                var sample = await ProcessOneAsync(asr, preprocessor, realNoise, entry, index, dataset.Count, ct).ConfigureAwait(false);
                summary.Samples.Add(sample);

                if (sample.Error is null)
                {
                    summary.TotalRefChars += sample.RefLength;
                    summary.TotalSubstitutions += sample.Substitutions;
                    summary.TotalInsertions += sample.Insertions;
                    summary.TotalDeletions += sample.Deletions;
                    summary.TotalAudioSeconds += sample.AudioSeconds;
                    summary.TotalInferenceSeconds += sample.InferenceSeconds;
                    summary.TotalPreprocSeconds += sample.PreprocSeconds;
                }
                else
                {
                    summary.FailedCount++;
                }
            }
            wallSw.Stop();

            summary.WallSeconds = wallSw.Elapsed.TotalSeconds;
            summary.AggregateCer = summary.TotalRefChars > 0
                ? (double)(summary.TotalSubstitutions + summary.TotalInsertions + summary.TotalDeletions) / summary.TotalRefChars
                : 0;
            int successCount = summary.SampleCount - summary.FailedCount;
            int exactCount = summary.Samples.Count(s => s.Error is null && s.ExactMatch);
            summary.ExactMatchRate = successCount > 0 ? (double)exactCount / successCount : 0;
            summary.AggregateRtf = summary.TotalAudioSeconds > 0
                ? summary.TotalInferenceSeconds / summary.TotalAudioSeconds
                : 0;

            return summary;
        }

        private async Task<SampleResult> ProcessOneAsync(
            IStreamASRService asr,
            AudioPreprocessor preprocessor,
            RealNoiseSource? realNoise,
            DatasetEntry entry,
            int index,
            int total,
            CancellationToken ct)
        {
            var sample = new SampleResult
            {
                Index = index,
                AudioPath = entry.AudioPath,
                Reference = entry.Label,
            };

            float[] audio;
            try
            {
                audio = WavReader.ReadMonoFloat32(entry.AudioPath, out _, out _);
                sample.AudioSeconds = audio.Length / (double)WavReader.ExpectedSampleRate;
            }
            catch (Exception ex)
            {
                sample.Error = "wav_read: " + ex.Message;
                _logger.LogWarning("[{Index}/{Total}] 读 WAV 失败: {Path} - {Err}", index, total, entry.AudioPath, ex.Message);
                return sample;
            }

            // 叠加噪声 (若启用): 在前处理之前, 模拟含噪采集。按路径稳定播种, 各配置看到同一段含噪信号。
            // 有 --noise-file 走真实噪声 (训练分布内), 否则白噪声 (分布外)。
            if (_config.AddNoiseSnrDb is double snr)
            {
                audio = realNoise is not null
                    ? realNoise.AddNoise(audio, snr, entry.AudioPath, _config.Seed)
                    : NoiseMixer.AddWhiteNoise(audio, snr, entry.AudioPath, _config.Seed);
            }

            // 前处理 (AEC/DNR) 在喂入 ASR 之前完成, 单独计时, 不计入 ASR 推理与 RTF。
            // RTF 仍以原始音频时长为分母, 反映 ASR 本身的实时性。
            try
            {
                var preSw = Stopwatch.StartNew();
                audio = preprocessor.Process(audio);
                preSw.Stop();
                sample.PreprocSeconds = preSw.Elapsed.TotalSeconds;
            }
            catch (Exception ex)
            {
                sample.Error = "preproc: " + ex.Message;
                _logger.LogWarning(ex, "[{Index}/{Total}] 前处理失败: {Path}", index, total, entry.AudioPath);
                return sample;
            }

            string? sessionId = null;
            var sw = Stopwatch.StartNew();
            try
            {
                var info = await asr.StartSessionAsync(ct).ConfigureAwait(false);
                sessionId = info.SessionId;

                // 分块喂入以贴近真实流式调用; 块大小取 1s, 不影响最终精度。
                const int feedChunk = WavReader.ExpectedSampleRate; // 1s
                for (int offset = 0; offset < audio.Length; offset += feedChunk)
                {
                    int len = Math.Min(feedChunk, audio.Length - offset);
                    asr.FeedAudio(sessionId, audio.AsSpan(offset, len));
                }

                var result = await asr.FinishSessionAsync(sessionId, ct).ConfigureAwait(false);
                sw.Stop();
                sample.InferenceSeconds = sw.Elapsed.TotalSeconds;
                sample.Rtf = sample.AudioSeconds > 0 ? sample.InferenceSeconds / sample.AudioSeconds : 0;
                sample.Hypothesis = result?.Text ?? string.Empty;

                var score = CerCalculator.Compute(sample.Reference, sample.Hypothesis);
                sample.RefLength = score.RefLength;
                sample.HypLength = score.HypLength;
                sample.Substitutions = score.Substitutions;
                sample.Insertions = score.Insertions;
                sample.Deletions = score.Deletions;
                sample.Cer = score.Cer;
                sample.ExactMatch = score.ExactMatch;

                if (_config.Verbose || index % 20 == 0 || index == total)
                {
                    _logger.LogInformation(
                        "[{Index}/{Total}] audio={Audio:F2}s inf={Inf:F2}s rtf={Rtf:F2} cer={Cer} ref=\"{Ref}\" hyp=\"{Hyp}\"",
                        index, total,
                        sample.AudioSeconds, sample.InferenceSeconds, sample.Rtf,
                        CerCalculator.FormatPercent(sample.Cer),
                        Truncate(sample.Reference, 30), Truncate(sample.Hypothesis, 30));
                }
            }
            catch (Exception ex)
            {
                sw.Stop();
                sample.Error = "asr: " + ex.Message;
                _logger.LogWarning(ex, "[{Index}/{Total}] ASR 失败: {Path}", index, total, entry.AudioPath);
                if (sessionId is not null)
                {
                    try { await asr.FinishSessionAsync(sessionId, CancellationToken.None).ConfigureAwait(false); }
                    catch { /* ignore */ }
                }
            }
            return sample;
        }

        private static string Truncate(string s, int maxRunes)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            int count = 0;
            foreach (var _ in s.EnumerateRunes()) count++;
            if (count <= maxRunes) return s;
            var sb = new System.Text.StringBuilder();
            int i = 0;
            foreach (var rune in s.EnumerateRunes())
            {
                if (i++ >= maxRunes) break;
                sb.Append(rune.ToString());
            }
            sb.Append('…');
            return sb.ToString();
        }

        private static OnnxExecutionProvider ParseProvider(string s) =>
            s.Trim().ToLowerInvariant() switch
            {
                "cpu" => OnnxExecutionProvider.Cpu,
                "cuda" => OnnxExecutionProvider.Cuda,
                "directml" or "dml" => OnnxExecutionProvider.DirectML,
                _ => throw new ArgumentException($"未知 ExecutionProvider: {s} (cpu|cuda|directml)"),
            };
    }
}
