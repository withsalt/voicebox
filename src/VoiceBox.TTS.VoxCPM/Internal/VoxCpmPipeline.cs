using System.Diagnostics;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// 单次合成的分项耗时统计 (Stopwatch ticks 累计, 由 Prepare 重置)。
    /// 用于定位流式 RTF 瓶颈 (cfm / lm / feat_encoder / vae 各占多少), 常开且开销可忽略
    /// (每步几次 GetTimestamp)。VaeTicks 由消费侧线程累计 (流水线化后与 AR 循环并行)。
    /// </summary>
    internal sealed class VoxCpmStageStats
    {
        public long PrefillTicks;
        public long CfmTicks;
        public long FeatEncTicks;
        public long LmStepTicks;
        public long VaeTicks;
        public int Steps;

        /// <summary>逐步明细 (ms): 观察步耗时随 KV 长度增长的趋势。</summary>
        public readonly List<float> PerStepCfmMs;
        public readonly List<float> PerStepLmMs;

        public VoxCpmStageStats(int expectedSteps = 0)
        {
            int capacity = Math.Max(expectedSteps, 0);
            PerStepCfmMs = new List<float>(capacity);
            PerStepLmMs = new List<float>(capacity);
        }

        public static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        private static double Avg(IReadOnlyList<float> xs, int from, int count)
        {
            if (xs.Count == 0) return 0;
            from = Math.Clamp(from, 0, xs.Count - 1);
            count = Math.Min(count, xs.Count - from);
            double s = 0;
            for (int i = from; i < from + count; i++) s += xs[i];
            return count > 0 ? s / count : 0;
        }

        /// <summary>汇总一行 (patchMs = 每步实时预算, 如 80ms)。</summary>
        public string Summary(double patchMs)
        {
            double cfm = Ms(CfmTicks), lm = Ms(LmStepTicks), fe = Ms(FeatEncTicks), vae = Ms(VaeTicks);
            double ar = cfm + lm + fe;
            int head = Math.Min(10, PerStepLmMs.Count);
            return $"prefill={Ms(PrefillTicks):F0}ms cfm={cfm:F0}ms lm={lm:F0}ms fe={fe:F0}ms vae={vae:F0}ms " +
                   $"steps={Steps} AR均值={(Steps > 0 ? ar / Steps : 0):F1}ms/步 vae均值={(Steps > 0 ? vae / Steps : 0):F1}ms/步 " +
                   $"(预算{patchMs:F0}ms/步; lm首{head}步={Avg(PerStepLmMs, 0, head):F1}ms " +
                   $"末{head}步={Avg(PerStepLmMs, PerStepLmMs.Count - head, head):F1}ms, " +
                   $"cfm首{head}步={Avg(PerStepCfmMs, 0, head):F1}ms 末{head}步={Avg(PerStepCfmMs, PerStepCfmMs.Count - head, head):F1}ms)";
        }
    }

    /// <summary>
    /// VoxCPM 端到端管线: tokenize(+中文拆字) → prefill (lm.onnx, P=0) → 循环
    /// { cfm_decode 扩散出 1 patch (80ms) → stop 判停 → feat_encoder → lm 单步 }
    /// → audio_vae_decoder 解码。忠实移植已验证的 Python 参考 (_engine.py):
    /// fp32 下与 PyTorch 噪声对拍 SNR 80-100dB, 步数逐一致。
    ///
    /// 流式: 每步取最近 streaming_prefix_patches(=3) 个 patch 滑窗解码, 只发尾部 80ms
    /// (前 2 patch 只作卷积上下文, 消除边界伪影)。
    /// </summary>
    internal sealed class VoxCpmPipeline : IDisposable
    {
        private readonly VoxCPMTTSOptions _options;
        private readonly ILogger? _logger;
        private readonly VoxCpmModelConfig _cfg;
        private readonly VoxCpmTokenizer _tokenizer;
        private readonly EmbeddingStore _embeddings;
        private readonly ILmDecoder _lm;
        private readonly CfmDecoder _cfm;
        private readonly FeatEncoder _featEncoder;
        private readonly AudioVaeCodec _vae;
        private readonly VoiceStore _voices;
        private readonly SessionOptions _so;
        private readonly SessionOptions _vaeSo;
        private readonly int _H, _Dz, _Pz, _patchLen;
        private int _seedCounter;
        private static readonly Regex WhitespaceRegex = new(@"\s+", RegexOptions.Compiled);
        private const int BadCaseMaxAttempts = 3;

        /// <summary>最近一次合成的分项耗时 (Prepare 重置; 服务层 gate 保证单并发)。</summary>
        internal VoxCpmStageStats Stats { get; private set; } = new();

        public int SampleRate => _cfg.SampleRate;
        public int Channels => 1;
        public string DefaultVoiceName { get; }

        public VoxCpmPipeline(VoxCPMTTSOptions options, ILogger? logger, bool enableGqaLm = true)
        {
            _options = options;
            _logger = logger;
            if (string.IsNullOrWhiteSpace(options.ModelPath) || !Directory.Exists(options.ModelPath))
                throw new DirectoryNotFoundException($"VoxCPM 模型目录不存在: {options.ModelPath}");

            _cfg = VoxCpmModelConfig.Load(options.ModelPath);
            _H = _cfg.HiddenSize;
            _Dz = _cfg.FeatDim;
            _Pz = _cfg.PatchSize;
            _patchLen = _cfg.SamplesPerStep;

            // 避免 ORT 默认 EXHAUSTIVE cuDNN 搜索 (动态 shape 下病态)
            if (options.ExecutionProvider == OnnxExecutionProvider.Cuda)
                options.CudaTuning ??= new CudaExecutionTuning();

            _tokenizer = new VoxCpmTokenizer(
                _cfg.FilePath(options.ModelPath, "tokenizer"),
                _cfg.FilePath(options.ModelPath, "multichar_zh"));
            _embeddings = new EmbeddingStore(
                _cfg.FilePath(options.ModelPath, "text_embedding"),
                _cfg.TextEmbeddingShape[0], _cfg.TextEmbeddingShape[1]);

            // 原生资源先建到局部变量, 全部成功才赋字段 (半初始化失败不泄漏 session)
            SessionOptions? so = null, vaeSo = null;
            ILmDecoder? lm = null;
            CfmDecoder? cfm = null;
            FeatEncoder? fe = null;
            AudioVaeCodec? vae = null;
            try
            {
                so = OnnxSessionFactory.Create(options);                    // lm / cfm / feat_encoder
                // (实测: feat_encoder 拆独立 1/2 线程池均比共享大池更慢, 保持共享)
                bool cpuEp = options.ExecutionProvider == OnnxExecutionProvider.Cpu;
                if (cpuEp)
                {
                    so.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
                    // 关 intra-op 自旋: AR 循环 cfm→fe→lm 步间隙里, 自旋等待的计算线程会空转
                    // 烧核, 与实时音频链路 (AEC/VAD/barge-in) 争抢 CPU。代价是每步唤醒开销,
                    // RTF 会损失几个百分点 —— 改动后用 "VoxCPM 流式分项" 日志复核仍 <1。
                    so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                }
                // VAE 恒 fp32; CUDA 主 EP 时也放 CUDA, 否则 CPU。
                // CPU 时给 VAE 独立的 (更小的) 线程池: 流式下它与 AR 循环并行在另一条线程上,
                // 与主 session 分开配置, 避免两个自旋线程池争抢同一批核。
                bool vaeOnCuda = options.ExecutionProvider == OnnxExecutionProvider.Cuda;
                int vaeThreads = Math.Max(2, (options.ThreadsNum > 0 ? options.ThreadsNum : Environment.ProcessorCount / 2) / 2);
                if (vaeOnCuda)
                {
                    vaeSo = OnnxSessionFactory.Create(options);
                }
                else
                {
                    vaeSo = OnnxSessionFactory.CreateCpu(vaeThreads);
                    vaeSo.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                    vaeSo.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
                }
                // LM 的增长 KV 在 CUDA 下走 IO-binding 常驻显存
                bool lmIoBinding = options.ExecutionProvider == OnnxExecutionProvider.Cuda;
                // CPU 且包内存在时默认使用 GQA 图（与标准图同权重，非比特一致）。
                string gqaPath = _cfg.Files.TryGetValue("lm_gqa", out string? gqaRel)
                    ? Path.Combine(options.ModelPath, gqaRel)
                    : Path.Combine(options.ModelPath, "lm_gqa.onnx");
                bool useGqa = enableGqaLm && cpuEp && File.Exists(gqaPath);
                if (enableGqaLm && cpuEp && !useGqa)
                    _logger?.LogDebug("未使用 GQA LM 图: 包内缺少 {FileName}", Path.GetFileName(gqaPath));
                if (useGqa)
                    _logger?.LogInformation("使用 GQA LM 图: {Path}", gqaPath);
                string decPath = _cfg.FilePath(options.ModelPath, "audio_vae_decoder");
                string? encPath = null;
                if (_cfg.Files.TryGetValue("audio_vae_encoder", out string? encRel))
                {
                    string p = Path.Combine(options.ModelPath, encRel);
                    if (File.Exists(p)) encPath = p;
                }
                if (encPath is null && options.CloneVoices.Count > 0)
                    throw new FileNotFoundException("配置了克隆音色但部署包缺少 audio_vae_encoder.onnx。");

                // 四组件并行加载: lm (1.8GB) 是长杆, cfm/fe/vae 与其重叠 (瓶颈是反序列化+
                // 单线程 CPU 阶段，而不是磁盘。InferenceSession 构造互相独立，共享的
                // SessionOptions 仅被只读使用 → 并发安全。任一失败时按实际完成状态回收其余。
                SessionOptions soL = so, vaeSoL = vaeSo;   // lambda 捕获非空局部
                var lmTask = Task.Run<ILmDecoder>(() => useGqa
                    ? new LmGqaDecoder(gqaPath, soL, _cfg)
                    : new LmDecoder(_cfg.FilePath(options.ModelPath, "lm"), soL, _cfg, lmIoBinding, options.GpuDeviceId));
                var cfmTask = Task.Run(() => new CfmDecoder(_cfg.FilePath(options.ModelPath, "cfm_decode"), soL, _cfg));
                var feTask = Task.Run(() => new FeatEncoder(_cfg.FilePath(options.ModelPath, "feat_encoder"), soL, _cfg));
                var vaeTask = Task.Run(() => new AudioVaeCodec(decPath, encPath, vaeSoL, _cfg));
                try { Task.WaitAll(lmTask, cfmTask, feTask, vaeTask); }
                catch (AggregateException ex)
                {
                    lm = Completed(lmTask); cfm = Completed(cfmTask); fe = Completed(feTask); vae = Completed(vaeTask);
                    throw ex.InnerExceptions.Count == 1 ? ex.InnerException! : ex;
                }
                lm = lmTask.Result;
                cfm = cfmTask.Result;
                fe = feTask.Result;
                vae = vaeTask.Result;
            }
            catch
            {
                vae?.Dispose(); fe?.Dispose(); cfm?.Dispose(); lm?.Dispose(); vaeSo?.Dispose(); so?.Dispose();
                throw;
            }
            _so = so; _vaeSo = vaeSo; _lm = lm; _cfm = cfm; _featEncoder = fe; _vae = vae;

            _voices = new VoiceStore(options.CloneVoices, _vae, _cfg, logger);
            DefaultVoiceName = !string.IsNullOrWhiteSpace(options.DefaultVoice)
                ? options.DefaultVoice!
                : VoiceStore.DefaultVoiceName;
            if (!_voices.Contains(DefaultVoiceName))
                throw new ArgumentException($"默认音色 {DefaultVoiceName} 未注册。");

            TryWarmup();
        }

        public IReadOnlyList<TtsVoice> ListVoices()
        {
            var list = new List<TtsVoice>(_voices.Names.Count + 1)
            {
                new() { Name = VoiceStore.DefaultVoiceName, Description = "模型默认音色 (无 prompt)" },
            };
            foreach (string name in _voices.Names)
                list.Add(new TtsVoice { Name = name, Description = "克隆音色" });
            return list;
        }

        // ---- 整段 / 流式入口 ----

        public float[] SynthesizeWhole(string text, TtsSynthesisOptions? opts, out string voice, CancellationToken ct)
        {
            var run = Prepare(text, opts, out voice, out long seed);
            // badcase 重试 (对齐 core.py retry_badcase 默认开启): 生成 patch 数达到
            // 文本 token 数 × ratio 视为跑飞 (复读/呓语), 换 seed 重跑, 最多 3 次生成。
            int maxAttempts = BadCaseMaxAttempts;
            List<float[]> patches = [];
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                patches = new List<float[]>(run.MaxSteps);
                try
                {
                    foreach (var p in GeneratePatches(run, ct)) patches.Add(p);
                }
                finally { _lm.EndGeneration(); }
                ct.ThrowIfCancellationRequested();

                if (patches.Count < run.TargetTokens * _cfg.Generation.MaxLenTextRatio ||
                    attempt == maxAttempts - 1)
                    break;
                _logger?.LogWarning(
                    "VoxCPM badcase: {Patches} patches ≥ {Tokens} tokens × {Ratio}, 换 seed 重试 ({Attempt}/{Max})",
                    patches.Count, run.TargetTokens, _cfg.Generation.MaxLenTextRatio, attempt + 1, maxAttempts - 1);
                run.Noise = new RngNoise(seed + attempt + 1);
            }
            float[] result = DecodeWhole(run, patches);
            LogStats("VoxCPM 整段分项: {Summary}");
            return result;
        }

        public IEnumerable<float[]> SynthesizeStream(string text, TtsSynthesisOptions? opts, CancellationToken ct)
        {
            // 流式不支持 badcase 重试 (块已发出, 与 python 参考一致)
            var run = Prepare(text, opts, out _, out _);
            int prefix = _cfg.Generation.StreamingPrefixPatches;

            // ---- 流水线: AR 循环 (cfm→fe→lm) 与 VAE 滑窗解码无数据依赖, 分两条线程并行 ----
            // 生产者跑 AR 循环产出 latent patch; 本迭代器线程消费 patch 做滑窗解码并发块。
            // patch 顺序 / 滑窗组成 / 解码输入与串行版完全一致 → 输出逐比特一致;
            // 步耗时从 (AR + VAE) 降为 max(AR, VAE)。队列有界 (patch 仅 512B, 容量只为解耦)。
            using var queue = new System.Collections.Concurrent.BlockingCollection<float[]>(boundedCapacity: 16);
            using var prodCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            var producer = Task.Run(() =>
            {
                try
                {
                    foreach (var patch in GeneratePatches(run, prodCts.Token))
                        queue.Add(patch, prodCts.Token);
                }
                catch (OperationCanceledException) when (prodCts.IsCancellationRequested)
                {
                    // 流式消费结束或请求被打断属于正常控制流。取消异常必须在生产任务
                    // 边界内收口，否则 Task 会以异常状态结束，并可能被宿主当成未处理异常。
                }
                finally
                {
                    _lm.EndGeneration();
                    // 无论正常结束/取消/异常都要收口, 否则消费侧永久阻塞;
                    // 消费侧提前退出已 Dispose 队列时此调用可能抛 ODE, 吞掉即可。
                    try { queue.CompleteAdding(); } catch (ObjectDisposedException) { }
                }
            }, CancellationToken.None);

            // 滑窗上下文: 克隆时用 prompt 末 (prefix-1) 个 patch 预填, 与 torch 一致
            var window = new List<float[]>(prefix + 1);
            var latentBuffers = new float[prefix + 1][];
            if (run.ContextPatches.Count > 0)
                window.AddRange(run.ContextPatches);
            try
            {
                foreach (var patch in queue.GetConsumingEnumerable(ct))
                {
                    window.Add(patch);
                    if (window.Count > prefix) window.RemoveAt(0);
                    // 解码滑窗 [w*Pz] 帧, 只发尾部 patchLen 样本
                    long t0 = Stopwatch.GetTimestamp();
                    float[] lat = GetLatentBuffer(latentBuffers, window.Count);
                    FillPatchesToLatents(window, lat);
                    float[] wav = _vae.Decode(lat, window.Count * _Pz);
                    var chunk = new float[_patchLen];
                    Array.Copy(wav, wav.Length - _patchLen, chunk, 0, _patchLen);
                    Stats.VaeTicks += Stopwatch.GetTimestamp() - t0;
                    yield return chunk;
                }
                // 正常消费完: 传播生产者的异常 (若有); 生产者此刻已 CompleteAdding, 等待是短暂的
                producer.GetAwaiter().GetResult();
            }
            finally
            {
                // 消费侧提前退出 (break / 取消 / 下游异常): 叫停生产者并等它放掉 LM 资源,
                // 保证返回后 pipeline 可立即开始下一次合成 (服务层 gate 依赖此语义)。
                prodCts.Cancel();
                try { producer.Wait(TimeSpan.FromSeconds(30)); }
                catch (AggregateException) { /* 异常已在上面传播过, 或为取消 */ }
                LogStats("VoxCPM 流式分项: {Summary}");
            }
        }

        // ---- 测试钩子 ----

        /// <summary>测试钩子: 用外部噪声序列生成 latent patch 序列 (与 golden noise 对拍)。</summary>
        internal List<float[]> GeneratePatchesForTest(string text, string? promptText, float[]? promptFeat,
            int promptPatches, IReadOnlyList<float[]> noises, CancellationToken ct = default)
        {
            var run = PrepareRaw(text, promptText, promptFeat, promptPatches,
                _cfg.Generation.CfgValue, _cfg.Generation.MinLen, _cfg.MaxLength, new QueueNoise(noises));
            var patches = new List<float[]>(run.MaxSteps);
            try
            {
                foreach (var p in GeneratePatches(run, ct)) patches.Add(p);
            }
            finally { _lm.EndGeneration(); }
            return patches;
        }

        /// <summary>测试钩子: latent patch 序列 → 整段波形 (无 prompt 上下文)。</summary>
        internal float[] DecodePatchesForTest(IReadOnlyList<float[]> patches)
            => _vae.Decode(PatchesToLatents(patches), patches.Count * _Pz);

        internal int[] TokenizeForTest(string text) => _tokenizer.Encode(text);

        /// <summary>测试钩子: 从包配置读派生量, 测试不再硬编码 1280/3 等常量。</summary>
        internal int SamplesPerStepForTest => _patchLen;
        internal int StreamingPrefixPatchesForTest => _cfg.Generation.StreamingPrefixPatches;

        /// <summary>
        /// 短文本自动补标点: ≤4 个字 (按 Unicode 码点计, 忽略首尾空白) 且末尾无标点时补句号。
        /// 超短无标点文本的判停不稳, 模型易跑飞成呓语; 补标点给出明确的语句边界。
        /// </summary>
        internal static string AutoPunctuateShortText(string text)
        {
            int start = 0;
            int end = text.Length;
            while (start < end && char.IsWhiteSpace(text[start])) start++;
            while (end > start && char.IsWhiteSpace(text[end - 1])) end--;

            ReadOnlySpan<char> trimmed = text.AsSpan(start, end - start);
            if (trimmed.IsEmpty) return text;

            int runes = 0;
            foreach (var _ in trimmed.EnumerateRunes())
            {
                if (++runes > 4) return text;   // 超过 4 个字, 不处理
            }

            var last = System.Text.Rune.DecodeLastFromUtf16(trimmed, out System.Text.Rune lastRune, out _)
                == System.Buffers.OperationStatus.Done ? lastRune : default;
            if (System.Text.Rune.IsPunctuation(last) || System.Text.Rune.IsSymbol(last))
                return text;   // 已带标点 (含 。！？…～ 等; ～ 属 Symbol)

            // 末字是 CJK 补中文句号, 否则补英文句号
            bool cjk = last.Value is >= 0x4E00 and <= 0x9FFF;
            return text.Insert(end, cjk ? "。" : ".");
        }

        internal float[] EncodePromptForTest(string wavPath)
        {
            var voice = new VoxCpmCloneVoice { Name = "__test", PromptWavPath = wavPath, PromptText = "t" };
            var store = new VoiceStore([voice], _vae, _cfg, null);
            return store.Resolve("__test")!.PromptFeat;
        }

        // ---- 生成核心 ----

        private interface INoiseSource { float[] Next(int count); }

        private sealed class RngNoise(long seed) : INoiseSource
        {
            private readonly GaussianNoise _rng = new(seed);
            public float[] Next(int count) => _rng.Fill(count);
        }

        private sealed class QueueNoise(IReadOnlyList<float[]> noises) : INoiseSource
        {
            private int _i;
            public float[] Next(int count)
            {
                if (_i >= noises.Count)
                    throw new InvalidOperationException($"外部噪声序列耗尽 (提供 {noises.Count} 步)。");
                var n = noises[_i++];
                return n.Length == count ? n : throw new ArgumentException($"噪声长度 {n.Length} != {count}");
            }
        }

        private sealed class RunState
        {
            public required int[] TextIds { get; init; }          // 含 audio_start, 不含 prompt pad
            public required int TargetTokens { get; init; }       // 目标文本 token 数 (max_len 约束用)
            public required float[]? PromptFeat { get; init; }    // [Tp*Pz*Dz]
            public required int PromptPatches { get; init; }
            public required float CfgValue { get; init; }
            public required int MinLen { get; init; }
            public required int MaxSteps { get; init; }
            public required INoiseSource Noise { get; set; }      // badcase 重试时换 seed 重建
            public List<float[]> ContextPatches { get; } = [];    // prompt 末 (prefix-1) patch
        }

        private RunState Prepare(string text, TtsSynthesisOptions? opts, out string voiceName, out long actualSeed)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("合成文本不能为空。", nameof(text));
            // 与 python core._generate 一致的文本清理: 换行归一 + 空白折叠 (参考实现不 Trim)
            text = WhitespaceRegex.Replace(text.Replace('\n', ' '), " ");
            text = AutoPunctuateShortText(text);

            voiceName = opts?.Voice is { Length: > 0 } v ? v : DefaultVoiceName;
            var voice = _voices.Resolve(voiceName);

            float cfg = _options.CfgValue ?? _cfg.Generation.CfgValue;
            int minLen = _options.MinLen ?? _cfg.Generation.MinLen;
            // 服务级 MaxLen 是护栏: 单次调用的 MaxNewFrames 只能收紧, 不能放宽
            int maxLen = Math.Min(opts?.MaxNewFrames ?? int.MaxValue, _options.MaxLen ?? _cfg.MaxLength);
            int? seed = opts?.Seed ?? _options.DefaultSeed;
            actualSeed = seed ?? Interlocked.Increment(ref _seedCounter) * 7919L + Environment.TickCount64;

            return PrepareRaw(text, voice?.PromptText, voice?.PromptFeat, voice?.PromptPatches ?? 0,
                cfg, minLen, maxLen, new RngNoise(actualSeed));
        }

        private RunState PrepareRaw(string text, string? promptText, float[]? promptFeat, int promptPatches,
            float cfgValue, int minLen, int maxLen, INoiseSource noise)
        {
            string full = string.IsNullOrEmpty(promptText) ? text : promptText + text;
            int[] ids = _tokenizer.Encode(full);
            int targetTokens = string.IsNullOrEmpty(promptText) ? ids.Length : _tokenizer.Encode(text).Length;
            var withStart = new int[ids.Length + 1];
            Array.Copy(ids, withStart, ids.Length);
            withStart[^1] = _cfg.AudioStartToken;

            // prefill 长度 + 生成步数受 lm.onnx 内烘焙的 RoPE 表 (max_length=4096 行) 约束;
            // 越界会在推理中途 Gather 崩溃, 这里前置为清晰错误/自然截断。
            int totalPrefill = withStart.Length + promptPatches;
            if (totalPrefill >= _cfg.MaxLength)
                throw new ArgumentException(
                    $"输入过长: 文本 {withStart.Length} tokens + prompt {promptPatches} patches = {totalPrefill}, " +
                    $"超出模型上限 {_cfg.MaxLength}。请缩短文本或 prompt 音频。");

            int limit = Math.Min((int)(targetTokens * _cfg.Generation.MaxLenTextRatio) + _cfg.Generation.MaxLenTextBias,
                Math.Min(maxLen, _cfg.MaxLength));
            limit = Math.Min(limit, _cfg.MaxLength - totalPrefill);
            int maxSteps = Math.Max(limit, 1);
            Stats = new VoxCpmStageStats(maxSteps);   // Prepare 与测试钩子共同的入口, 每次合成重置
            var run = new RunState
            {
                TextIds = withStart,
                TargetTokens = targetTokens,
                PromptFeat = promptFeat,
                PromptPatches = promptPatches,
                CfgValue = cfgValue,
                MinLen = minLen,
                MaxSteps = maxSteps,
                Noise = noise,
            };
            if (promptFeat is not null && promptPatches > 0)
            {
                int ctx = Math.Min(_cfg.Generation.StreamingPrefixPatches - 1, promptPatches);
                int stride = _Pz * _Dz;
                for (int i = promptPatches - ctx; i < promptPatches; i++)
                {
                    var p = new float[stride];
                    Array.Copy(promptFeat, (long)i * stride, p, 0, stride);
                    run.ContextPatches.Add(p);
                }
            }
            return run;
        }

        /// <summary>核心 AR 循环, 产出 latent patches ([Pz*Dz] 行主序 [Pz,Dz])。</summary>
        private IEnumerable<float[]> GeneratePatches(RunState run, CancellationToken ct)
        {
            int t = run.TextIds.Length;
            int tp = run.PromptPatches;
            int total = t + tp;
            int stride = _Pz * _Dz;

            // ---- prefill 输入构造 (复刻 _modules.build_prefill_arrays) ----
            // ids: [文本ids ++ tp个pad(0)]; text_mask 前 t 个 1; feat 前 t 个全零, 后 tp 个 prompt
            var idsPadded = new int[total];
            Array.Copy(run.TextIds, idsPadded, t);
            float[] textEmbed = _embeddings.Lookup(idsPadded);   // [total*H]

            var fem = new float[total * _H];         // audio_mask * feat_embed
            var combined = textEmbed;                // 前 t 行 = text_embed (audio 行稍后覆盖)
            var fsqMask = new float[total];          // == audio_mask
            if (tp > 0)
            {
                float[] featEmbed = _featEncoder.Encode(run.PromptFeat!, tp);   // [tp*H]
                for (int i = 0; i < tp; i++)
                {
                    Array.Copy(featEmbed, i * _H, fem, (t + i) * _H, _H);
                    Array.Copy(featEmbed, i * _H, combined, (t + i) * _H, _H);  // text_mask=0 处 = feat_embed
                    fsqMask[t + i] = 1f;
                }
            }

            _lm.Reset();
            long t0 = Stopwatch.GetTimestamp();
            var lmOut = _lm.Prefill(combined, fem, fsqMask, total);
            Stats.PrefillTicks += Stopwatch.GetTimestamp() - t0;

            // cond = 最后一个 patch 的特征 [Dz,Pz] (无 prompt 时全零)
            var cond = new float[stride];
            if (tp > 0)
                TransposePatch(run.PromptFeat!.AsSpan((tp - 1) * stride, stride), cond);

            for (int i = 0; i < run.MaxSteps; i++)
            {
                ct.ThrowIfCancellationRequested();

                float[] noise = run.Noise.Next(stride);
                t0 = Stopwatch.GetTimestamp();
                float[] pred = _cfm.Decode(lmOut.DitHidden, cond, noise, run.CfgValue);   // [Dz,Pz]
                long tCfm = Stopwatch.GetTimestamp() - t0;
                Stats.CfmTicks += tCfm;
                Stats.PerStepCfmMs.Add((float)VoxCpmStageStats.Ms(tCfm));
                Stats.Steps++;
                var patch = new float[stride];
                TransposePatchBack(pred, patch);                                          // → [Pz,Dz]
                yield return patch;
                cond = pred;

                // stop 用本步开头的 stop_logits (上一次 lm 调用的输出), 与 torch 一致
                if (i > run.MinLen && lmOut.StopLogits[1] > lmOut.StopLogits[0])
                    break;

                t0 = Stopwatch.GetTimestamp();
                float[] currEmbed = _featEncoder.Encode(patch, 1);   // [1,Pz,Dz] → [H]
                Stats.FeatEncTicks += Stopwatch.GetTimestamp() - t0;

                t0 = Stopwatch.GetTimestamp();
                lmOut = _lm.Step(currEmbed);
                long tLm = Stopwatch.GetTimestamp() - t0;
                Stats.LmStepTicks += tLm;
                Stats.PerStepLmMs.Add((float)VoxCpmStageStats.Ms(tLm));
            }
        }

        private float[] DecodeWhole(RunState run, List<float[]> patches)
        {
            if (patches.Count == 0) return [];
            // 克隆时前置 prompt 上下文 patch, 解码后丢弃其样本 (边界连续性), 与 torch 一致
            IReadOnlyList<float[]> all = patches;
            if (run.ContextPatches.Count > 0)
            {
                var withContext = new List<float[]>(run.ContextPatches.Count + patches.Count);
                withContext.AddRange(run.ContextPatches);
                withContext.AddRange(patches);
                all = withContext;
            }
            long t0 = Stopwatch.GetTimestamp();
            float[] wav = _vae.Decode(PatchesToLatents(all), all.Count * _Pz);
            Stats.VaeTicks += Stopwatch.GetTimestamp() - t0;
            if (run.ContextPatches.Count == 0) return wav;
            int skip = _patchLen * run.ContextPatches.Count;
            var outp = new float[wav.Length - skip];
            Array.Copy(wav, skip, outp, 0, outp.Length);
            return outp;
        }

        /// <summary>patch 列表 ([Pz,Dz] 行主序) → [Dz, n*Pz] 行主序 latent 矩阵。</summary>
        private float[] PatchesToLatents(IReadOnlyList<float[]> patches)
        {
            int n = patches.Count, frames = n * _Pz;
            var lat = new float[_Dz * frames];
            FillPatchesToLatents(patches, lat);
            return lat;
        }

        private float[] GetLatentBuffer(float[][] buffers, int patchCount)
        {
            int length = _Dz * patchCount * _Pz;
            float[]? buffer = buffers[patchCount];
            if (buffer is null || buffer.Length != length)
            {
                buffer = new float[length];
                buffers[patchCount] = buffer;
            }
            return buffer;
        }

        private void FillPatchesToLatents(IReadOnlyList<float[]> patches, Span<float> lat)
        {
            int n = patches.Count, frames = n * _Pz;
            for (int i = 0; i < n; i++)
            {
                float[] p = patches[i];
                for (int r = 0; r < _Pz; r++)
                    for (int d = 0; d < _Dz; d++)
                        lat[d * frames + i * _Pz + r] = p[r * _Dz + d];
            }
        }

        private void LogStats(string message)
        {
            if (_logger?.IsEnabled(LogLevel.Debug) == true)
                _logger.LogDebug(message, Stats.Summary(1000.0 * _patchLen / SampleRate));
        }

        /// <summary>[Pz,Dz] → [Dz,Pz]。</summary>
        private void TransposePatch(ReadOnlySpan<float> src, Span<float> dst)
        {
            for (int r = 0; r < _Pz; r++)
                for (int d = 0; d < _Dz; d++)
                    dst[d * _Pz + r] = src[r * _Dz + d];
        }

        /// <summary>并行加载失败时取回已成功的组件 (供回收); 未完成/失败的返回 null。</summary>
        private static T? Completed<T>(Task<T> t) where T : class
            => t.IsCompletedSuccessfully ? t.Result : null;

        /// <summary>[Dz,Pz] → [Pz,Dz]。</summary>
        private void TransposePatchBack(ReadOnlySpan<float> src, Span<float> dst)
        {
            for (int d = 0; d < _Dz; d++)
                for (int r = 0; r < _Pz; r++)
                    dst[r * _Dz + d] = src[d * _Pz + r];
        }

        private void TryWarmup()
        {
            try
            {
                var sw = Stopwatch.StartNew();
                var opts = new TtsSynthesisOptions { MaxNewFrames = 3, Seed = 0, Voice = VoiceStore.DefaultVoiceName };
                // 走流式路径: 除 prefill/decode/cfm 外, 还预热滑窗 VAE 解码的 1/2/3-patch
                // 三种输入形状 (流式热路径的全部 kernel), 消除首次流式合成的冷核尖峰。
                foreach (var _ in SynthesizeStream("你好。", opts, CancellationToken.None)) { }
                _logger?.LogInformation("VoxCPM 预热完成, 用时 {Ms} ms", sw.ElapsedMilliseconds);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "VoxCPM 预热失败 (忽略, 不影响后续合成)。");
            }
        }

        public void Dispose()
        {
            _vae.Dispose();
            _featEncoder.Dispose();
            _cfm.Dispose();
            _lm.Dispose();
            _vaeSo.Dispose();
            _so.Dispose();
            _embeddings.Dispose();
        }
    }
}
