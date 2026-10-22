using System.Buffers;
using System.Collections.ObjectModel;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using System.Security.Cryptography;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.SpeakerService.Models;
using VoiceBox.Speaker.WeSpeaker.Internal;
using VoiceBox.Speaker.WeSpeaker.Models;

namespace VoiceBox.Speaker.WeSpeaker
{
    /// <summary>
    /// 基于 WeSpeaker 非 LM cnceleb_resnet34.onnx 的说话人验证服务。
    /// <para>
    /// 前端: 80 维 Hamming Kaldi fbank + 逐窗 CMN；模型输出 256 维 embedding，使用原始余弦打分。
    /// ResNet34/TSTP 是非因果段级模型，不存在可缓存的增量状态。流式检测采用应用层滑动窗口:
    /// 每 <see cref="WeSpeakerOptions.HopMs"/> 对最近 <see cref="WeSpeakerOptions.WindowMs"/> 音频做整窗推理，与注册说话人
    /// 比对后经指数平滑产出 <see cref="OnResult"/>。注册与离线验证走同一动态长度模型。
    /// 可选 early-exit (<see cref="WeSpeakerOptions.EnableEarlyExit"/>): 首个完整窗就绪前对前缀音频
    /// 提前打分, 产出 IsEarlyExit=true 的临时结果, 不影响完整窗时间线。
    /// </para>
    /// 线程安全: 所有公共方法可被并发调用 (内部串行化); 事件在锁外触发。
    /// 注意: 多线程并发调用 FeedAudio 时, OnResult 的送达顺序不做保证 (窗口计算串行, 但发布可交错);
    /// 实时音频链路请从单一线程喂入。
    /// </summary>
    public sealed class WeSpeakerService : ISpeakerService
    {
        private readonly ILogger<WeSpeakerService> _logger;
        private readonly WeSpeakerModelConfig _config;
        private readonly WeSpeakerOnnxRunner _runner;
        private readonly StreamingWeSpeakerFbank _fbank;
        private readonly Dictionary<string, float[]> _profiles = new(StringComparer.Ordinal);
        private readonly Dictionary<string, float> _smoothedScores = new(StringComparer.Ordinal);
        private static readonly IReadOnlyDictionary<string, float> EmptyScores =
            new ReadOnlyDictionary<string, float>(
                new Dictionary<string, float>(StringComparer.Ordinal));
        private readonly float[] _windowFeats;
        private readonly int _windowFrames;
        private readonly int _hopFrames;
        private readonly float _threshold;
        private readonly float _scoreSmoothing;
        private readonly float _minWindowRms;
        private readonly int _minEnrollMs;
        private readonly int _minEnrollSamples;
        private readonly bool _enableEarlyExit;
        private readonly int _earlyExitMinFrames;
        private long _nextWindowEnd;
        private long _nextEarlyExitEnd;
        private readonly object _lock = new();
        private bool _disposed;

        /// <inheritdoc/>
        public event EventHandler<SpeakerEventArgs>? OnResult;

        /// <inheritdoc/>
        public int EmbeddingSize => _config.EmbeddingSize;

        /// <inheritdoc/>
        public int SampleRate => _config.SampleRate;

        /// <inheritdoc/>
        public float Threshold => _threshold;

        /// <inheritdoc/>
        public int StreamingWindowSamples => checked(
            (_windowFrames - 1) * StreamingWeSpeakerFbank.FrameShift + StreamingWeSpeakerFbank.FrameLength);

        /// <inheritdoc/>
        public int StreamingHopSamples => checked(_hopFrames * StreamingWeSpeakerFbank.FrameShift);

        /// <inheritdoc/>
        public IReadOnlyCollection<string> EnrolledSpeakers
        {
            get { lock (_lock) return _profiles.Keys.ToArray(); }
        }

        public WeSpeakerService(IOptions<WeSpeakerOptions> options)
            : this(options, NullLogger<WeSpeakerService>.Instance)
        {
        }

        public WeSpeakerService(IOptions<WeSpeakerOptions> options, ILogger<WeSpeakerService> logger)
        {
            WeSpeakerOptions effectiveOptions =
                options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<WeSpeakerService>.Instance;

            if (effectiveOptions.SampleRate != StreamingWeSpeakerFbank.SampleRate)
                throw new NotSupportedException($"WeSpeaker only supports {StreamingWeSpeakerFbank.SampleRate}Hz sample rate, got {effectiveOptions.SampleRate}.");
            if (effectiveOptions.ChannelCount != 1)
                throw new NotSupportedException($"WeSpeaker only supports mono audio, got {effectiveOptions.ChannelCount} channels.");
            if (effectiveOptions.SampleFormat != SampleFormat.Float32)
                throw new NotSupportedException($"WeSpeaker only supports Float32 samples, got {effectiveOptions.SampleFormat}.");
            // 全部用"不在合法区间即拒绝"的否定式写法, NaN 无法通过任何区间判断
            if (effectiveOptions.WindowMs is < 500 or > 30000)
                throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.WindowMs, "WindowMs must be in [500, 30000].");
            if (effectiveOptions.HopMs <= 0 || effectiveOptions.HopMs > effectiveOptions.WindowMs)
                throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.HopMs, "HopMs must be in (0, WindowMs].");
            if (!(effectiveOptions.ScoreSmoothing > 0f && effectiveOptions.ScoreSmoothing <= 1f))
                throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.ScoreSmoothing, "ScoreSmoothing must be in (0, 1].");
            if (!(effectiveOptions.MinWindowRms >= 0f && effectiveOptions.MinWindowRms < 1f))
                throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.MinWindowRms, "MinWindowRms must be in [0, 1).");
            _minEnrollSamples = GetMinEnrollSamples(effectiveOptions.MinEnrollMs, nameof(options));
            if (effectiveOptions.Threshold is { } t && !(t >= -1f && t <= 1f))
                throw new ArgumentOutOfRangeException(nameof(options), t, "Threshold must be in [-1, 1].");
            if (effectiveOptions.EnableEarlyExit &&
                (effectiveOptions.EarlyExitMinMs < 100 || effectiveOptions.EarlyExitMinMs >= effectiveOptions.WindowMs))
                throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.EarlyExitMinMs, "EarlyExitMinMs must be in [100, WindowMs).");

            string modelDir = ResolveModelDir(effectiveOptions);
            string configPath = Path.Combine(modelDir, "config.ini");
            _config = WeSpeakerModelConfig.Load(configPath);
            string modelPath = Path.Combine(modelDir, _config.ModelFile);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"WeSpeaker model file not found: {modelPath}");
            ValidateModelHash(modelPath, _config.ModelSha256);

            _threshold = effectiveOptions.Threshold ?? _config.Threshold;
            _scoreSmoothing = effectiveOptions.ScoreSmoothing;
            _minWindowRms = effectiveOptions.MinWindowRms;
            _minEnrollMs = effectiveOptions.MinEnrollMs;
            _windowFrames = SamplesToFrames(MillisecondsToSamples(effectiveOptions.WindowMs, _config.SampleRate));
            _hopFrames = Math.Max(1, MillisecondsToSamples(effectiveOptions.HopMs, _config.SampleRate) / StreamingWeSpeakerFbank.FrameShift);
            _nextWindowEnd = _windowFrames;
            _enableEarlyExit = effectiveOptions.EnableEarlyExit;
            if (_enableEarlyExit)
            {
                _earlyExitMinFrames = SamplesToFrames(MillisecondsToSamples(effectiveOptions.EarlyExitMinMs, _config.SampleRate));
                // 毫秒校验通过后帧粒度取整仍可能与整窗帧数撞车 (如 499ms 对 500ms 窗), 必须按帧再验一次
                if (!(_earlyExitMinFrames >= 1 && _earlyExitMinFrames < _windowFrames))
                    throw new ArgumentOutOfRangeException(nameof(options), effectiveOptions.EarlyExitMinMs,
                        $"EarlyExitMinMs resolves to {_earlyExitMinFrames} frames; must be in [1, window frames {_windowFrames}).");
                _nextEarlyExitEnd = _earlyExitMinFrames;
            }
            _windowFeats = new float[_windowFrames * _config.FeatureDim];

            _fbank = CreateFbank();
            _runner = new WeSpeakerOnnxRunner(modelPath, effectiveOptions, _config);

            _logger.LogInformation(
                "WeSpeaker SV loaded: {ModelPath}, embedding={EmbeddingSize}, window={WindowFrames} frames, hop={HopFrames} frames, threshold={Threshold}",
                modelPath, _config.EmbeddingSize, _windowFrames, _hopFrames, _threshold);
        }

        // ---------------- 离线 ----------------

        /// <inheritdoc/>
        public float[] ExtractEmbedding(ReadOnlySpan<float> samples)
        {
            float[] feats = ComputeOfflineFeatures(samples, out int frames);
            try
            {
                lock (_lock)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    return RunNormalized(feats, frames);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(feats);
            }
        }

        /// <inheritdoc/>
        public float CompareEmbeddings(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
        {
            if (embedding1.Length == 0 || embedding1.Length != embedding2.Length)
                throw new ArgumentException($"Embedding dimensions mismatch: {embedding1.Length} vs {embedding2.Length}.");
            double dot = Dot(embedding1, embedding2);
            double n1 = Dot(embedding1, embedding1);
            double n2 = Dot(embedding2, embedding2);
            if (!double.IsFinite(dot) || !double.IsFinite(n1) || !double.IsFinite(n2))
                throw new ArgumentException("Embeddings must contain only finite values.");
            if (n1 <= 0 || n2 <= 0) return 0f;
            return Math.Clamp((float)(dot / (Math.Sqrt(n1) * Math.Sqrt(n2))), -1f, 1f);
        }

        /// <inheritdoc/>
        public float Verify(ReadOnlySpan<float> samples1, ReadOnlySpan<float> samples2)
        {
            float[] e1 = ExtractEmbedding(samples1);
            float[] e2 = ExtractEmbedding(samples2);
            return CompareEmbeddings(e1, e2);
        }

        // ---------------- 注册 ----------------

        /// <inheritdoc/>
        public void Enroll(string speakerName, ReadOnlySpan<float> samples)
        {
            if (string.IsNullOrWhiteSpace(speakerName))
                throw new ArgumentException("Speaker name cannot be null or empty.", nameof(speakerName));
            if (samples.Length < _minEnrollSamples)
                throw new ArgumentException(
                    $"Enrollment audio too short: {samples.Length / (float)_config.SampleRate:F2}s < {_minEnrollMs / 1000f:F2}s.", nameof(samples));

            float[] feats = ComputeOfflineFeatures(samples, out int frames);
            try
            {
                int dim = _config.FeatureDim;

                lock (_lock)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    float[] profile;
                    if (frames <= _windowFrames)
                    {
                        profile = RunNormalized(feats, frames);
                    }
                    else
                    {
                        // 与流式打分同窗长的滑窗平均 (窗长失配会拉低余弦得分); 半窗重叠, 强制覆盖末尾。
                        var acc = new float[_config.EmbeddingSize];
                        int step = Math.Max(1, _windowFrames / 2);
                        int last = frames - _windowFrames;
                        var sub = new float[_windowFrames * dim];
                        for (int start = 0; ; start += step)
                        {
                            int s = Math.Min(start, last);
                            Array.Copy(feats, s * dim, sub, 0, _windowFrames * dim);
                            // feats 已整段 CMN; 逐窗再次 CMN 与流式路径对齐
                            if (_config.ApplyCmn) CmnInPlace(sub, _windowFrames, dim);
                            float[] emb = RunNormalized(sub, _windowFrames);
                            for (int i = 0; i < acc.Length; i++) acc[i] += emb[i];
                            if (s == last) break;
                        }
                        if (!Normalize(acc))
                            throw new ArgumentException("Enrollment audio produced a non-finite embedding.", nameof(samples));
                        profile = acc;
                    }
                    _profiles[speakerName] = profile;
                    _smoothedScores.Remove(speakerName);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(feats);
            }
            _logger.LogInformation("WeSpeaker SV enrolled speaker '{Speaker}' from {Seconds:F2}s audio.",
                speakerName, samples.Length / (float)_config.SampleRate);
        }

        /// <inheritdoc/>
        public bool Unenroll(string speakerName)
        {
            if (string.IsNullOrWhiteSpace(speakerName)) return false;
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _smoothedScores.Remove(speakerName);
                return _profiles.Remove(speakerName);
            }
        }

        // ---------------- 流式 ----------------

        /// <inheritdoc/>
        public void FeedAudio(ReadOnlySpan<float> samples)
        {
            SpeakerEventArgs? firstResult = null;
            List<SpeakerEventArgs>? additionalResults = null;
            void Collect(SpeakerEventArgs result)
            {
                if (firstResult is null)
                    firstResult = result;
                else
                    (additionalResults ??= new List<SpeakerEventArgs>()).Add(result);
            }

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _fbank.AcceptWaveform(samples);

                // early-exit: 首个完整窗就绪前, 对最新可用的前缀网格点提前打分
                if (_enableEarlyExit && _nextWindowEnd == _windowFrames &&
                    _fbank.NumFramesReady < _windowFrames &&
                    _fbank.NumFramesReady >= _nextEarlyExitEnd)
                {
                    long prefixEnd = _earlyExitMinFrames +
                        (_fbank.NumFramesReady - _earlyExitMinFrames) / _hopFrames * _hopFrames;
                    SpeakerEventArgs? earlyResult = ProcessPrefixWindow(prefixEnd);
                    _nextEarlyExitEnd = prefixEnd + _hopFrames;
                    if (earlyResult is not null)
                        Collect(earlyResult);
                }

                while (_fbank.NumFramesReady >= _nextWindowEnd)
                {
                    SpeakerEventArgs? result = ProcessWindow(_nextWindowEnd);
                    if (result is not null)
                        Collect(result);
                    // 下一窗起点之前的帧不再需要
                    long nextStart = _nextWindowEnd + _hopFrames - _windowFrames;
                    if (nextStart > 0) _fbank.Pop(nextStart);
                    _nextWindowEnd += _hopFrames;
                }
            }
            if (firstResult is not null)
            {
                OnResult?.Invoke(this, firstResult);
                if (additionalResults is null) return;
                foreach (var args in additionalResults)
                    OnResult?.Invoke(this, args);
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _fbank.Reset();
                _nextWindowEnd = _windowFrames;
                _nextEarlyExitEnd = _earlyExitMinFrames;
                _smoothedScores.Clear();
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _runner.Dispose();
                _profiles.Clear();
                _smoothedScores.Clear();
                OnResult = null;
            }
        }

        // ---------------- 内部 ----------------

        private SpeakerEventArgs? ProcessWindow(long windowEnd)
        {
            if (OnResult is null)
                return null; // 无订阅者: 结果无人消费, 跳过整窗推理 (帧推进/丢弃由调用方照常进行)

            long start = windowEnd - _windowFrames;
            int dim = _config.FeatureDim;

            if (_minWindowRms > 0f)
            {
                double sumSq = _fbank.SumSquaredRms(start, _windowFrames);
                if (!(Math.Sqrt(sumSq / _windowFrames) >= _minWindowRms))
                    return null; // 静音窗 (或含 NaN 的坏窗): 跳过推理, 不产出结果
            }

            _fbank.CopyFramesTo(start, _windowFrames, _windowFeats);
            if (_config.ApplyCmn) CmnInPlace(_windowFeats, _windowFrames, dim);

            float[] embedding = _runner.RunReusable(_windowFeats, _windowFrames);
            if (!Normalize(embedding))
                return null; // 非有限 embedding (上游 NaN/Inf 样本): 丢弃该窗, 避免毒化 EMA 平滑分数

            string? best = null;
            float bestScore = 0f;
            Dictionary<string, float>? scores = _profiles.Count == 0
                ? null
                : new Dictionary<string, float>(_profiles.Count, StringComparer.Ordinal);
            foreach (var (name, profile) in _profiles)
            {
                float raw = Math.Clamp((float)Dot(embedding, profile), -1f, 1f);
                float smoothed = _smoothedScores.TryGetValue(name, out float prev)
                    ? _scoreSmoothing * raw + (1f - _scoreSmoothing) * prev
                    : raw;
                _smoothedScores[name] = smoothed;
                scores![name] = smoothed;
                if (best is null || smoothed > bestScore)
                {
                    best = name;
                    bestScore = smoothed;
                }
            }

            var audioTime = TimeSpan.FromSeconds(
                ((windowEnd - 1) * (double)StreamingWeSpeakerFbank.FrameShift + StreamingWeSpeakerFbank.FrameLength) / _config.SampleRate);
            return new SpeakerEventArgs(
                audioTime, best, bestScore, best is not null && bestScore > _threshold,
                scores ?? EmptyScores, embedding);
        }

        /// <summary>
        /// 前缀窗提前打分: 对 [流起点, prefixEnd) 的不足整窗前缀推理产出 IsEarlyExit=true 的临时结果。
        /// 得分为原始余弦 (不读写 EMA 平滑状态), 完整窗时间线保持与未启用时逐位一致。
        /// </summary>
        private SpeakerEventArgs? ProcessPrefixWindow(long prefixEnd)
        {
            if (OnResult is null)
                return null; // 无订阅者: 结果无人消费, 跳过推理

            int frames = (int)prefixEnd;
            int dim = _config.FeatureDim;

            if (_minWindowRms > 0f)
            {
                double sumSq = _fbank.SumSquaredRms(0, frames);
                if (!(Math.Sqrt(sumSq / frames) >= _minWindowRms))
                    return null; // 静音前缀 (或含 NaN 的坏窗): 跳过推理, 不产出结果
            }

            _fbank.CopyFramesTo(0, frames, _windowFeats);
            if (_config.ApplyCmn) CmnInPlace(_windowFeats, frames, dim);

            // 动态长度路径: 前缀帧数逐次变化, 走 Run 避免反复重建整窗的复用 IO 绑定
            float[] embedding = _runner.Run(_windowFeats, frames);
            if (!Normalize(embedding))
                return null;

            string? best = null;
            float bestScore = 0f;
            Dictionary<string, float>? scores = _profiles.Count == 0
                ? null
                : new Dictionary<string, float>(_profiles.Count, StringComparer.Ordinal);
            foreach (var (name, profile) in _profiles)
            {
                float raw = Math.Clamp((float)Dot(embedding, profile), -1f, 1f);
                scores![name] = raw;
                if (best is null || raw > bestScore)
                {
                    best = name;
                    bestScore = raw;
                }
            }

            var audioTime = TimeSpan.FromSeconds(
                ((prefixEnd - 1) * (double)StreamingWeSpeakerFbank.FrameShift + StreamingWeSpeakerFbank.FrameLength) / _config.SampleRate);
            return new SpeakerEventArgs(
                audioTime, best, bestScore, best is not null && bestScore > _threshold,
                scores ?? EmptyScores, embedding, isEarlyExit: true);
        }

        /// <summary>整段音频 → fbank (+整段 CMN) 特征, 行优先 [frames, dim]。</summary>
        private float[] ComputeOfflineFeatures(ReadOnlySpan<float> samples, out int frames)
        {
            if (samples.Length < StreamingWeSpeakerFbank.FrameLength)
                throw new ArgumentException(
                    $"Audio too short: {samples.Length} samples < one frame ({StreamingWeSpeakerFbank.FrameLength}).", nameof(samples));

            frames = SamplesToFrames(samples.Length);
            int dim = _config.FeatureDim;
            int featureCount = checked(frames * dim);
            float[] feats = ArrayPool<float>.Shared.Rent(featureCount);
            try
            {
                _fbank.ComputeFeatures(samples, feats.AsSpan(0, featureCount));
                if (_config.ApplyCmn) CmnInPlace(feats, frames, dim);
                return feats;
            }
            catch
            {
                ArrayPool<float>.Shared.Return(feats);
                throw;
            }
        }

        private StreamingWeSpeakerFbank CreateFbank() => new(
            _config.FeatureDim, _config.WindowType, _config.Preemphasis, _config.RemoveDcOffset, _config.WaveformScale,
            _windowFrames);

        private static int SamplesToFrames(int samples) =>
            samples < StreamingWeSpeakerFbank.FrameLength
                ? 0
                : (samples - StreamingWeSpeakerFbank.FrameLength) / StreamingWeSpeakerFbank.FrameShift + 1;

        private static int MillisecondsToSamples(int milliseconds, int sampleRate)
        {
            long samples = (long)sampleRate * milliseconds / 1000;
            if (samples <= 0 || samples > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(milliseconds), milliseconds, "Duration is not representable as an audio sample count.");
            return (int)samples;
        }

        private static int GetMinEnrollSamples(int minEnrollMs, string paramName)
        {
            if (minEnrollMs < 100)
                throw new ArgumentOutOfRangeException(paramName, minEnrollMs, "MinEnrollMs must be >= 100.");
            long sampleCount = (long)StreamingWeSpeakerFbank.SampleRate * minEnrollMs / 1000;
            if (sampleCount <= 0 || sampleCount > int.MaxValue)
                throw new ArgumentOutOfRangeException(
                    paramName, minEnrollMs,
                    "MinEnrollMs must be representable as an audio sample count.");
            return (int)sampleCount;
        }

        private static void CmnInPlace(float[] feats, int frames, int dim)
        {
            Span<float> means = dim <= 256 ? stackalloc float[dim] : new float[dim];
            for (int d = 0; d < dim; d++)
            {
                double sum = 0;
                for (int i = 0; i < frames; i++) sum += feats[i * dim + d];
                means[d] = (float)(sum / frames);
            }
            // 均值仍按原有逐维顺序计算；减均值改为逐帧连续访问，结果逐元素完全相同且更利于 CPU 缓存/向量化。
            for (int i = 0; i < frames; i++)
            {
                int offset = i * dim;
                for (int d = 0; d < dim; d++)
                    feats[offset + d] -= means[d];
            }
        }

        /// <summary>就地 L2 归一化。返回 false 表示向量非有限或全零 (不可用于余弦打分)。</summary>
        private static bool Normalize(float[] vector)
        {
            double sumSq = 0;
            for (int i = 0; i < vector.Length; i++) sumSq += (double)vector[i] * vector[i];
            if (!(sumSq > 0) || !double.IsFinite(sumSq)) return false;
            float inv = (float)(1.0 / Math.Sqrt(sumSq));
            for (int i = 0; i < vector.Length; i++) vector[i] *= inv;
            return true;
        }

        private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++) sum += (double)a[i] * b[i];
            return sum;
        }

        /// <summary>推理 + L2 归一化。特征含 NaN/Inf 时抛出 (离线路径的输入即错误)。</summary>
        private float[] RunNormalized(float[] feats, int frames)
        {
            float[] embedding = _runner.Run(feats, frames);
            if (!Normalize(embedding))
                throw new ArgumentException("Audio produced a non-finite embedding (NaN/Inf samples in input?).");
            return embedding;
        }

        private static string ResolveModelDir(WeSpeakerOptions options)
        {
            if (!string.IsNullOrWhiteSpace(options.ModelDir))
            {
                if (!Directory.Exists(options.ModelDir))
                    throw new DirectoryNotFoundException($"WeSpeaker model directory not found: {options.ModelDir}");
                return options.ModelDir;
            }
            string dir = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "speaker", "wespeaker");
            if (!Directory.Exists(dir))
                throw new DirectoryNotFoundException(
                    $"Built-in WeSpeaker model not found at '{dir}'. " +
                    "Ensure the model files are copied to the output directory, or set WeSpeakerOptions.ModelDir.");
            return dir;
        }

        private static void ValidateModelHash(string modelPath, string expectedSha256)
        {
            if (string.IsNullOrEmpty(expectedSha256)) return;
            using var stream = File.OpenRead(modelPath);
            string actual = Convert.ToHexString(SHA256.HashData(stream));
            if (!actual.Equals(expectedSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"WeSpeaker model SHA-256 mismatch for '{modelPath}'. Expected {expectedSha256}, got {actual}.");
        }
    }
}
