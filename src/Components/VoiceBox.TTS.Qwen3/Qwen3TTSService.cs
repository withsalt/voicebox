using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.Qwen3.Internal;

namespace VoiceBox.TTS.Qwen3
{
    /// <summary>
    /// Qwen3-TTS (12Hz CustomVoice) ONNX 推理服务。两级自回归 (talker + code_predictor) + 神经 codec 解码,
    /// 支持预置音色、可选风格指令 (1.7B) 与低延迟流式。同时实现 <see cref="ITTSService"/> 与 <see cref="IStreamTTSService"/>。
    /// </summary>
    public sealed class Qwen3TTSService : IStreamTTSService
    {
        private readonly Qwen3TTSOptions _options;
        private readonly ILogger<Qwen3TTSService>? _logger;
        private readonly SemaphoreSlim _gate;
        private readonly int _maxConcurrent;
        private readonly object _initLock = new();
        // 服务级关闭信号: 链入每个请求的 token, Dispose 时先 Cancel 让在途合成尽快退出 (释放 gate),
        // 再 Wait + Dispose 原生 session, 避免超时后强行释放造成 use-after-free。
        private readonly CancellationTokenSource _shutdownCts = new();
        private volatile Qwen3TtsPipeline? _pipeline;
        private volatile bool _disposed;

        public Qwen3TTSService(IOptions<Qwen3TTSOptions> options, ILogger<Qwen3TTSService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger;
            _maxConcurrent = 1;
            _gate = new SemaphoreSlim(_maxConcurrent);
        }

        public int OutputSampleRate => EnsurePipeline().SampleRate;
        public int OutputChannels => EnsurePipeline().Channels;

        public IReadOnlyList<TtsVoice> ListVoices() => EnsurePipeline().ListVoices();

        public TtsVoice? GetDefaultVoice()
        {
            var pipeline = EnsurePipeline();
            return pipeline.ListVoices()
                .FirstOrDefault(v => string.Equals(v.Name, pipeline.DefaultVoiceName, StringComparison.OrdinalIgnoreCase));
        }

        private Qwen3TtsPipeline EnsurePipeline()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_pipeline is not null) return _pipeline;
            lock (_initLock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_pipeline is null)
                {
                    _logger?.LogInformation("正在加载 Qwen3-TTS 模型: {Path}", _options.ModelPath);
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    _pipeline = new Qwen3TtsPipeline(_options, _logger);
                    _logger?.LogInformation("Qwen3-TTS 模型加载完成, 用时 {Ms} ms, 默认音色={Voice}",
                        sw.ElapsedMilliseconds, _pipeline.DefaultVoiceName);
                }
            }
            return _pipeline;
        }

        public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var pipeline = EnsurePipeline();
            // 调用方 ct 与服务级 shutdown 链接; using 保证退出时解除在 _shutdownCts 上的注册。
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            await _gate.WaitAsync(token).ConfigureAwait(false);
            try
            {
                return await Task.Run(() =>
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    float[] samples = pipeline.SynthesizeWhole(text, options, out string voice, token);
                    return new TtsResult
                    {
                        Samples = samples,
                        SampleRate = pipeline.SampleRate,
                        Channels = pipeline.Channels,
                        Voice = voice,
                        ElapsedSeconds = sw.Elapsed.TotalSeconds,
                    };
                }, token).ConfigureAwait(false);
            }
            finally { _gate.Release(); }
        }

        public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            string text, TtsSynthesisOptions? options = null, [EnumeratorCancellation] CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var pipeline = EnsurePipeline();
            int sr = pipeline.SampleRate, ch = pipeline.Channels;

            // 调用方 ct 与服务级 shutdown 链接: Dispose 取消 _shutdownCts 时在途合成及时停止。
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;

            var channel = Channel.CreateUnbounded<TtsAudioChunk>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
            });

            var producer = Task.Run(async () =>
            {
                Exception? error = null;
                bool acquired = false;
                try
                {
                    await _gate.WaitAsync(token).ConfigureAwait(false);
                    acquired = true;
                    long emitted = 0;
                    foreach (float[] chunk in pipeline.SynthesizeStream(text, options, token))
                    {
                        token.ThrowIfCancellationRequested();
                        if (chunk.Length == 0) continue;
                        emitted += chunk.Length;
                        await channel.Writer.WriteAsync(MakeChunk(chunk, sr, ch, emitted, false), token).ConfigureAwait(false);
                    }
                    await channel.Writer.WriteAsync(MakeChunk([], sr, ch, emitted, true), token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) { /* 正常取消 */ }
                catch (Exception ex) { error = ex; _logger?.LogError(ex, "Qwen3-TTS 流式合成失败。"); }
                finally
                {
                    if (acquired) _gate.Release();
                    channel.Writer.TryComplete(error);
                }
            }, token);

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(token).ConfigureAwait(false))
                    yield return chunk;
            }
            finally
            {
                // 消费结束 (正常完成 / 异常 / 提前 break): 取消并 join 生产者, 再释放 linked。
                // 必须释放 linked 才能解除其在 _shutdownCts 上的注册, 否则长跑会逐请求累积注册。
                linked.Cancel();
                try { await producer.ConfigureAwait(false); }
                catch { /* 生产者错误已经过 channel 传给消费方; 这里只保证 join, 不掩盖原异常 */ }
                linked.Dispose();
            }
        }

        private static TtsAudioChunk MakeChunk(float[] samples, int sr, int ch, long emittedSamples, bool isFinal) => new()
        {
            Samples = samples,
            SampleRate = sr,
            Channels = ch,
            EmittedSeconds = (double)emittedSamples / sr,
            IsFinal = isFinal,
        };

        public void Dispose()
        {
            Qwen3TtsPipeline? pipeline;
            lock (_initLock)
            {
                if (_disposed) return;
                _disposed = true;
                pipeline = _pipeline;
            }
            // 先发关闭信号: 在途合成经链接的 token 观察到取消, 在一帧内退出并释放 gate,
            // 使下面的 Wait 快速返回, 不再出现 "30s 超时后强行 Dispose 正在使用的 session"。
            try { _shutdownCts.Cancel(); } catch { /* 忽略 */ }
            try
            {
                for (int i = 0; i < _maxConcurrent; i++)
                    _gate.Wait(TimeSpan.FromSeconds(30));
            }
            catch { /* 忽略, 继续清理 */ }
            pipeline?.Dispose();
            _gate.Dispose();
            _shutdownCts.Dispose();
        }
    }
}
