using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.VoxCPM.Internal;

namespace VoiceBox.TTS.VoxCPM
{
    /// <summary>
    /// VoxCPM-0.5B (tokenizer-free TTS) ONNX 推理服务。自回归 LM + 局部流匹配扩散 + AudioVAE,
    /// 16kHz 输出, 每步产出 80ms —— 天然低延迟流式, 支持音色克隆 (prompt 音频+文本)。
    /// 同时实现 <see cref="ITTSService"/> 与 <see cref="IStreamTTSService"/>。
    /// </summary>
    public sealed class VoxCPMTTSService : IStreamTTSService
    {
        private readonly VoxCPMTTSOptions _options;
        private readonly ILogger<VoxCPMTTSService>? _logger;
        private readonly SemaphoreSlim _gate;
        private readonly int _maxConcurrent;
        private readonly object _initLock = new();
        // 服务级关闭信号: 链入每个请求的 token, Dispose 时先 Cancel 让在途合成尽快退出 (释放 gate),
        // 再 Wait + Dispose 原生 session, 避免超时后强行释放造成 use-after-free。
        private readonly CancellationTokenSource _shutdownCts = new();
        private volatile VoxCpmPipeline? _pipeline;
        private volatile bool _disposed;

        public VoxCPMTTSService(IOptions<VoxCPMTTSOptions> options, ILogger<VoxCPMTTSService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger;
            _maxConcurrent = 1;   // 有状态 KV cache, 单并发; 排队由 gate 完成
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

        private VoxCpmPipeline EnsurePipeline()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_pipeline is not null) return _pipeline;
            lock (_initLock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_pipeline is null)
                {
                    _logger?.LogInformation("正在加载 VoxCPM 模型: {Path}", _options.ModelPath);
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    _pipeline = new VoxCpmPipeline(_options, _logger);
                    _logger?.LogInformation("VoxCPM 模型加载完成, 用时 {Ms} ms, 默认音色={Voice}",
                        sw.ElapsedMilliseconds, _pipeline.DefaultVoiceName);
                }
            }
            return _pipeline;
        }

        public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var pipeline = EnsurePipeline();
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
                    await channel.Writer.WriteAsync(MakeChunk(Array.Empty<float>(), sr, ch, emitted, true), token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) { /* 正常取消 */ }
                catch (Exception ex) { error = ex; _logger?.LogError(ex, "VoxCPM 流式合成失败。"); }
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
                // 消费结束 (正常完成 / 异常 / 提前 break): 取消并 join 生产者, 再释放 linked,
                // 解除其在 _shutdownCts 上的注册 (否则长跑逐请求累积注册)。
                linked.Cancel();
                try { await producer.ConfigureAwait(false); }
                catch { /* 生产者错误已经过 channel 传给消费方 */ }
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
            VoxCpmPipeline? pipeline;
            lock (_initLock)
            {
                if (_disposed) return;
                _disposed = true;
                pipeline = _pipeline;
            }
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
