using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.MossNano.Internal;

namespace VoiceBox.TTS.MossNano
{
    /// <summary>
    /// MOSS-TTS-Nano 服务实现 (ONNX Runtime)。同时实现 <see cref="ITTSService"/> (整段合成)
    /// 与 <see cref="IStreamTTSService"/> (流式合成)。
    /// </summary>
    public sealed class MossNanoTTSService : IStreamTTSService
    {
        private readonly MossNanoTTSOptions _options;
        private readonly ILogger<MossNanoTTSService> _logger;
        private readonly MossNanoPipeline _pipeline;
        private readonly SemaphoreSlim _gate;
        private readonly string _defaultVoice;
        private readonly IReadOnlyList<TtsVoice> _voices;
        private readonly Dictionary<string, TtsVoice> _voiceLookup;
        private readonly Task? _warmupTask;
        // 服务级关闭信号: Dispose 时取消, 让在途流式合成经链接 token 及时停止 (与 Qwen3TTSService 一致)。
        private readonly CancellationTokenSource _shutdownCts = new();
        private int _disposed;

        public MossNanoTTSService(IOptions<MossNanoTTSOptions> options, ILogger<MossNanoTTSService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<MossNanoTTSService>.Instance;

            ValidateOptions(_options);

            _pipeline = new MossNanoPipeline(_options.ModelPath, _options, _logger);

            _voices = BuildVoiceList(_pipeline.BuiltinVoices);
            if (_voices.Count == 0)
                throw new InvalidDataException("manifest.builtin_voices 为空, 无可用音色。");

            _voiceLookup = new Dictionary<string, TtsVoice>(_voices.Count, StringComparer.OrdinalIgnoreCase);
            // 与原 LINQ FirstOrDefault 行为一致: 重名时保留第一个出现的, 用 TryAdd 而非索引赋值。
            foreach (var v in _voices) _voiceLookup.TryAdd(v.Name, v);

            _defaultVoice = ResolveDefaultVoice(_options.DefaultVoice, _voices);
            _gate = new SemaphoreSlim(1, 1);
            var warmupToken = _shutdownCts.Token;
            _warmupTask = Task.Run(() =>
            {
                bool acquired = false;
                try
                {
                    _gate.Wait(warmupToken);
                    acquired = true;
                    _pipeline.Warmup(warmupToken);
                }
                catch (OperationCanceledException) when (warmupToken.IsCancellationRequested)
                {
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "MOSS-TTS-Nano 预热失败, 将在首次请求时加载模型。");
                }
                finally
                {
                    if (acquired)
                    {
                        try { _gate.Release(); }
                        catch (ObjectDisposedException) { }
                    }
                }
            });

            _logger.LogInformation(
                "MOSS-TTS-Nano 已初始化: ModelPath={ModelPath} 音色数={VoiceCount} 默认音色={Voice} EP={EP}",
                _options.ModelPath, _voices.Count, _defaultVoice, _options.ExecutionProvider);
        }

        public int OutputSampleRate => _pipeline.SampleRate;

        public int OutputChannels => _pipeline.Channels;

        public IReadOnlyList<TtsVoice> ListVoices()
        {
            EnsureNotDisposed();
            return _voices;
        }

        public TtsVoice? GetDefaultVoice()
        {
            EnsureNotDisposed();
            return _voiceLookup.TryGetValue(_defaultVoice, out var voice) ? voice : null;
        }

        public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ArgumentException.ThrowIfNullOrEmpty(text);

            string voice = ResolveVoice(options?.Voice);
            double startTs = NowSeconds();

            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            await WaitForWarmupAsync(token).ConfigureAwait(false);
            await _gate.WaitAsync(token).ConfigureAwait(false);
            float[] waveform;
            try
            {
                waveform = await Task.Run(() =>
                {
                    var result = _pipeline.Synthesize(
                        text,
                        voice,
                        promptAudioPath: null,
                        sampleMode: ToInternalSampleMode(_options.SampleMode),
                        doSample: options?.DoSample ?? _options.DoSample,
                        streaming: false,
                        maxNewFrames: options?.MaxNewFrames ?? _options.MaxNewFrames,
                        voiceCloneMaxTextTokens: _options.VoiceCloneMaxTextTokens,
                        seed: options?.Seed ?? _options.DefaultSeed,
                        textTemperature: options?.TextTemperature ?? _options.TextTemperature,
                        textTopK: options?.TextTopK ?? _options.TextTopK,
                        textTopP: options?.TextTopP ?? _options.TextTopP,
                        audioTemperature: options?.AudioTemperature ?? _options.AudioTemperature,
                        audioTopK: options?.AudioTopK ?? _options.AudioTopK,
                        audioTopP: options?.AudioTopP ?? _options.AudioTopP,
                        audioRepetitionPenalty: options?.AudioRepetitionPenalty ?? _options.AudioRepetitionPenalty,
                        ct: token);
                    return result.InterleavedWaveform;
                }, token).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }

            return new TtsResult
            {
                Samples = waveform,
                SampleRate = _pipeline.SampleRate,
                Channels = _pipeline.Channels,
                Voice = voice,
                ElapsedSeconds = NowSeconds() - startTs,
            };
        }

        public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            string text,
            TtsSynthesisOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ArgumentException.ThrowIfNullOrEmpty(text);

            string voice = ResolveVoice(options?.Voice);
            int sampleRate = _pipeline.SampleRate;
            int channels = _pipeline.Channels;

            // 调用方 ct 与服务级 shutdown 链接: Dispose 取消 _shutdownCts 时在途合成及时停止。
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            await WaitForWarmupAsync(token).ConfigureAwait(false);

            var channel = Channel.CreateUnbounded<TtsAudioChunk>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
                AllowSynchronousContinuations = false,
            });

            var generationTask = Task.Run(() =>
            {
                Exception? failure = null;
                int emittedSamples = 0;
                try
                {
                    _gate.Wait(token);
                    try
                    {
                        _pipeline.SynthesizeStreaming(
                            text,
                            voice,
                            promptAudioPath: null,
                            sampleMode: ToInternalSampleMode(_options.SampleMode),
                            doSample: options?.DoSample ?? _options.DoSample,
                            maxNewFrames: options?.MaxNewFrames ?? _options.MaxNewFrames,
                            voiceCloneMaxTextTokens: _options.VoiceCloneMaxTextTokens,
                            streamingChunkFrames: _options.StreamingChunkFrames,
                            seed: options?.Seed ?? _options.DefaultSeed,
                            textTemperature: options?.TextTemperature ?? _options.TextTemperature,
                            textTopK: options?.TextTopK ?? _options.TextTopK,
                            textTopP: options?.TextTopP ?? _options.TextTopP,
                            audioTemperature: options?.AudioTemperature ?? _options.AudioTemperature,
                            audioTopK: options?.AudioTopK ?? _options.AudioTopK,
                            audioTopP: options?.AudioTopP ?? _options.AudioTopP,
                            audioRepetitionPenalty: options?.AudioRepetitionPenalty ?? _options.AudioRepetitionPenalty,
                            onChunk: (samples, isFinal) =>
                            {
                                int frameSamples = channels > 0 ? samples.Length / channels : samples.Length;
                                if (samples.Length > 0) emittedSamples += frameSamples;
                                var chunk = new TtsAudioChunk
                                {
                                    Samples = samples,
                                    SampleRate = sampleRate,
                                    Channels = channels,
                                    EmittedSeconds = (double)emittedSamples / Math.Max(1, sampleRate),
                                    IsFinal = isFinal,
                                };
                                if (!channel.Writer.TryWrite(chunk))
                                    throw new OperationCanceledException();
                            },
                            ct: token);
                    }
                    finally { _gate.Release(); }
                }
                catch (Exception ex)
                {
                    failure = ex;
                    if (ex is not OperationCanceledException)
                        _logger.LogError(ex, "MOSS-TTS-Nano 流式合成失败。");
                }
                finally
                {
                    channel.Writer.TryComplete(failure);
                }
            }, token);

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(token).ConfigureAwait(false))
                {
                    yield return chunk;
                }
            }
            finally
            {
                // 消费结束 (正常完成 / 异常 / 提前 break / barge-in): 取消并 join 生产者, 再释放 linked。
                // 不 join 会让生产者成为孤儿任务 (终态异常无人观测); 释放 linked 解除其在 _shutdownCts
                // 上的注册, 否则长跑会逐请求累积注册。
                linked.Cancel();
                try { await generationTask.ConfigureAwait(false); }
                catch { /* 生产者错误已经过 channel 传给消费方; 这里只保证 join, 不掩盖原异常 */ }
                linked.Dispose();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // 先发关闭信号, 让在途流式合成经链接 token 尽快退出并释放 gate。
            try { _shutdownCts.Cancel(); } catch { /* ignore */ }
            bool warmupCompleted = true;
            try
            {
                warmupCompleted = _warmupTask?.Wait(TimeSpan.FromSeconds(30)) ?? true;
            }
            catch { /* ignore */ }
            try
            {
                _gate.Wait(TimeSpan.FromSeconds(30));
            }
            catch { /* ignore, continue cleanup */ }
            try { _gate.Dispose(); } catch { /* ignore */ }
            if (warmupCompleted)
            {
                DisposePipeline();
            }
            else
            {
                _logger.LogWarning("MOSS-TTS-Nano 预热任务未在关闭等待窗口内结束, 将延后释放流水线以避免并发释放 ONNX session。");
                _ = _warmupTask!.ContinueWith(
                    static (_, state) => ((MossNanoTTSService)state!).DisposePipeline(),
                    this,
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }
            try { _shutdownCts.Dispose(); } catch { /* ignore */ }
        }

        // ---------------- Helpers ----------------

        private string ResolveVoice(string? requested)
        {
            if (string.IsNullOrWhiteSpace(requested)) return _defaultVoice;
            if (_voiceLookup.TryGetValue(requested, out var matched)) return matched.Name;
            throw new ArgumentException($"未知音色: {requested}", nameof(requested));
        }

        private static MossNanoSampleMode? ToInternalSampleMode(MossNanoSampleMode? m) => m;

        private async Task WaitForWarmupAsync(CancellationToken ct)
        {
            if (_warmupTask is { IsCompleted: false } warmup)
                await warmup.WaitAsync(ct).ConfigureAwait(false);
        }

        private void DisposePipeline()
        {
            try { _pipeline.Dispose(); }
            catch (Exception ex) { _logger.LogWarning(ex, "释放 MOSS-TTS-Nano 流水线失败。"); }
        }

        private static IReadOnlyList<TtsVoice> BuildVoiceList(IReadOnlyList<ManifestBuiltinVoice> voices)
        {
            var list = new List<TtsVoice>(voices.Count);
            foreach (var v in voices)
            {
                list.Add(new TtsVoice
                {
                    Name = v.Voice,
                    Description = v.DisplayName,
                    Language = v.Group,
                });
            }
            return list;
        }

        private static string ResolveDefaultVoice(string? configured, IReadOnlyList<TtsVoice> voices)
        {
            if (!string.IsNullOrWhiteSpace(configured))
            {
                var hit = voices.FirstOrDefault(v => string.Equals(v.Name, configured, StringComparison.OrdinalIgnoreCase));
                if (hit is not null) return hit.Name;
                throw new ArgumentException($"配置的默认音色不存在: {configured}");
            }
            return voices[0].Name;
        }

        private static void ValidateOptions(MossNanoTTSOptions o)
        {
            if (string.IsNullOrWhiteSpace(o.ModelPath))
                throw new ArgumentException($"{nameof(MossNanoTTSOptions)}.{nameof(MossNanoTTSOptions.ModelPath)} 不能为空。");
            if (!Directory.Exists(o.ModelPath))
                throw new DirectoryNotFoundException($"模型目录不存在: {o.ModelPath}");
            if (o.VoiceCloneMaxTextTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(MossNanoTTSOptions.VoiceCloneMaxTextTokens), "必须 ≥ 1。");
            if (o.StreamingChunkFrames < 1)
                throw new ArgumentOutOfRangeException(nameof(MossNanoTTSOptions.StreamingChunkFrames), "必须 ≥ 1。");
        }

        private static double NowSeconds() => Environment.TickCount64 / 1000.0;

        private void EnsureNotDisposed()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        }
    }
}
