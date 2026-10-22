using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.DoubaoOnline.Session;
using VoiceBox.TTS.DoubaoOnline.Transport;

namespace VoiceBox.TTS.DoubaoOnline
{
    /// <summary>
    /// 基于火山引擎「豆包语音合成大模型 2.0 (seed-tts-2.0)」双向流式 WebSocket 接口的在线 TTS 服务。
    /// </summary>
    /// <remarks>
    /// 架构
    /// ----
    ///   * <see cref="SeedTtsConnectionWorker"/> — 每次合成一条 WebSocket，驱动
    ///     StartConnection → StartSession → TaskRequest(可多次) → FinishSession，接收 loop 解析下行事件。
    ///   * <see cref="SeedTtsSession"/> + <see cref="SeedTtsSessionManager"/> — 每会话状态、音频输出
    ///     channel、握手门。
    ///   * <see cref="Transport.IWebSocketChannel"/> — 可替换的传输抽象 (生产用 ClientWebSocket，测试用 fake)。
    ///
    /// 与本地 ONNX TTS 不同，这里没有共享推理闸门；并发能力受
    /// <see cref="DoubaoOnlineTTSOptions.MaxConcurrentSessions"/> 与云端配额约束。经 Core 契约输出
    /// <c>float[]</c> PCM，故内部固定请求 <c>pcm</c> 格式。
    /// </remarks>
    public sealed class DoubaoOnlineTTSService : IStreamTTSService
    {
        private readonly DoubaoOnlineTTSOptions _options;
        private readonly ILogger<DoubaoOnlineTTSService> _logger;
        private readonly IWebSocketChannelFactory _channelFactory;
        private readonly SeedTtsSessionManager _sessions = new();
        private readonly IReadOnlyList<TtsVoice> _voices;
        // 服务级关闭信号: Dispose 时取消, 让在途流式合成经链接 token 及时停止 (与本地 TTS 一致)。
        private readonly CancellationTokenSource _shutdownCts = new();

        private int _activeSessionCount;
        private int _disposed;

        public DoubaoOnlineTTSService(
            IOptions<DoubaoOnlineTTSOptions> options,
            ILogger<DoubaoOnlineTTSService>? logger = null)
            : this(options, channelFactory: null, logger) { }

        // 供测试注入 fake 传输工厂的构造。
        internal DoubaoOnlineTTSService(
            IOptions<DoubaoOnlineTTSOptions> options,
            IWebSocketChannelFactory? channelFactory,
            ILogger<DoubaoOnlineTTSService>? logger = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            _options = options.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<DoubaoOnlineTTSService>.Instance;
            _channelFactory = channelFactory ?? new ClientWebSocketChannelFactory();

            ValidateOptions(_options);
            _voices = BuildVoiceList(_options);
        }

        public int OutputSampleRate => _options.SampleRate;
        public int OutputChannels => _options.ChannelCount;

        public IReadOnlyList<TtsVoice> ListVoices() => _voices;

        public TtsVoice? GetDefaultVoice()
        {
            if (!string.IsNullOrWhiteSpace(_options.DefaultVoice))
            {
                return _voices.FirstOrDefault(v => string.Equals(v.Name, _options.DefaultVoice, StringComparison.OrdinalIgnoreCase))
                       ?? new TtsVoice { Name = _options.DefaultVoice! };
            }
            return _voices.Count > 0 ? _voices[0] : null;
        }

        /// <summary>整段合成：消费流式结果并拼成一个 <see cref="TtsResult"/>。</summary>
        public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ArgumentException.ThrowIfNullOrEmpty(text);

            string voice = ResolveVoice(options);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var buffer = new List<float>(_options.SampleRate); // 预留约 1s

            await foreach (var chunk in GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.Samples.Length > 0)
                    buffer.AddRange(chunk.Samples);
            }

            return new TtsResult
            {
                Samples = buffer.ToArray(),
                SampleRate = _options.SampleRate,
                Channels = _options.ChannelCount,
                Voice = voice,
                ElapsedSeconds = sw.Elapsed.TotalSeconds,
            };
        }

        /// <summary>单段文本流式合成。</summary>
        public IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            ArgumentException.ThrowIfNullOrEmpty(text);
            return StreamCoreAsync(Single(text), options, ct);

            static async IAsyncEnumerable<string> Single(string t)
            {
                yield return t;
                await Task.CompletedTask;
            }
        }

        /// <summary>
        /// 双向流式合成：文本分块流式输入 (例如来自 LLM 的 token 流)，音频流式输出。
        /// 每个非空文本块发送一次 TaskRequest；文本流结束后自动 FinishSession。
        /// </summary>
        public IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            IAsyncEnumerable<string> textStream, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            ArgumentNullException.ThrowIfNull(textStream);
            return StreamCoreAsync(textStream, options, ct);
        }

        private async IAsyncEnumerable<TtsAudioChunk> StreamCoreAsync(
            IAsyncEnumerable<string> textStream,
            TtsSynthesisOptions? options,
            [EnumeratorCancellation] CancellationToken ct)
        {
            EnsureNotDisposed();
            string voice = ResolveVoice(options);

            // 原子先占位, 失败回滚, 避免并发突破上限。
            int reserved = Interlocked.Increment(ref _activeSessionCount);
            if (reserved > _options.MaxConcurrentSessions)
            {
                Interlocked.Decrement(ref _activeSessionCount);
                throw new InvalidOperationException(
                    $"Reached the maximum concurrent session limit: {_options.MaxConcurrentSessions}.");
            }

            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;

            string sessionId = Guid.NewGuid().ToString("N");
            string requestId = Guid.NewGuid().ToString();
            string connectId = Guid.NewGuid().ToString();
            var session = new SeedTtsSession(sessionId, requestId, connectId, _options.SampleRate, _options.ChannelCount, _logger);

            SeedTtsConnectionWorker? worker = null;
            IWebSocketChannel? channel = null;
            Task? sender = null;

            // --- 建连 + 握手 (可能抛出; 此段无 yield) ---
            try
            {
                Uri uri = SeedTtsRequestBuilder.BuildUri(_options);
                var headers = SeedTtsRequestBuilder.BuildHeaders(_options, requestId, connectId);
                string startJson = SeedTtsRequestBuilder.BuildStartSessionJson(_options, voice, requestId);

                channel = _channelFactory.Create(uri, headers, _options.ConnectTimeoutSec);
                worker = new SeedTtsConnectionWorker(session, channel, _options, startJson, _logger);

                await worker.StartAsync(token).ConfigureAwait(false);
                _sessions.TryAdd(session, worker);
                _logger.LogDebug("SeedTTS session created: {SessionId} (logid={LogId})", sessionId, session.LogId);

                // 后台发送文本块并收尾。音频经会话 channel 与本方法的消费并行流出。
                sender = Task.Run(() => SendTextLoopAsync(worker, session, textStream, token), token);
            }
            catch (Exception ex)
            {
                _sessions.TryRemove(sessionId, out _, out _);
                Interlocked.Decrement(ref _activeSessionCount);
                if (worker is not null) { try { await worker.AbortAsync().ConfigureAwait(false); } catch { } }
                else { try { channel?.Dispose(); } catch { } }
                session.Dispose();
                linked.Dispose();
                if (ex is not OperationCanceledException)
                    _logger.LogError(ex, "Failed to start SeedTTS session.");
                throw;
            }

            // --- 消费音频 (在 try/finally 中 yield) ---
            try
            {
                await foreach (var chunk in ReadWithIdleTimeoutAsync(session, _options.SessionTimeoutSec, token).ConfigureAwait(false))
                    yield return chunk;
            }
            finally
            {
                _sessions.TryRemove(sessionId, out _, out _);

                bool finishedOk = session.SessionFinished.IsCompletedSuccessfully;
                if (finishedOk)
                {
                    if (sender is not null) { try { await sender.ConfigureAwait(false); } catch { } }
                    using var stopCts = CreateStopCts(ct);
                    if (worker is not null) { try { await worker.StopAsync(stopCts.Token).ConfigureAwait(false); } catch { } }
                }
                else
                {
                    // 提前 break / 取消 / 超时 / 错误: 取消并强制中止。
                    linked.Cancel();
                    if (sender is not null) { try { await sender.ConfigureAwait(false); } catch { } }
                    if (worker is not null) { try { await worker.AbortAsync().ConfigureAwait(false); } catch { } }
                }

                Interlocked.Decrement(ref _activeSessionCount);
                session.Dispose();
                linked.Dispose();
            }
        }

        private async Task SendTextLoopAsync(
            SeedTtsConnectionWorker worker, SeedTtsSession session,
            IAsyncEnumerable<string> textStream, CancellationToken token)
        {
            try
            {
                await foreach (var text in textStream.WithCancellation(token).ConfigureAwait(false))
                {
                    if (!string.IsNullOrEmpty(text))
                        await worker.SendTextAsync(text, token).ConfigureAwait(false);
                }
                await worker.FinishSessionAsync(token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { /* 正常取消 */ }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "SeedTTS text send loop faulted (session {SessionId}).", session.SessionId);
                session.TrySetError(new DoubaoOnlineTTSException("Text send loop faulted.", ex));
            }
        }

        /// <summary>
        /// 从会话音频 channel 读块，带空闲超时保护：若 <paramref name="timeoutSec"/> 内无任何新块
        /// 且会话未结束，抛 <see cref="TimeoutException"/>。0 表示不设超时。
        /// </summary>
        private static async IAsyncEnumerable<TtsAudioChunk> ReadWithIdleTimeoutAsync(
            SeedTtsSession session, int timeoutSec, [EnumeratorCancellation] CancellationToken ct)
        {
            var reader = session.AudioReader;
            while (await WaitReadableAsync(reader, session.SessionId, timeoutSec, ct).ConfigureAwait(false))
            {
                while (reader.TryRead(out var chunk))
                    yield return chunk;
            }
        }

        private static async Task<bool> WaitReadableAsync(
            ChannelReader<TtsAudioChunk> reader, string sessionId, int timeoutSec, CancellationToken ct)
        {
            if (timeoutSec <= 0)
                return await reader.WaitToReadAsync(ct).ConfigureAwait(false);

            using var timerCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            var waitTask = reader.WaitToReadAsync(ct).AsTask();
            var delayTask = Task.Delay(TimeSpan.FromSeconds(timeoutSec), timerCts.Token);
            var completed = await Task.WhenAny(waitTask, delayTask).ConfigureAwait(false);
            if (completed == delayTask)
            {
                // 超时: waitTask 被遗弃。若此刻 channel 恰以异常收尾 (服务端错误帧与超时竞争),
                // 该 task 会 faulted 且无人观测, GC 时触发 UnobservedTaskException。主动观测其异常。
                _ = waitTask.ContinueWith(
                    static t => _ = t.Exception,
                    CancellationToken.None,
                    TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
                throw new TimeoutException($"SeedTTS session {sessionId} idle for {timeoutSec}s waiting for audio.");
            }

            timerCts.Cancel(); // 取消未命中的 delay, 避免遗留计时器
            return await waitTask.ConfigureAwait(false); // channel 以异常收尾时在此抛出服务端错误
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

            var snapshot = _sessions.Snapshot();
            try { _shutdownCts.Cancel(); } catch { }

            var abortTasks = new List<Task>(snapshot.Count);
            foreach (var (session, w) in snapshot)
            {
                try { session.Cancel(); } catch { }
                abortTasks.Add(w.AbortAsync());
            }

            try { Task.WhenAll(abortTasks).Wait(TimeSpan.FromSeconds(_disposeDrainTimeoutSec)); }
            catch (Exception ex) { _logger.LogWarning(ex, "SeedTTS workers did not all drain before Dispose."); }

            foreach (var (session, _) in snapshot)
            {
                try { session.Dispose(); } catch { }
            }
            _sessions.Dispose();
            _shutdownCts.Dispose();
        }

        private const int _disposeDrainTimeoutSec = 10;

        private CancellationTokenSource CreateStopCts(CancellationToken ct)
        {
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
            int budget = _options.SessionTimeoutSec > 0 ? _options.SessionTimeoutSec : _disposeDrainTimeoutSec;
            linked.CancelAfter(TimeSpan.FromSeconds(budget));
            return linked;
        }

        private string ResolveVoice(TtsSynthesisOptions? options)
        {
            string? voice = !string.IsNullOrWhiteSpace(options?.Voice) ? options!.Voice : _options.DefaultVoice;
            if (string.IsNullOrWhiteSpace(voice) && _voices.Count > 0)
                voice = _voices[0].Name;
            if (string.IsNullOrWhiteSpace(voice))
                throw new InvalidOperationException(
                    "No voice specified. Set TtsSynthesisOptions.Voice, DoubaoOnlineTTSOptions.DefaultVoice, or KnownVoices.");
            return voice!;
        }

        private void EnsureNotDisposed()
            => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        private static IReadOnlyList<TtsVoice> BuildVoiceList(DoubaoOnlineTTSOptions o)
        {
            var list = new List<TtsVoice>();
            if (o.KnownVoices is { Count: > 0 })
            {
                foreach (var v in o.KnownVoices)
                    list.Add(new TtsVoice { Name = v.Name, Description = v.Description, Language = v.Language });
            }
            // 保证默认音色在列表中 (便于 Demo/上层展示)。
            if (!string.IsNullOrWhiteSpace(o.DefaultVoice) &&
                !list.Any(v => string.Equals(v.Name, o.DefaultVoice, StringComparison.OrdinalIgnoreCase)))
            {
                list.Insert(0, new TtsVoice { Name = o.DefaultVoice! });
            }
            return list;
        }

        private static void ValidateOptions(DoubaoOnlineTTSOptions o)
        {
            if (string.IsNullOrWhiteSpace(o.ApiKey))
                throw new ArgumentException(
                    "Authentication is not configured: set ApiKey.");
            if (string.IsNullOrWhiteSpace(o.ResourceId))
                throw new ArgumentException("DoubaoOnlineTTSOptions.ResourceId must be set.");
            if (string.IsNullOrWhiteSpace(o.Host))
                throw new ArgumentException("DoubaoOnlineTTSOptions.Host must be set.");
            if (o.SampleRate is not (8000 or 16000 or 22050 or 24000 or 32000 or 44100 or 48000))
                throw new ArgumentOutOfRangeException(nameof(o),
                    "SampleRate must be one of 8000/16000/22050/24000/32000/44100/48000.");
            if (o.ChannelCount != 1)
                throw new ArgumentOutOfRangeException(nameof(o), "Channels must be 1 (the interface outputs mono).");
            if (o.SpeechRate is < -50 or > 100)
                throw new ArgumentOutOfRangeException(nameof(o), "SpeechRate must be within [-50, 100].");
            if (o.LoudnessRate is < -50 or > 100)
                throw new ArgumentOutOfRangeException(nameof(o), "LoudnessRate must be within [-50, 100].");
            if (o.Pitch is < -12 or > 12)
                throw new ArgumentOutOfRangeException(nameof(o), "Pitch must be within [-12, 12].");
            if (o.MaxConcurrentSessions < 1)
                throw new ArgumentOutOfRangeException(nameof(o), "MaxConcurrentSessions must be ≥ 1.");
        }
    }
}
