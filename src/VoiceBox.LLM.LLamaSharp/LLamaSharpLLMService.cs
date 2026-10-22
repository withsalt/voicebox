using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LLama.Native;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.LLamaSharp.Internal;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// 基于 LLamaSharp (llama.cpp) 的通用 GGUF 模型推理服务。
    ///
    /// 设计要点:
    /// <list type="bullet">
    ///   <item>构造期完成全部重资源加载 (Weights / Context / Profile) 与可选预热; 之后调用零冷启动开销。</item>
    ///   <item>内部以 <see cref="SemaphoreSlim"/> 串行化 llama.cpp context 的使用 —— 单 context 不支持并发访问。</item>
    ///   <item>流式路径用 <see cref="System.Threading.Channels.Channel{T}"/> 解耦生产/消费, 与 OnnxGenAI 服务风格一致。</item>
    ///   <item><see cref="CancellationToken"/> 全程透传; 取消时已发出的块会被消费者读完, 然后通道收到 <see cref="OperationCanceledException"/>。</item>
    ///   <item>llama.cpp 推理是 CPU 密集型同步调用; CompleteAsync/StreamAsync 均通过 <c>Task.Run</c> 放到线程池, 保持调用方的真正异步。</item>
    /// </list>
    /// </summary>
    public sealed class LLamaSharpLLMService : IConversationalLLMService, IToolCallingStreamLLMService
    {
        private static readonly object NativeConfigGate = new();

        private readonly LLamaSharpLLMOptions _options;
        private readonly ILogger<LLamaSharpLLMService> _logger;
        private readonly LlamaModelSession _session;
        private readonly SemaphoreSlim _gate;
        private readonly CancellationTokenSource _shutdownCts = new();
        private int _disposed;

        /// <summary>当前模型按 GGUF 元数据解析出的 profile 名称，供诊断与状态显示。</summary>
        public string ModelProfileName => _session.Profile.Name;

        public LLamaSharpLLMService(
            IOptions<LLamaSharpLLMOptions> options,
            ILogger<LLamaSharpLLMService>? logger = null,
            IEnumerable<ILlamaModelProfile>? modelProfiles = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<LLamaSharpLLMService>.Instance;

            LLamaSharpLLMOptionsValidator.ThrowIfInvalid(_options);

            // llama.cpp 默认直接把日志写到 stderr，不经过 Microsoft.Extensions.Logging。
            // 首次 native API 调用前接管日志，并固定丢弃原生 Debug/Trace。
            ConfigureNativeLogging(_logger);

            _session = new LlamaModelSession(_options, modelProfiles);
            _gate = new SemaphoreSlim(_options.MaxConcurrentRequests, _options.MaxConcurrentRequests);

            if (_options.WarmupOnLoad)
            {
                try
                {
                    _session.Warmup();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "LLamaSharp model warmup failed; first request may be slower.");
                }
            }

            _logger.LogDebug(
                "LLamaSharpLLMService initialized: model={ModelPath}, profile={Profile}, ctx={CtxSize}",
                _options.ModelPath, _session.Profile.Name, _session.SharedContext.ContextSize);
        }

        public async Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);

            var linked = LlamaCancellation.LinkIfNeeded(ct, _shutdownCts.Token, out var token);
            var sw = Stopwatch.StartNew();
            var visibleSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            try
            {
                // llama.cpp 推理是阻塞式 CPU 调用; Task.Run 让 CompleteAsync 对调用方保持真正异步。
                GenerationStats stats = await Task.Run(
                    () => GenerateAsync(
                        messages,
                        options,
                        onChunk: chunk =>
                        {
                            if (chunk.Kind == LlmChunkKind.Text) visibleSb.Append(chunk.Text);
                            else thinkingSb.Append(chunk.Text);
                            return ValueTask.CompletedTask;
                        },
                        token),
                    token).ConfigureAwait(false);

                sw.Stop();
                return new LlmCompletionResult
                {
                    Text = visibleSb.ToString(),
                    Thinking = thinkingSb.ToString(),
                    TotalTokens = stats.TotalTokens,
                    PromptTokens = stats.TotalTokens - stats.NewTokens,
                    CompletionTokens = stats.NewTokens,
                    ElapsedSeconds = sw.Elapsed.TotalSeconds,
                };
            }
            finally
            {
                linked?.Dispose();
            }
        }

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);

            var linked = LlamaCancellation.CreateLinkedSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            var channel = LlamaGeneration.CreateChannel(_options.StreamChannelCapacity);

            var generationTask = Task.Run(async () =>
            {
                Exception? failure = null;
                try
                {
                    await GenerateAsync(
                        messages,
                        options,
                        onChunk: chunk => channel.Writer.WriteAsync(chunk, token),
                        token).ConfigureAwait(false);
                }
                catch (OperationCanceledException ex)
                {
                    failure = token.IsCancellationRequested ? new OperationCanceledException(token) : ex;
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
                finally
                {
                    channel.Writer.TryComplete(failure);
                }
            });

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(token).ConfigureAwait(false))
                {
                    yield return chunk;
                }

                await generationTask.ConfigureAwait(false);
            }
            finally
            {
                linked.Cancel();
                try { await generationTask.ConfigureAwait(false); }
                catch { /* 错误已通过 channel 传播给消费方; 这里只保证生产任务收尾。 */ }
                linked.Dispose();
            }
        }

        public async IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
            LlmToolTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(request);
            EnsureNotDisposed();
            ValidateMessages(request.Messages);

            LlamaToolCallCatalog catalog = LlamaToolCallCatalog.Create(request.Tools);
            IReadOnlyList<ChatMessage> promptMessages = LlamaToolPromptBuilder.AddToolInstructions(
                request.Messages,
                catalog);
            var linked = LlamaCancellation.LinkIfNeeded(
                cancellationToken,
                _shutdownCts.Token,
                out CancellationToken token);
            var response = new StringBuilder();
            GenerationStats stats;
            try
            {
                await _gate.WaitAsync(token).ConfigureAwait(false);
                try
                {
                    stats = await Task.Run(
                        () => LlamaGeneration.RunStatelessAsync(
                            _session,
                            _options,
                            promptMessages,
                            request.Options,
                            onChunk: chunk =>
                            {
                                if (chunk.Kind == LlmChunkKind.Text)
                                    response.Append(chunk.Text);
                                return ValueTask.CompletedTask;
                            },
                            _logger,
                            token,
                            catalog.Grammar),
                        token).ConfigureAwait(false);
                }
                finally
                {
                    _gate.Release();
                }
            }
            finally
            {
                linked?.Dispose();
            }

            token.ThrowIfCancellationRequested();
            LlamaToolCallOutcome outcome = LlamaToolCallEnvelopeParser.Parse(response.ToString(), catalog);
            switch (outcome)
            {
                case LlamaToolCallOutcome.Final final:
                    yield return new LlmTurnTextDelta(final.Text);
                    break;
                case LlamaToolCallOutcome.ToolCalls calls:
                    string correlation = Guid.NewGuid().ToString("N");
                    for (int index = 0; index < calls.Calls.Count; index++)
                    {
                        LlamaToolCall call = calls.Calls[index];
                        yield return new LlmToolCallEvent(
                            $"llama_{correlation}_{index}",
                            call.ToolName,
                            call.ArgumentsJson);
                    }
                    break;
                default:
                    throw new InvalidOperationException("The structured response parser returned an unsupported result.");
            }

            yield return new LlmTurnCompletedEvent(
                PromptTokens: stats.TotalTokens - stats.NewTokens,
                CompletionTokens: stats.NewTokens,
                TotalTokens: stats.TotalTokens);
        }

        /// <summary>
        /// 无状态完整生成: 清空共享 context KV 后全量 prefill。
        /// 这是 <see cref="CompleteAsync"/> 与 <see cref="StreamAsync"/> 共用的核心实现。
        /// </summary>
        private async Task<GenerationStats> GenerateAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            Func<LlmTextChunk, ValueTask> onChunk,
            CancellationToken ct)
        {
            await _gate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                return await LlamaGeneration.RunStatelessAsync(
                    _session, _options, messages, options, onChunk, _logger, ct).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <summary>
        /// 创建一段有状态对话。当 <see cref="LLamaSharpLLMOptions.EnableSessionCache"/> 开启且
        /// <see cref="LLamaSharpLLMOptions.MaxConcurrentRequests"/> == 1 时, 会话跨轮复用 KV-cache;
        /// 否则会话退化为每轮清空后全量 prefill (行为等价于无状态路径)。
        /// 每个会话持有独立的 llama.cpp context (独占 KV-cache 内存), 用完必须 Dispose。
        /// </summary>
        public ILLMConversation CreateConversation()
        {
            EnsureNotDisposed();
            bool reuse = _options.EnableSessionCache && _options.MaxConcurrentRequests == 1;
            return new LLamaSharpLLMConversation(
                _session, _options, _gate, _shutdownCts.Token, reuse, _logger);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try { _shutdownCts.Cancel(); } catch { }
            try
            {
                // 等待所有正在进行的推理释放 gate slot 后再销毁资源, 最多等 30 s。
                for (int i = 0; i < _options.MaxConcurrentRequests; i++)
                    _gate.Wait(TimeSpan.FromSeconds(30));
            }
            catch { }
            _gate.Dispose();
            _session.Dispose();
            _shutdownCts.Dispose();
        }

        private void EnsureNotDisposed()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        }

        private static void ValidateMessages(IReadOnlyList<ChatMessage> messages)
        {
            ArgumentNullException.ThrowIfNull(messages);
            if (messages.Count == 0)
                throw new ArgumentException("Messages cannot be empty.", nameof(messages));
        }

        private static void ConfigureNativeLogging(ILogger<LLamaSharpLLMService> logger)
        {
            lock (NativeConfigGate)
            {
                // NativeLibraryConfig 是进程级全局配置；native library 加载后不能再修改。
                // 若宿主更早加载了 LLamaSharp，则保留宿主已有配置而不是抛异常。
                if (!NativeLibraryConfig.LLama.LibraryHasLoaded)
                    NativeLibraryConfig.LLama.WithLogCallback(new NativeLogFilterLogger(logger));
            }
        }

        /// <summary>
        /// llama.cpp 的 Debug/Trace 信息量极大且通常只用于底层诊断，组件固定丢弃；
        /// Information 及以上继续交给宿主 ILogger，由宿主决定最终是否输出。
        /// </summary>
        private sealed class NativeLogFilterLogger(ILogger inner) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull =>
                inner.BeginScope(state);

            public bool IsEnabled(LogLevel logLevel)
            {
                if (logLevel is LogLevel.Trace or LogLevel.Debug or LogLevel.None)
                    return false;
                return inner.IsEnabled(logLevel);
            }

            public void Log<TState>(
                LogLevel logLevel,
                EventId eventId,
                TState state,
                Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                if (IsEnabled(logLevel))
                    inner.Log(logLevel, eventId, state, exception, formatter);
            }
        }
    }
}
