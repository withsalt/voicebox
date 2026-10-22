using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OnnxGenAI.Internal;

namespace VoiceBox.LLM.OnnxGenAI
{
    /// <summary>
    /// 基于 onnxruntime-genai 的多模态 LLM 推理服务。
    ///
    /// 设计要点:
    /// <list type="bullet">
    ///   <item>构造期完成全部重资源加载 (Model / Tokenizer / Processor) 与可选预热; 之后调用零冷启动开销。</item>
    ///   <item>内部以 <see cref="SemaphoreSlim"/> 串行化 <see cref="Generator"/> 的创建与运行 — 同一 Model 上多并发会拖垮 KV-cache。</item>
    ///   <item>流式路径用 <see cref="Channel{T}"/> 解耦生产/消费, 与现有 <c>KokoroTTSService</c> 风格一致。</item>
    ///   <item><see cref="CancellationToken"/> 全程透传; 取消时已发出的块会被消费者读完, 然后通道收到 <see cref="OperationCanceledException"/>。</item>
    /// </list>
    /// </summary>
    public sealed class OnnxGenAILLMService : IConversationalLLMService
    {
        private readonly OnnxGenAILLMOptions _options;
        private readonly ILogger<OnnxGenAILLMService> _logger;
        private readonly GenAIInferenceSession _session;
        private readonly GenAIPromptTemplate _template;
        private readonly SemaphoreSlim _gate;
        private readonly CancellationTokenSource _shutdownCts = new();
        private int _disposed;

        public OnnxGenAILLMService(IOptions<OnnxGenAILLMOptions> options, ILogger<OnnxGenAILLMService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<OnnxGenAILLMService>.Instance;

            OnnxGenAILLMOptionsValidator.ThrowIfInvalid(_options);

            _session = new GenAIInferenceSession(_options);
            _template = new GenAIPromptTemplate(_options.ModelPath);
            _gate = new SemaphoreSlim(_options.MaxConcurrentRequests, _options.MaxConcurrentRequests);

            try
            {
                _session.Warmup();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Model warmup failed; first request may be slower.");
            }

            _logger.LogDebug(
                "OnnxGenAILLMService initialized: model={ModelPath}, ep={Provider}, multimodal={MultiModal}",
                _options.ModelPath, _options.ExecutionProvider, _session.Processor != null);
        }

        public async Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);

            var linked = GenAICancellation.LinkIfNeeded(ct, _shutdownCts.Token, out var token);
            var sw = Stopwatch.StartNew();
            var visibleSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            try
            {
                // 生成主循环是阻塞式 CPU 调用, 且这里的 onChunk 不会产生真正的 await 让步;
                // 直接 await GenerateAsync 会把整段推理压在调用线程上 (在 ASP.NET/UI 线程尤其糟)。
                // 放到线程池执行, 使 CompleteAsync 对调用方保持真正异步。
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

            var linked = GenAICancellation.CreateLinkedSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            var channel = GenAIGeneration.CreateChannel(_options.StreamChannelCapacity);

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
                // 消费方提前停止枚举时, 必须取消生产任务, 否则它可能继续占用 gate 或阻塞在 bounded channel 写入。
                linked.Cancel();
                try { await generationTask.ConfigureAwait(false); }
                catch { /* 错误已通过 channel 传播给消费方; 这里只保证生产任务收尾。 */ }
                linked.Dispose();
            }
        }

        /// <summary>
        /// 无状态的完整生成: 每次新建 Generator, 不跨调用复用 KV-cache。
        /// 这是 <see cref="CompleteAsync"/> 与 <see cref="StreamAsync"/> 共用的核心实现;
        /// 内核逻辑抽在 <see cref="GenAIGeneration"/> 中, 与会话路径共享。
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
                return await GenAIGeneration.RunStatelessAsync(
                    _session, _options, _template, messages, options, onChunk, ct).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <summary>
        /// 创建一段有状态对话。当 <see cref="OnnxGenAILLMOptions.EnableSessionCache"/> 开启且
        /// <see cref="OnnxGenAILLMOptions.MaxConcurrentRequests"/> == 1 时, 会话跨轮复用 KV-cache;
        /// 否则会话退化为每轮新建 Generator (行为等价于无状态路径)。
        /// </summary>
        public ILLMConversation CreateConversation()
        {
            EnsureNotDisposed();
            bool reuse = _options.EnableSessionCache && _options.MaxConcurrentRequests == 1;
            return new OnnxGenAILLMConversation(
                _session, _template, _options, _gate, _shutdownCts.Token, reuse, _logger);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try { _shutdownCts.Cancel(); } catch { }
            try
            {
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
    }
}
