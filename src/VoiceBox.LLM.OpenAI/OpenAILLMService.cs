using System;
using System.ClientModel;
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
using OpenAI;
using OpenAI.Chat;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI.Internal;
using OAIChatMessage = OpenAI.Chat.ChatMessage;
using ChatMessage = VoiceBox.Core.Services.LLMService.Models.ChatMessage;

namespace VoiceBox.LLM.OpenAI
{
    /// <summary>
    /// 基于官方 <a href="https://github.com/openai/openai-dotnet">OpenAI .NET SDK</a> 的远程 LLM 服务,
    /// 适配任意 OpenAI 兼容端点 (官方 OpenAI / Azure / 火山方舟 Ark / DeepSeek 等)。
    ///
    /// 设计要点:
    /// <list type="bullet">
    ///   <item>重资源 (<see cref="ChatClient"/> 及其底层 HTTP 管线/连接池) 在构造期一次性建好,
    ///         进程内复用同一实例, 避免每次请求重建连接。</item>
    ///   <item>用 <see cref="SemaphoreSlim"/> 限制客户端侧并发上限, 作为一道反压闸门, 降低触发
    ///         上游 429 限流的概率。</item>
    ///   <item><see cref="StreamAsync"/> 用 <see cref="Channel{T}"/> 解耦生产/消费, 与
    ///         其他 LLM 服务保持一致的流式风格;
    ///         <see cref="CompleteAsync"/> 在内部消费同一条流。</item>
    ///   <item>推理模型的 <c>reasoning_content</c> 通过 <see cref="LlmChunkKind.Thinking"/> 单独路由,
    ///         不混入 <see cref="LlmChunkKind.Text"/>。</item>
    /// </list>
    /// </summary>
    public sealed class OpenAILLMService : IToolCallingStreamLLMService
    {
        private readonly OpenAILLMOptions _options;
        private readonly ILogger<OpenAILLMService> _logger;
        private readonly ChatClient _client;
        private readonly SemaphoreSlim _gate;
        private readonly CancellationTokenSource _shutdownCts = new();
        private int _disposed;

        public OpenAILLMService(IOptions<OpenAILLMOptions> options, ILogger<OpenAILLMService>? logger = null)
            : this(options, chatClient: null, logger)
        {
        }

        /// <summary>
        /// 允许调用方传入已构造好的 <see cref="ChatClient"/> (例如带自定义传输/重试策略, 或用于单测)。
        /// 传 <c>null</c> 时由本服务依据 <see cref="OpenAILLMOptions"/> 自行构造。
        /// </summary>
        public OpenAILLMService(
            IOptions<OpenAILLMOptions> options,
            ChatClient? chatClient,
            ILogger<OpenAILLMService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<OpenAILLMService>.Instance;

            ValidateOptions(_options);

            _client = chatClient ?? CreateChatClient(_options);
            _gate = new SemaphoreSlim(_options.MaxConcurrentRequests, _options.MaxConcurrentRequests);

            _logger.LogDebug(
                "OpenAILLMService initialized: endpoint={Endpoint}, model={Model}, maxConcurrent={Concurrent}",
                _options.Endpoint, _options.Model, _options.MaxConcurrentRequests);
        }

        private static ChatClient CreateChatClient(OpenAILLMOptions options)
        {
            var clientOptions = new OpenAIClientOptions
            {
                Endpoint = options.Endpoint,
                NetworkTimeout = options.HttpTimeout,
            };
            var credential = new ApiKeyCredential(options.ApiKey);
            return new ChatClient(options.Model, credential, clientOptions);
        }

        public async Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);

            var sw = Stopwatch.StartNew();
            var visibleSb = new StringBuilder();
            var thinkingSb = new StringBuilder();

            await foreach (var chunk in StreamInternalAsync(messages, options, ct).ConfigureAwait(false))
            {
                if (chunk.Kind == LlmChunkKind.Text) visibleSb.Append(chunk.Text);
                else thinkingSb.Append(chunk.Text);
            }

            sw.Stop();
            return new LlmCompletionResult
            {
                Text = visibleSb.ToString(),
                Thinking = thinkingSb.ToString(),
                // 流式协议的 usage 仅在末块出现且依赖服务端开启, 这里用字符数作
                // 粗略指标; 需精确计费请改用非流式 /chat/completions 读取 response.Usage。
                TotalTokens = visibleSb.Length + thinkingSb.Length,
                ElapsedSeconds = sw.Elapsed.TotalSeconds,
            };
        }

        public IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);
            return StreamInternalAsync(messages, options, ct);
        }

        /// <summary>
        /// Runs one native OpenAI tool-capable model turn. Tool calls are accumulated from
        /// streaming deltas and emitted only after their id, name and JSON arguments are complete.
        /// </summary>
        public async IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
            LlmToolTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(request);
            EnsureNotDisposed();
            ValidateMessages(request.Messages);

            using var linked = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken,
                _shutdownCts.Token);
            CancellationToken token = linked.Token;
            await _gate.WaitAsync(token).ConfigureAwait(false);
            try
            {
                List<OAIChatMessage> messages = OpenAIMessageConverter.Convert(request.Messages);
                ChatCompletionOptions options = OpenAIChatOptionsFactory.Build(_options, request.Options);
                foreach (LlmToolDefinition definition in request.Tools)
                {
                    options.Tools.Add(ChatTool.CreateFunctionTool(
                        definition.Name,
                        definition.Description,
                        BinaryData.FromString(definition.InputSchemaJson),
                        definition.Strict));
                }

                var calls = new Dictionary<int, StreamingToolCallBuilder>();
                ChatTokenUsage? usage = null;
                bool surfaceReasoning = _options.SurfaceReasoning ||
                    (request.Options?.EnableThinking ?? false);

                AsyncCollectionResult<StreamingChatCompletionUpdate> stream =
                    _client.CompleteChatStreamingAsync(messages, options, token);
                await foreach (StreamingChatCompletionUpdate update in stream
                    .WithCancellation(token)
                    .ConfigureAwait(false))
                {
                    if (surfaceReasoning)
                    {
                        string? reasoning = OpenAIReasoningExtractor.TryExtract(update);
                        if (!string.IsNullOrEmpty(reasoning))
                            yield return new LlmTurnTextDelta(reasoning, LlmChunkKind.Thinking);
                    }

                    foreach (ChatMessageContentPart part in update.ContentUpdate)
                    {
                        if (!string.IsNullOrEmpty(part.Text))
                            yield return new LlmTurnTextDelta(part.Text, LlmChunkKind.Text);
                    }

                    foreach (StreamingChatToolCallUpdate toolUpdate in update.ToolCallUpdates)
                    {
                        if (!calls.TryGetValue(toolUpdate.Index, out StreamingToolCallBuilder? call))
                        {
                            call = new StreamingToolCallBuilder();
                            calls.Add(toolUpdate.Index, call);
                        }

                        call.Append(toolUpdate);
                    }

                    usage = update.Usage ?? usage;
                }

                foreach (StreamingToolCallBuilder call in calls.OrderBy(static item => item.Key)
                    .Select(static item => item.Value))
                {
                    yield return call.Build();
                }

                yield return new LlmTurnCompletedEvent(
                    usage?.InputTokenCount ?? 0,
                    usage?.OutputTokenCount ?? 0,
                    usage?.TotalTokenCount ?? 0);
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <summary>
        /// 共用核心: 构造请求 → CompleteChatStreamingAsync → 把 reasoning/content 增量转成
        /// <see cref="LlmTextChunk"/>。用 <see cref="Channel{T}"/> 解耦生产/消费, 配合
        /// <see cref="OpenAILLMOptions.StreamChannelCapacity"/> 反压。
        /// </summary>
        private async IAsyncEnumerable<LlmTextChunk> StreamInternalAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            [EnumeratorCancellation] CancellationToken ct)
        {
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;
            var channel = CreateChannel();

            // gate.WaitAsync 必须在生产任务里, 而不是外层 —— 否则即使消费者已取消, 我们也会卡在锁上。
            var generationTask = Task.Run(() => ProduceAsync(messages, options, channel.Writer, token), token)
                .ContinueWith(t =>
                {
                    Exception? failure = t.IsFaulted ? t.Exception?.GetBaseException() : null;
                    if (t.IsCanceled) failure = new OperationCanceledException(token);
                    channel.Writer.TryComplete(failure);
                }, TaskScheduler.Default);

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

        private async Task ProduceAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            ChannelWriter<LlmTextChunk> writer,
            CancellationToken ct)
        {
            await _gate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                List<OAIChatMessage> chatMessages = OpenAIMessageConverter.Convert(messages);
                ChatCompletionOptions chatOptions = OpenAIChatOptionsFactory.Build(_options, options);

                bool surfaceReasoning = _options.SurfaceReasoning || (options?.EnableThinking ?? false);
                bool sawVisible = false;

                AsyncCollectionResult<StreamingChatCompletionUpdate> stream =
                    _client.CompleteChatStreamingAsync(chatMessages, chatOptions, ct);

                await foreach (StreamingChatCompletionUpdate update in stream.ConfigureAwait(false))
                {
                    if (surfaceReasoning)
                    {
                        string? reasoning = OpenAIReasoningExtractor.TryExtract(update);
                        if (!string.IsNullOrEmpty(reasoning))
                        {
                            await writer.WriteAsync(new LlmTextChunk
                            {
                                Text = reasoning,
                                Kind = LlmChunkKind.Thinking,
                            }, ct).ConfigureAwait(false);
                        }
                    }

                    foreach (ChatMessageContentPart part in update.ContentUpdate)
                    {
                        string? text = part.Text;
                        if (string.IsNullOrEmpty(text)) continue;

                        sawVisible = true;
                        await writer.WriteAsync(new LlmTextChunk
                        {
                            Text = text,
                            Kind = LlmChunkKind.Text,
                        }, ct).ConfigureAwait(false);
                    }
                }

                // 末尾补发一个空的 IsFinal 标记块, 让下游 (TTS / UI) 做收尾,
                // 避免为了拼 IsFinal 而延迟首 token。
                if (sawVisible)
                {
                    await writer.WriteAsync(new LlmTextChunk
                    {
                        Text = string.Empty,
                        Kind = LlmChunkKind.Text,
                        IsFinal = true,
                    }, ct).ConfigureAwait(false);
                }
            }
            finally
            {
                _gate.Release();
            }
        }

        private Channel<LlmTextChunk> CreateChannel()
        {
            if (_options.StreamChannelCapacity > 0)
            {
                return Channel.CreateBounded<LlmTextChunk>(new BoundedChannelOptions(_options.StreamChannelCapacity)
                {
                    SingleReader = true,
                    SingleWriter = true,
                    FullMode = BoundedChannelFullMode.Wait,
                });
            }
            return Channel.CreateUnbounded<LlmTextChunk>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
                AllowSynchronousContinuations = false,
            });
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
            _shutdownCts.Dispose();
            // ChatClient 不持有需显式释放的非托管资源 (其底层 HTTP 管线随 GC 回收),
            // 这里没有可 Dispose 的句柄, 仅释放本服务自有的信号量。
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

        private static void ValidateOptions(OpenAILLMOptions options)
        {
            if (string.IsNullOrWhiteSpace(options.ApiKey))
                throw new ArgumentException("OpenAILLMOptions.ApiKey cannot be null or empty.", nameof(options));
            if (options.Endpoint is null)
                throw new ArgumentException("OpenAILLMOptions.Endpoint cannot be null.", nameof(options));
            if (string.IsNullOrWhiteSpace(options.Model))
                throw new ArgumentException("OpenAILLMOptions.Model cannot be null or empty.", nameof(options));
            if (options.Thinking is ThinkingMode thinking && !Enum.IsDefined(thinking))
                throw new ArgumentOutOfRangeException(nameof(options), "Thinking is not a valid value.");
            if (options.ReasoningEffort is OpenAIReasoningEffortLevel effort && !Enum.IsDefined(effort))
                throw new ArgumentOutOfRangeException(nameof(options), "ReasoningEffort is not a valid value.");
            if (options.MaxConcurrentRequests < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentRequests must be >= 1.");
            if (options.StreamChannelCapacity < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "StreamChannelCapacity must be >= 0.");
            if (options.HttpTimeout <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(options), "HttpTimeout must be > 0.");
        }

        private sealed class StreamingToolCallBuilder
        {
            private readonly MemoryStream _arguments = new();
            private string? _callId;
            private string? _functionName;

            public void Append(StreamingChatToolCallUpdate update)
            {
                if (!string.IsNullOrEmpty(update.ToolCallId))
                    _callId = update.ToolCallId;
                if (!string.IsNullOrEmpty(update.FunctionName))
                    _functionName = update.FunctionName;
                BinaryData? delta = update.FunctionArgumentsUpdate;
                if (delta is not null)
                {
                    ReadOnlyMemory<byte> bytes = delta.ToMemory();
                    _arguments.Write(bytes.Span);
                }
            }

            public LlmToolCallEvent Build()
            {
                if (string.IsNullOrWhiteSpace(_callId) || string.IsNullOrWhiteSpace(_functionName))
                {
                    throw new InvalidOperationException(
                        "OpenAI returned an incomplete streaming tool call.");
                }

                string arguments = _arguments.Length == 0
                    ? "{}"
                    : Encoding.UTF8.GetString(_arguments.ToArray());
                return new LlmToolCallEvent(_callId, _functionName, arguments);
            }
        }
    }
}
