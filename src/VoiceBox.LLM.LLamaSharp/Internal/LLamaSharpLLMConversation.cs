using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LLama;
using LLama.Native;
using Microsoft.Extensions.Logging;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    /// <summary>
    /// <see cref="ILLMConversation"/> 的 LLamaSharp 实现: 持有独立的 llama.cpp context,
    /// 跨轮复用 KV-cache。
    ///
    /// 复用机制 (token-level LCP + MemorySequenceRemove):
    /// 每轮把整段 prompt 渲染+tokenize 成 <c>newIds</c>, 与 KV-cache 现有序列 <c>_cachedIds</c>
    /// 求最长公共前缀 L, <c>MemorySequenceRemove(0, L, ∞)</c> 丢掉分叉后的 KV,
    /// 再只对 <c>newIds[L..]</c> 做增量 prefill。生成期间每个成功落入 KV 的 token
    /// 都即时记入 <c>_cachedIds</c> — 因此无论取消还是异常, <c>_cachedIds</c> 恒等于 KV 实际内容,
    /// 下一轮的 LCP 对账天然正确。因缓存里存的是真实生成 token, tokenizer 对回复文本通常
    /// round-trip 回同一批 id, 故 assistant 回复段也能命中复用; 分叉时自动退化为更短前缀。
    /// 对支持 cache shift 的纯 Transformer，Remove+Prefill 后 KV 序列恒等于 newIds。
    /// Qwen3.5/Gemma 4 等 hybrid/recurrent context 不支持从中间回滚；检测到 prompt 分叉时
    /// 自动清空 memory 并全量 prefill，避免 recurrent state 保留较大的旧位置而导致
    /// M-RoPE <c>X &lt; Y</c> 校验失败。此降级不改变对话语义。
    ///
    /// 与 ORT-GenAI 实现不同, llama.cpp 的采样管线独立于 KV-cache, 采样参数变化不需要作废缓存。
    /// 会话<b>非线程安全</b>: 同一实例的调用必须串行 (由调用方保证)。
    /// </summary>
    internal sealed class LLamaSharpLLMConversation : ILLMConversation
    {
        private readonly LlamaModelSession _session;
        private readonly LLamaSharpLLMOptions _options;
        private readonly SemaphoreSlim _serviceGate;
        private readonly CancellationToken _shutdownToken;
        private readonly bool _reuseEnabled;
        private readonly ILogger _logger;

        // 持久 KV-cache 状态。仅在持有 _serviceGate (生成期间) 时读写。
        private LLamaContext? _context;
        private readonly List<LLamaToken> _cachedIds = new();

        private int _disposed;

        public LLamaSharpLLMConversation(
            LlamaModelSession session,
            LLamaSharpLLMOptions options,
            SemaphoreSlim serviceGate,
            CancellationToken shutdownToken,
            bool reuseEnabled,
            ILogger logger)
        {
            _session = session;
            _options = options;
            _serviceGate = serviceGate;
            _shutdownToken = shutdownToken;
            _reuseEnabled = reuseEnabled;
            _logger = logger;
        }

        public async Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            EnsureNotDisposed();
            ValidateMessages(messages);

            var linked = LlamaCancellation.LinkIfNeeded(ct, _shutdownToken, out var token);
            var sw = Stopwatch.StartNew();
            var visibleSb = new StringBuilder();
            var thinkingSb = new StringBuilder();
            try
            {
                GenerationStats stats = await Task.Run(
                    () => RunTurnAsync(
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

            var linked = LlamaCancellation.CreateLinkedSource(ct, _shutdownToken);
            var token = linked.Token;
            var channel = LlamaGeneration.CreateChannel(_options.StreamChannelCapacity);

            var generationTask = Task.Run(async () =>
            {
                Exception? failure = null;
                try
                {
                    await RunTurnAsync(
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

        /// <summary>一轮生成的核心: 全程持有 service gate (与无状态调用互斥)。</summary>
        private async Task<GenerationStats> RunTurnAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            Func<LlmTextChunk, ValueTask> onChunk,
            CancellationToken ct)
        {
            await _serviceGate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                bool enableThinking = options?.EnableThinking ?? false;
                string prompt = LlamaPromptBuilder.Build(
                    _session, messages, addGenerationPrompt: true, enableThinking);

                var context = _context ??= _session.CreateConversationContext();
                LLamaToken[] newIds = context.Tokenize(prompt, addBos: true, special: true);
                if (newIds.Length == 0)
                    throw new InvalidOperationException("Prompt tokenized to zero tokens.");

                int reuse;
                if (_reuseEnabled)
                {
                    reuse = LlamaTokenUtils.CommonPrefixLength(_cachedIds, newIds);
                    // prompt 是缓存的完整前缀时 (如重新生成同一轮), 至少回退一个 token,
                    // 重新 decode 它以获得末位 logits。
                    if (reuse == newIds.Length)
                        reuse = newIds.Length - 1;
                }
                else
                {
                    reuse = 0;
                }

                bool mustRollback = reuse < _cachedIds.Count;
                if (mustRollback && !context.NativeHandle.MemoryCanShift)
                {
                    // Hybrid/recurrent memory（如 Qwen3.5 的 Gated Delta Net）不能只删除
                    // 序列尾部。MemorySequenceRemove 会留下 recurrent state 的旧最大位置，
                    // 下一批从较小位置 decode 时触发 M-RoPE X < Y 错误。完整清空可安全重建。
                    context.NativeHandle.MemoryClear(true);
                    _cachedIds.Clear();
                    reuse = 0;
                    _logger.LogDebug(
                        "LLamaSharp session: partial cache rollback unsupported; rebuilding full prompt.");
                }
                else
                {
                    // 上一轮若中途取消/失败，KV 里可能残留分叉后的 token；支持回滚的
                    // context 只移除尾部并保留最长公共前缀。
                    context.NativeHandle.MemorySequenceRemove(LLamaSeqId.Zero, reuse, -1);
                    if (_cachedIds.Count > reuse)
                        _cachedIds.RemoveRange(reuse, _cachedIds.Count - reuse);
                }

                _logger.LogDebug(
                    "LLamaSharp session: prompt={Prompt} tok, reused={Reused}, appended={Appended}",
                    newIds.Length, reuse, newIds.Length - reuse);

                try
                {
                    return await LlamaGeneration.RunAsync(
                        _session, context, newIds, reuse,
                        _session.Profile.StartsInsideThinking(enableThinking),
                        LlamaSamplingOptions.Resolve(_options, options),
                        options?.MaxTokens ?? _options.MaxTokens,
                        onChunk,
                        idTracker: _cachedIds,
                        _logger, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // 主动取消: _cachedIds 已随每次成功 decode 即时更新, 与 KV 一致, 保留供下轮复用。
                    throw;
                }
                catch
                {
                    // 非预期错误: 不再信任 KV 状态, 清空后下轮全量重建。
                    InvalidateCache();
                    throw;
                }
            }
            finally
            {
                _serviceGate.Release();
            }
        }

        private void InvalidateCache()
        {
            _cachedIds.Clear();
            try
            {
                _context?.NativeHandle.MemoryClear(true);
            }
            catch
            {
                // context 已不可用: 丢弃, 下轮重建。
                _context?.Dispose();
                _context = null;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // 约定: 调用方保证 Dispose 时本会话无进行中的生成 (会话非线程安全)。
            _cachedIds.Clear();
            _context?.Dispose();
            _context = null;
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
