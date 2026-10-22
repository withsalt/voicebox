using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>
    /// <see cref="ILLMConversation"/> 的 GenAI 实现: 持有一个长期存活的 <see cref="Generator"/>,
    /// 跨轮复用 KV-cache。
    ///
    /// 复用机制 (token-level LCP + RewindTo):
    /// 每轮把整段 prompt 渲染+encode 成 <c>newIds</c>, 与 KV-cache 现有序列 <c>_cachedIds</c> 求最长公共前缀 L,
    /// <c>RewindTo(L)</c> 丢掉分叉后的 KV, 再 <c>AppendTokens(newIds[L..])</c> 只补分叉后缀。生成结束后把
    /// <c>_cachedIds</c> 快照为 Generator 真实序列 (含生成出的 token)。因缓存里存的是真实生成 token,
    /// tokenizer 对回复文本通常 round-trip 回同一批 id, 故 assistant 回复段也能命中复用;
    /// round-trip 分叉时自动退化为更短前缀。<b>无论 L 多少, RewindTo+Append 后 KV 序列恒等于 newIds</b>,
    /// 与"新建 Generator 喂 newIds"完全等价 —— 复用只影响省多少 prefill, 不改变输出。
    ///
    /// 不复用的情形 (回退到一次性 Generator): 未启用复用、含音频/图像、采样参数变化。
    /// 会话<b>非线程安全</b>: 同一实例的调用必须串行 (由调用方保证)。
    /// </summary>
    internal sealed class OnnxGenAILLMConversation : ILLMConversation
    {
        private readonly GenAIInferenceSession _session;
        private readonly GenAIPromptTemplate _template;
        private readonly OnnxGenAILLMOptions _options;
        private readonly SemaphoreSlim _serviceGate;
        private readonly CancellationToken _shutdownToken;
        private readonly bool _reuseEnabled;
        private readonly ILogger _logger;

        // 持久 KV-cache 状态。仅在持有 _serviceGate (生成期间) 时读写。
        private Generator? _generator;
        private int[] _cachedIds = Array.Empty<int>();
        private GenAISamplingOptions _cachedSampling;
        private bool _cachedValid;

        private int _disposed;

        public OnnxGenAILLMConversation(
            GenAIInferenceSession session,
            GenAIPromptTemplate template,
            OnnxGenAILLMOptions options,
            SemaphoreSlim serviceGate,
            CancellationToken shutdownToken,
            bool reuseEnabled,
            ILogger logger)
        {
            _session = session;
            _template = template;
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

            var linked = GenAICancellation.LinkIfNeeded(ct, _shutdownToken, out var token);
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

            var linked = GenAICancellation.CreateLinkedSource(ct, _shutdownToken);
            var token = linked.Token;
            var channel = GenAIGeneration.CreateChannel(_options.StreamChannelCapacity);

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
                linked.Cancel();
                try { await generationTask.ConfigureAwait(false); }
                catch { /* 错误已通过 channel 传给消费方; 这里只保证生产任务收尾。 */ }
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
                using var build = GenAIInputBuilder.Build(
                    _template, _session.Tokenizer, messages,
                    addGenerationPrompt: true, enableThinking: options?.EnableThinking ?? false);
                bool hasMedia = build.Audios != null || build.Images != null;

                using var promptSequences = _session.Tokenizer.Encode(build.Prompt);

                // 不复用: 未启用 / 多模态。多模态会打断 token 缓存的连续性, 故作废缓存后走一次性 Generator。
                if (!_reuseEnabled || hasMedia)
                {
                    if (hasMedia) InvalidateCache();
                    return await GenAIGeneration.RunOneShotAsync(
                        _session, _options, build, promptSequences, hasMedia, options, onChunk, ct).ConfigureAwait(false);
                }

                // 复用路径 (纯文本)。
                int[] newIds = promptSequences[0].ToArray();
                int maxNewTokens = options?.MaxTokens ?? _options.MaxTokens;
                var sampling = GenAISamplingOptions.Resolve(_options, options);

                // 采样参数变化 → 重建 Generator (搜索选项创建后不可改)。
                if (_cachedValid && !_cachedSampling.Equals(sampling))
                    InvalidateCache();
                if (!_cachedValid)
                    RebuildGenerator(sampling);

                int reuse = GenAITokenUtils.CommonPrefixLength(_cachedIds, newIds);
                if (reuse < _cachedIds.Length)
                    _generator!.RewindTo((ulong)reuse);
                if (reuse < newIds.Length)
                    _generator!.AppendTokens(newIds.AsSpan(reuse));

                _logger.LogDebug(
                    "GenAI session: prompt={Prompt} tok, reused={Reused}, appended={Appended}",
                    newIds.Length, reuse, newIds.Length - reuse);

                try
                {
                    GenerationStats stats = await GenAIGeneration.RunLoopAsync(
                        _generator!, _session.Tokenizer, maxNewTokens, onChunk, ct).ConfigureAwait(false);
                    SnapshotCache();
                    return stats;
                }
                catch (OperationCanceledException)
                {
                    // 主动取消: Generator 仍处于一致状态, 快照后保留供下轮复用。
                    SnapshotCache();
                    throw;
                }
                catch
                {
                    // 非预期错误: 作废缓存, 下轮全量重建。
                    InvalidateCache();
                    throw;
                }
            }
            finally
            {
                _serviceGate.Release();
            }
        }

        /// <summary>用给定采样参数重建持久 Generator (空 KV-cache)。GeneratorParams 建完即可释放。</summary>
        private void RebuildGenerator(GenAISamplingOptions sampling)
        {
            InvalidateCache();
            using (var p = new GeneratorParams(_session.Model))
            {
                sampling.ApplyTo(p, _options.SessionMaxLength);
                _generator = new Generator(_session.Model, p);
            }
            _cachedIds = Array.Empty<int>();
            _cachedSampling = sampling;
            _cachedValid = true;
        }

        /// <summary>把 _cachedIds 对账为 Generator 的真实序列; 读取失败则作废缓存。</summary>
        private void SnapshotCache()
        {
            if (_generator is null) return;
            try
            {
                _cachedIds = _generator.GetSequence(0).ToArray();
            }
            catch
            {
                InvalidateCache();
            }
        }

        private void InvalidateCache()
        {
            _generator?.Dispose();
            _generator = null;
            _cachedIds = Array.Empty<int>();
            _cachedValid = false;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // 约定: 调用方保证 Dispose 时本会话无进行中的生成 (会话非线程安全)。
            InvalidateCache();
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
