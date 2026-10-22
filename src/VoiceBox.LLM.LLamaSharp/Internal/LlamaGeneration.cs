using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using LLama;
using LLama.Native;
using LLama.Sampling;
using Microsoft.Extensions.Logging;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    /// <summary>一次生成的 token 统计: 总 token 数 (prompt + 新增) 与新增 token 数。</summary>
    internal readonly record struct GenerationStats(int TotalTokens, int NewTokens);

    /// <summary>
    /// 解析后的采样选项。作为 record struct 天然可比较, 便于测试与日志。
    /// 与 ORT-GenAI 不同, llama.cpp 的采样器独立于 KV-cache — 每轮新建采样管线即可,
    /// 参数变化不需要作废任何缓存。
    /// </summary>
    internal readonly record struct LlamaSamplingOptions(
        bool DoSample, float Temperature, int TopK, float TopP, float RepetitionPenalty, int PenaltyWindow, int? Seed)
    {
        public static LlamaSamplingOptions Resolve(LLamaSharpLLMOptions defaults, LlmCompleteOptions? call) => new(
            call?.DoSample ?? defaults.DoSample,
            call?.Temperature ?? defaults.Temperature,
            call?.TopK ?? defaults.TopK,
            call?.TopP ?? defaults.TopP,
            call?.RepetitionPenalty ?? defaults.RepetitionPenalty,
            defaults.RepetitionPenaltyWindow,
            defaults.RandomSeed is int s && s >= 0 ? s : null);

        /// <summary>
        /// 建出本轮使用的采样管线。管线持有 native 采样链, 用完必须 Dispose。
        /// 贪心路径 (<see cref="DoSample"/>=false) 完全确定; 采样路径的重复惩罚只回看
        /// 本轮已生成的 token (管线每轮新建, 与无状态推理语义一致)。
        /// </summary>
        public ISamplingPipeline CreatePipeline(string? grammar = null)
        {
            if (!DoSample && grammar is null)
                return new GreedySamplingPipeline();

            var nativeGrammar = grammar is null ? null : new Grammar(grammar, "root");
            if (!DoSample)
                return new GreedySamplingPipeline { Grammar = nativeGrammar };

            var pipeline = new DefaultSamplingPipeline
            {
                Temperature = Temperature,
                TopK = TopK,
                TopP = TopP,
                RepeatPenalty = RepetitionPenalty,
                PenaltyCount = PenaltyWindow,
                Grammar = nativeGrammar,
            };
            if (Seed is int seed)
                pipeline.Seed = (uint)seed;
            return pipeline;
        }
    }

    /// <summary>
    /// llama.cpp 生成的可复用内核: prefill、解码循环、通道创建。
    /// 无状态服务路径与有状态会话路径共用这些, 避免两处各写一遍。
    /// 这些方法均<b>不</b>获取并发闸门 —— 调用方需在持有 gate 的前提下调用;
    /// 同一 context 上的调用必须串行。
    /// </summary>
    internal static class LlamaGeneration
    {
        /// <summary>
        /// 在给定 context 上跑一轮完整生成: 增量 prefill <paramref name="promptTokens"/> 的
        /// [<paramref name="reusedTokens"/>..] 后缀, 然后逐 token 解码直到 EOG / 上限 / 停止词。
        ///
        /// 前置契约: context 的 KV 中当前恰好含有 <paramref name="promptTokens"/> 的前
        /// <paramref name="reusedTokens"/> 个 token (无状态路径为 0, 即已清空)。
        /// <paramref name="idTracker"/> 非空时, 每个成功进入 KV 的 token 都会即时追加进去 —
        /// 无论中途取消还是异常, 它始终与 KV 实际内容一致, 会话据此实现跨轮增量 prefill。
        ///
        /// 输出约定 (与 OnnxGenAI/OpenAI 等服务一致): 内容块立即发出 (不为拼 IsFinal 而缓冲);
        /// 结束后仅在产生过可见文本时补一个空的 <see cref="LlmTextChunk.IsFinal"/> 标记块 (Kind=Text)。
        /// </summary>
        public static async Task<GenerationStats> RunAsync(
            LlamaModelSession session,
            LLamaContext context,
            IReadOnlyList<LLamaToken> promptTokens,
            int reusedTokens,
            bool startsInsideThinking,
            LlamaSamplingOptions sampling,
            int maxNewTokens,
            Func<LlmTextChunk, ValueTask> onChunk,
            List<LLamaToken>? idTracker,
            ILogger logger,
            CancellationToken ct,
            string? grammar = null)
        {
            int ctxSize = checked((int)context.ContextSize);
            if (promptTokens.Count >= ctxSize)
            {
                throw new InvalidOperationException(
                    $"Prompt ({promptTokens.Count} tokens) does not fit the context window ({ctxSize}). " +
                    "Increase LLamaSharpLLMOptions.ContextSize or trim the conversation history.");
            }
            if (reusedTokens < 0 || reusedTokens >= promptTokens.Count)
                throw new ArgumentOutOfRangeException(nameof(reusedTokens));

            var batch = new LLamaBatch();

            // ---- prefill: 按 BatchSize 分批投喂 prompt 后缀, 仅最后一个 token 要 logits ----
            int pos = reusedTokens;
            while (pos < promptTokens.Count)
            {
                ct.ThrowIfCancellationRequested();
                int take = Math.Min(session.BatchSize, promptTokens.Count - pos);
                batch.Clear();
                for (int i = 0; i < take; i++)
                {
                    int tokenPos = pos + i;
                    batch.Add(promptTokens[tokenPos], tokenPos, LLamaSeqId.Zero, logits: tokenPos == promptTokens.Count - 1);
                }
                Decode(context, batch);
                if (idTracker is not null)
                {
                    for (int i = 0; i < take; i++)
                        idTracker.Add(promptTokens[pos + i]);
                }
                pos += take;
            }

            // ---- 解码循环 ----
            var tokenDecoder = new StreamingTokenDecoder(session.Encoding, session.Weights)
            {
                // special token 解码为文本, 由 LlamaStreamDecoder 按 profile 标记表统一路由/剔除;
                // 设为 false 会让 <think> 等标记直接不可见, thinking 路由就无从谈起。
                DecodeSpecialTokens = true,
            };
            var streamDecoder = new LlamaStreamDecoder(session.DecoderConfig, startsInsideThinking);
            var vocab = session.Weights.Vocab;

            bool sawVisible = false;
            int newTokens = 0;
            var pipeline = sampling.CreatePipeline(grammar);
            try
            {
                int logitIndex = batch.TokenCount - 1;
                while (true)
                {
                    // 计数上限在采样前检查: maxNewTokens<=0 时不再生成任何 token (旧实现先采样再判断, 会多发 1 个)。
                    if (newTokens >= maxNewTokens) break;
                    ct.ThrowIfCancellationRequested();

                    LLamaToken token = pipeline.Sample(context.NativeHandle, logitIndex);
                    if (token.IsEndOfGeneration(vocab))
                        break;

                    if (pos >= ctxSize)
                    {
                        logger.LogWarning(
                            "Generation stopped early: context window is full ({ContextSize} tokens).", ctxSize);
                        break;
                    }

                    // 先把 token 落进 KV (下一步采样依赖它的 logits), 成功后再计入 idTracker,
                    // 保证 idTracker 与 KV 严格一致。
                    batch.Clear();
                    batch.Add(token, pos, LLamaSeqId.Zero, logits: true);
                    Decode(context, batch);
                    idTracker?.Add(token);
                    pos++;
                    newTokens++;
                    logitIndex = 0;

                    tokenDecoder.Add(token);
                    string raw = tokenDecoder.Read();
                    if (!string.IsNullOrEmpty(raw))
                    {
                        streamDecoder.Feed(raw, out string visible, out string thinking);
                        if (thinking.Length > 0)
                            await onChunk(new LlmTextChunk { Text = thinking, Kind = LlmChunkKind.Thinking }).ConfigureAwait(false);
                        if (visible.Length > 0)
                        {
                            sawVisible = true;
                            await onChunk(new LlmTextChunk { Text = visible, Kind = LlmChunkKind.Text }).ConfigureAwait(false);
                        }
                    }

                    if (newTokens >= maxNewTokens || streamDecoder.Stopped)
                        break;
                }
            }
            finally
            {
                pipeline.Dispose();
            }

            streamDecoder.Flush(out string tailVisible, out string tailThinking);
            if (tailThinking.Length > 0)
                await onChunk(new LlmTextChunk { Text = tailThinking, Kind = LlmChunkKind.Thinking }).ConfigureAwait(false);
            if (tailVisible.Length > 0)
            {
                sawVisible = true;
                await onChunk(new LlmTextChunk { Text = tailVisible, Kind = LlmChunkKind.Text }).ConfigureAwait(false);
            }

            if (sawVisible)
                await onChunk(new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true }).ConfigureAwait(false);

            return new GenerationStats(promptTokens.Count + newTokens, newTokens);
        }

        /// <summary>
        /// 无状态完整生成: 清空共享 context 的 KV 后全量 prefill。
        /// 供 <c>LLamaSharpLLMService</c> 的 Complete/Stream 共用。
        /// </summary>
        public static Task<GenerationStats> RunStatelessAsync(
            LlamaModelSession session,
            LLamaSharpLLMOptions defaults,
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            Func<LlmTextChunk, ValueTask> onChunk,
            ILogger logger,
            CancellationToken ct,
            string? grammar = null)
        {
            bool enableThinking = grammar is null && (options?.EnableThinking ?? false);
            string prompt = LlamaPromptBuilder.Build(
                session, messages, addGenerationPrompt: true, enableThinking);
            LLamaToken[] tokens = session.SharedContext.Tokenize(prompt, addBos: true, special: true);
            if (tokens.Length == 0)
                throw new InvalidOperationException("Prompt tokenized to zero tokens.");

            session.SharedContext.NativeHandle.MemoryClear(true);

            return RunAsync(
                session, session.SharedContext, tokens,
                reusedTokens: 0,
                startsInsideThinking: session.Profile.StartsInsideThinking(enableThinking),
                LlamaSamplingOptions.Resolve(defaults, options),
                options?.MaxTokens ?? defaults.MaxTokens,
                onChunk,
                idTracker: null,
                logger, ct, grammar);
        }

        /// <summary>decode 一个 batch 并把非 Ok 结果转成明确的异常。</summary>
        public static void Decode(LLamaContext context, LLamaBatch batch)
        {
            DecodeResult result = context.Decode(batch);
            if (result == DecodeResult.Ok) return;

            throw new InvalidOperationException(result == DecodeResult.NoKvSlot
                ? "llama.cpp could not find a KV cache slot (context window exhausted or fragmented)."
                : $"llama_decode failed: {result}.");
        }

        /// <summary>流式输出用的 Channel: 有界 (反压) 或无界。</summary>
        public static Channel<LlmTextChunk> CreateChannel(int capacity)
        {
            if (capacity > 0)
            {
                return Channel.CreateBounded<LlmTextChunk>(new BoundedChannelOptions(capacity)
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
    }
}
