using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>一次生成的 token 统计: 总 token 数 (prompt + 新增) 与新增 token 数。</summary>
    internal readonly record struct GenerationStats(int TotalTokens, int NewTokens);

    /// <summary>
    /// 解析后的搜索选项 (采样 + 种子)。作为 <see cref="record struct"/> 天然可比较,
    /// 会话据此判断"采样参数是否变化"以决定是否需要重建 Generator。
    /// </summary>
    internal readonly record struct GenAISamplingOptions(
        bool DoSample, float Temperature, int TopK, float TopP, float RepetitionPenalty, int? RandomSeed)
    {
        public static GenAISamplingOptions Resolve(OnnxGenAILLMOptions defaults, LlmCompleteOptions? call) => new(
            call?.DoSample ?? defaults.DoSample,
            call?.Temperature ?? defaults.Temperature,
            call?.TopK ?? defaults.TopK,
            call?.TopP ?? defaults.TopP,
            call?.RepetitionPenalty ?? defaults.RepetitionPenalty,
            defaults.RandomSeed is int s && s >= 0 ? s : null);

        /// <summary>把 max_length 与全部采样选项写入 <paramref name="p"/>。</summary>
        public void ApplyTo(GeneratorParams p, int maxLength)
        {
            p.SetSearchOption("max_length", maxLength);
            p.SetSearchOption("do_sample", DoSample);
            p.SetSearchOption("temperature", Temperature);
            p.SetSearchOption("top_k", TopK);
            p.SetSearchOption("top_p", TopP);
            if (Math.Abs(RepetitionPenalty - 1.0f) > float.Epsilon)
                p.SetSearchOption("repetition_penalty", RepetitionPenalty);
            if (RandomSeed is int seed)
                p.SetSearchOption("random_seed", seed);
        }
    }

    /// <summary>
    /// GenAI 生成的可复用内核: 解码循环、一次性生成、输入投喂、通道创建、前缀比对。
    /// 无状态服务路径与有状态会话路径共用这些, 避免两处各写一遍。
    /// 这些方法均<b>不</b>获取并发闸门 —— 调用方需在持有 gate 的前提下调用。
    /// </summary>
    internal static class GenAIGeneration
    {
        /// <summary>
        /// 在一个"已投喂好 prompt"的 Generator 上跑解码循环, 边解码边通过 <paramref name="onChunk"/> 送出增量。
        ///
        /// 约定 (与其他流式 LLM 服务一致): 内容块立即发出 (最低延迟, 不为拼 IsFinal 而缓冲);
        /// 结束后仅在产生过可见文本时补一个空的 <see cref="LlmTextChunk.IsFinal"/> 标记块 (Kind=Text)。
        /// 新 token 上限由 <paramref name="maxNewTokens"/> 按计数权威强制 (max_length 仅作 ORT 侧兜底)。
        /// </summary>
        public static async Task<GenerationStats> RunLoopAsync(
            Generator generator,
            Tokenizer tokenizer,
            int maxNewTokens,
            Func<LlmTextChunk, ValueTask> onChunk,
            CancellationToken ct)
        {
            using var tokenStream = tokenizer.CreateStream();
            var decoder = new GenAIStreamDecoder();

            bool sawVisible = false;
            int newTokens = 0;
            while (!generator.IsDone())
            {
                // 计数上限在生成前检查: maxNewTokens<=0 时不再生成任何 token (旧实现先生成再判断, 会多发 1 个)。
                if (newTokens >= maxNewTokens) break;
                ct.ThrowIfCancellationRequested();
                generator.GenerateNextToken();

                var nextTokens = generator.GetNextTokens();
                if (nextTokens.Length == 0) continue;
                newTokens++;  // 单序列每步产出一个新 token; 只解码/发出 nextTokens[0]。

                string raw = tokenStream.Decode(nextTokens[0]);
                if (!string.IsNullOrEmpty(raw))
                {
                    decoder.Feed(raw, out string visible, out string thinking);
                    if (thinking.Length > 0)
                        await onChunk(new LlmTextChunk { Text = thinking, Kind = LlmChunkKind.Thinking }).ConfigureAwait(false);
                    if (visible.Length > 0)
                    {
                        sawVisible = true;
                        await onChunk(new LlmTextChunk { Text = visible, Kind = LlmChunkKind.Text }).ConfigureAwait(false);
                    }
                }

                if (newTokens >= maxNewTokens) break;
            }

            decoder.Flush(out string tailVisible, out string tailThinking);
            if (tailThinking.Length > 0)
                await onChunk(new LlmTextChunk { Text = tailThinking, Kind = LlmChunkKind.Thinking }).ConfigureAwait(false);
            if (tailVisible.Length > 0)
            {
                sawVisible = true;
                await onChunk(new LlmTextChunk { Text = tailVisible, Kind = LlmChunkKind.Text }).ConfigureAwait(false);
            }

            if (sawVisible)
                await onChunk(new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true }).ConfigureAwait(false);

            return new GenerationStats((int)generator.TokenCount(), newTokens);
        }

        /// <summary>
        /// 用一个一次性 Generator 跑完整生成 (无 KV-cache 复用)。调用方已 build+encode。
        /// max_length = prompt + 新 token (+ 多模态余量), 作 ORT 侧兜底。
        /// </summary>
        public static async Task<GenerationStats> RunOneShotAsync(
            GenAIInferenceSession session,
            OnnxGenAILLMOptions defaults,
            GenAIInputBuilder.BuildResult build,
            Sequences promptSequences,
            bool hasMedia,
            LlmCompleteOptions? options,
            Func<LlmTextChunk, ValueTask> onChunk,
            CancellationToken ct)
        {
            int promptTokenCount = promptSequences[0].Length;
            int maxNewTokens = options?.MaxTokens ?? defaults.MaxTokens;
            int maxLength = promptTokenCount + maxNewTokens + (hasMedia ? defaults.MaxLengthMediaSafetyMargin : 0);

            using var generatorParams = new GeneratorParams(session.Model);
            GenAISamplingOptions.Resolve(defaults, options).ApplyTo(generatorParams, maxLength);
            using var generator = new Generator(session.Model, generatorParams);

            FeedInput(session, generator, build, promptSequences, hasMedia);
            return await RunLoopAsync(generator, session.Tokenizer, maxNewTokens, onChunk, ct).ConfigureAwait(false);
        }

        /// <summary>build + encode + <see cref="RunOneShotAsync"/> 的组合, 供无状态服务路径直接调用。</summary>
        public static async Task<GenerationStats> RunStatelessAsync(
            GenAIInferenceSession session,
            OnnxGenAILLMOptions defaults,
            GenAIPromptTemplate template,
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options,
            Func<LlmTextChunk, ValueTask> onChunk,
            CancellationToken ct)
        {
            using var build = GenAIInputBuilder.Build(
                template, session.Tokenizer, messages,
                addGenerationPrompt: true, enableThinking: options?.EnableThinking ?? false);
            bool hasMedia = build.Audios != null || build.Images != null;

            using var promptSequences = session.Tokenizer.Encode(build.Prompt);
            return await RunOneShotAsync(session, defaults, build, promptSequences, hasMedia, options, onChunk, ct).ConfigureAwait(false);
        }

        /// <summary>把 build 出的输入投喂给 Generator: 多模态走 Processor, 纯文本复用已 encode 的 sequences。</summary>
        public static void FeedInput(
            GenAIInferenceSession session, Generator generator,
            GenAIInputBuilder.BuildResult build, Sequences promptSequences, bool hasMedia)
        {
            if (hasMedia)
            {
                if (session.Processor == null)
                {
                    throw new InvalidOperationException(
                        "Multi-modal input was provided but the model is text-only " +
                        "(no vision/speech section in genai_config.json).");
                }

                if (build.Images is not null && build.Audios is not null)
                {
                    using var inputs = session.Processor.ProcessImagesAndAudios(build.Prompt, build.Images, build.Audios);
                    generator.SetInputs(inputs);
                }
                else if (build.Images is not null)
                {
                    using var inputs = session.Processor.ProcessImages(build.Prompt, build.Images);
                    generator.SetInputs(inputs);
                }
                else
                {
                    using var inputs = session.Processor.ProcessAudios(build.Prompt, build.Audios!);
                    generator.SetInputs(inputs);
                }
            }
            else
            {
                generator.AppendTokenSequences(promptSequences);
            }
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
