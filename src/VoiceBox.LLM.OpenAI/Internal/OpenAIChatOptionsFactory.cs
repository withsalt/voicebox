using System;
using OpenAI.Chat;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OpenAI.Internal
{
    /// <summary>
    /// 把 <see cref="OpenAILLMOptions"/> (服务级默认) 与单次调用的 <see cref="LlmCompleteOptions"/>
    /// (按调用覆盖) 折叠为 OpenAI 的 <see cref="ChatCompletionOptions"/>。
    ///
    /// 协议差异提示: OpenAI 协议没有 <c>top_k</c> / <c>repetition_penalty</c> 字段, 所以
    /// <see cref="LlmCompleteOptions.TopK"/> 与 <see cref="LlmCompleteOptions.RepetitionPenalty"/>
    /// 在此被有意忽略; 抑制重复请用服务级的
    /// <see cref="OpenAILLMOptions.FrequencyPenalty"/> / <see cref="OpenAILLMOptions.PresencePenalty"/>。
    /// </summary>
    internal static class OpenAIChatOptionsFactory
    {
        public static ChatCompletionOptions Build(OpenAILLMOptions baseOptions, LlmCompleteOptions? perCall)
        {
            int maxTokens = perCall?.MaxTokens ?? baseOptions.MaxTokens;
            bool doSample = perCall?.DoSample ?? baseOptions.DoSample;

            float temperature = doSample
                ? (perCall?.Temperature ?? baseOptions.Temperature)
                : 0f;

            float topP = doSample
                ? (perCall?.TopP ?? baseOptions.TopP)
                : 1f;

            var options = new ChatCompletionOptions
            {
                Temperature = temperature,
            };

            if (maxTokens > 0)
                options.MaxOutputTokenCount = maxTokens;

            // top_p=1.0 等于不裁剪, 没必要下发; 区间外的非法值也不下发, 交给服务端默认。
            if (topP > 0f && topP < 1f)
                options.TopP = topP;

            if (baseOptions.FrequencyPenalty is float fp)
                options.FrequencyPenalty = fp;

            if (baseOptions.PresencePenalty is float pp)
                options.PresencePenalty = pp;

            if (baseOptions.Seed is int seed)
            {
                // Seed 在当前 OpenAI SDK 标注为 [Experimental(OPENAI001)]; 我们刻意使用并就地抑制该诊断,
                // 不做项目级全局抑制, 以免误掩其它实验性 API 的使用。
#pragma warning disable OPENAI001
                options.Seed = seed;
#pragma warning restore OPENAI001
            }

            ThinkingMode? thinking = ResolveThinkingMode(baseOptions, perCall);
            if (thinking is ThinkingMode thinkingMode && thinkingMode != ThinkingMode.Auto)
            {
#pragma warning disable SCME0001
                options.Patch.Set("$.thinking"u8, ToThinkingJson(thinkingMode));
#pragma warning restore SCME0001
            }

            if (baseOptions.ReasoningEffort is OpenAIReasoningEffortLevel reasoningEffort)
            {
#pragma warning disable SCME0001
                options.Patch.Set("$.reasoning_effort"u8, ToReasoningEffortValue(reasoningEffort));
#pragma warning restore SCME0001
            }

            if (baseOptions.StopSequences is { Length: > 0 } stops)
            {
                foreach (var stop in stops)
                {
                    if (!string.IsNullOrEmpty(stop))
                        options.StopSequences.Add(stop);
                }
            }

            return options;
        }

        private static ThinkingMode? ResolveThinkingMode(OpenAILLMOptions baseOptions, LlmCompleteOptions? perCall)
        {
            if (baseOptions.Thinking is ThinkingMode thinking)
                return thinking;

            if (perCall?.EnableThinking == true)
                return ThinkingMode.Enabled;

            return baseOptions.EnableThinking switch
            {
                true => ThinkingMode.Enabled,
                false => ThinkingMode.Disabled,
                null => null,
            };
        }

        private static BinaryData ToThinkingJson(ThinkingMode thinking)
        {
            return BinaryData.FromString("{\"type\":\"" + ToThinkingValue(thinking) + "\"}");
        }

        private static string ToThinkingValue(ThinkingMode thinking)
        {
            return thinking switch
            {
                ThinkingMode.Enabled => "enabled",
                ThinkingMode.Disabled => "disabled",
                _ => throw new ArgumentOutOfRangeException(nameof(thinking), thinking, "Unsupported thinking mode."),
            };
        }

        private static string ToReasoningEffortValue(OpenAIReasoningEffortLevel effort)
        {
            return effort switch
            {
                OpenAIReasoningEffortLevel.None => "none",
                OpenAIReasoningEffortLevel.Minimal => "minimal",
                OpenAIReasoningEffortLevel.Low => "low",
                OpenAIReasoningEffortLevel.Medium => "medium",
                OpenAIReasoningEffortLevel.High => "high",
                OpenAIReasoningEffortLevel.Max => "max",
                _ => throw new ArgumentOutOfRangeException(nameof(effort), effort, "Unsupported reasoning effort."),
            };
        }
    }
}
