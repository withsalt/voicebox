using System;
using System.Collections.Generic;
using LLama;
using LLama.Transformers;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// 可继承的通用 profile 基类：chat template 走 llama.cpp 内置渲染，
    /// thinking 标记采用 <c>&lt;think&gt;</c> 约定。
    /// </summary>
    public class DefaultLlamaModelProfile : ILlamaModelProfile
    {
        private static readonly string[] DefaultThinkStart = ["<think>"];
        private static readonly string[] DefaultThinkEnd = ["</think>"];
        private static readonly string[] DefaultStrip =
        [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        ];

        public virtual string Name => "default";

        public virtual bool Matches(LlamaModelInfo info) => true;

        public virtual string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(weights);
            ArgumentNullException.ThrowIfNull(messages);

            var template = new LLamaTemplate(weights, strict: false)
            {
                AddAssistant = addGenerationPrompt,
            };
            foreach (var message in messages)
                template.Add(message.Role, message.Text);

            return PromptTemplateTransformer.ToModelPrompt(template);
        }

        public virtual IReadOnlyList<string> ThinkingStartMarkers => DefaultThinkStart;
        public virtual IReadOnlyList<string> ThinkingEndMarkers => DefaultThinkEnd;
        public virtual IReadOnlyList<string> StripMarkers => DefaultStrip;
        public virtual IReadOnlyList<string> StopTexts => Array.Empty<string>();
        public virtual bool StartsInsideThinking(bool enableThinking) => false;
    }
}
