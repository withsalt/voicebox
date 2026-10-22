using System;
using System.Collections.Generic;
using LLama;
using LLama.Transformers;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>使用 GGUF 内嵌 chat template 的原生渲染 Profile。</summary>
    public sealed class NativeLlamaModelProfile : DefaultLlamaModelProfile
    {
        private readonly RawLlamaModelProfile _fallback = new();

        public override string Name => "native";

        public override bool Matches(LlamaModelInfo info) =>
            !string.IsNullOrWhiteSpace(info.ChatTemplate);

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(weights);
            ArgumentNullException.ThrowIfNull(messages);
            try
            {
                var template = new LLamaTemplate(weights, strict: true)
                {
                    AddAssistant = addGenerationPrompt,
                };
                foreach (var message in messages)
                    template.Add(message.Role, message.Text);

                string prompt = PromptTemplateTransformer.ToModelPrompt(template);
                return string.IsNullOrWhiteSpace(prompt)
                    ? _fallback.BuildPrompt(
                        weights, messages, addGenerationPrompt, enableThinking)
                    : prompt;
            }
            catch (Exception exception) when (
                exception is not OutOfMemoryException and not AccessViolationException)
            {
                return _fallback.BuildPrompt(
                    weights, messages, addGenerationPrompt, enableThinking);
            }
        }
    }
}
