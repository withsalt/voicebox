using System;
using System.Collections.Generic;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Qwen3.5 thinking 模板。</summary>
    public sealed class Qwen35LlamaModelProfile : ChatMlLlamaModelProfile
    {
        public override string Name => "qwen3.5";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.Name, "Qwen3.5") || Contains(info.Name, "Qwen-3.5");

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            string prompt = base.BuildPrompt(
                weights, messages, addGenerationPrompt, enableThinking);
            if (!addGenerationPrompt)
                return prompt;

            return enableThinking
                ? prompt + "<think>\n"
                : prompt + "<think>\n\n</think>\n\n";
        }

        public override bool StartsInsideThinking(bool enableThinking) => enableThinking;
    }
}
