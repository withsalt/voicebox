using System;
using System.Collections.Generic;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Qwen3 hybrid-thinking 模板。</summary>
    public sealed class Qwen3LlamaModelProfile : ChatMlLlamaModelProfile
    {
        public override string Name => "qwen3";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.Equals("qwen3", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.Name, "Qwen3") || Contains(info.Name, "Qwen-3");

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            string prompt = base.BuildPrompt(
                weights, messages, addGenerationPrompt, enableThinking);
            return addGenerationPrompt && !enableThinking
                ? prompt + "<think>\n\n</think>\n\n"
                : prompt;
        }
    }
}
