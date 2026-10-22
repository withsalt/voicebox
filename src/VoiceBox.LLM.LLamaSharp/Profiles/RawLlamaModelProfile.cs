using System;
using System.Collections.Generic;
using System.Text;
using LLama;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>没有可识别 chat template 时的最终安全兜底。</summary>
    public sealed class RawLlamaModelProfile : DefaultLlamaModelProfile
    {
        public override string Name => "raw";
        public override bool Matches(LlamaModelInfo info) => true;

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(messages);
            var prompt = new StringBuilder();
            foreach (var message in messages)
            {
                string role = message.Role switch
                {
                    "system" => "System",
                    "assistant" => "Assistant",
                    "tool" => "Tool",
                    _ => "User",
                };
                prompt.Append(role).Append(": ").Append(message.Text.Trim()).Append('\n');
            }
            if (addGenerationPrompt)
                prompt.Append("Assistant: ");
            return prompt.ToString();
        }
    }
}
