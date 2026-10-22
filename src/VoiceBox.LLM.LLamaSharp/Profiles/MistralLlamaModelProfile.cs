using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Mistral 与 Llama 2 常见的 <c>[INST]</c> 模板。</summary>
    public sealed class MistralLlamaModelProfile : DefaultLlamaModelProfile
    {
        private static readonly string[] Strip = ["[INST]", "[/INST]"];

        public override string Name => "mistral";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.Contains("mistral", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.Name, "Mistral") || Contains(info.ChatTemplate, "[INST]");

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(messages);
            var prompt = new StringBuilder();
            string? system = messages.FirstOrDefault(message => message.Role == "system").Text;
            for (int index = 0; index < messages.Count; index++)
            {
                LlamaChatMessage message = messages[index];
                if (message.Role == "system")
                    continue;
                if (message.Role == "user")
                {
                    prompt.Append("[INST] ");
                    if (!string.IsNullOrWhiteSpace(system))
                    {
                        prompt.Append(system.Trim()).Append("\n\n");
                        system = null;
                    }
                    prompt.Append(message.Text.Trim()).Append(" [/INST]");
                }
                else if (message.Role == "assistant")
                {
                    prompt.Append(' ').Append(message.Text.Trim()).Append("</s>");
                }
                else if (message.Role == "tool")
                {
                    prompt.Append("[INST] Tool results:\n");
                    do
                    {
                        prompt.Append(message.Text.Trim()).Append('\n');
                        if (index + 1 >= messages.Count || messages[index + 1].Role != "tool")
                            break;
                        message = messages[++index];
                    } while (true);
                    prompt.Append("[/INST]");
                }
            }
            return prompt.ToString();
        }

        public override IReadOnlyList<string> StripMarkers => Strip;
    }
}
