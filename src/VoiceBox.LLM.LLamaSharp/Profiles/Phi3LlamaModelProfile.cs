using System;
using System.Collections.Generic;
using System.Text;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Phi-3 role/end 模板。</summary>
    public sealed class Phi3LlamaModelProfile : DefaultLlamaModelProfile
    {
        private static readonly string[] Strip =
        ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"];

        public override string Name => "phi3";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.StartsWith("phi3", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.Name, "Phi-3") ||
            (Contains(info.ChatTemplate, "<|system|>") &&
             Contains(info.ChatTemplate, "<|end|>"));

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
                prompt.Append("<|").Append(message.Role).Append("|>\n")
                    .Append(message.Text.Trim()).Append("<|end|>\n");
            }
            if (addGenerationPrompt)
                prompt.Append("<|assistant|>\n");
            return prompt.ToString();
        }

        public override IReadOnlyList<string> StripMarkers => Strip;
    }
}
