using System;
using System.Collections.Generic;
using System.Text;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Llama 3/3.x header 模板。</summary>
    public sealed class Llama3LlamaModelProfile : DefaultLlamaModelProfile
    {
        private static readonly string[] Strip =
        ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"];

        public override string Name => "llama3";

        public override bool Matches(LlamaModelInfo info) =>
            Contains(info.ChatTemplate, "<|start_header_id|>") ||
            Contains(info.Name, "Llama 3") || Contains(info.Name, "Llama-3");

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
                prompt.Append("<|start_header_id|>").Append(message.Role)
                    .Append("<|end_header_id|>\n\n").Append(message.Text.Trim())
                    .Append("<|eot_id|>");
            }
            if (addGenerationPrompt)
                prompt.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            return prompt.ToString();
        }

        public override IReadOnlyList<string> StripMarkers => Strip;
    }
}
