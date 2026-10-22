using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using LLama;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Gemma 系列 <c>start_of_turn</c> 模板。</summary>
    public class GemmaLlamaModelProfile : DefaultLlamaModelProfile
    {
        private static readonly string[] ThinkStart = ["<think>", "<thought>"];
        private static readonly string[] ThinkEnd = ["</think>", "</thought>"];
        private static readonly string[] Strip =
        [
            "<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>",
        ];

        public override string Name => "gemma";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.StartsWith("gemma", StringComparison.OrdinalIgnoreCase) ||
            (info.ChatTemplate?.Contains(
                "<start_of_turn>", StringComparison.OrdinalIgnoreCase) ?? false);

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(messages);

            var prompt = new StringBuilder();
            string? system = messages.FirstOrDefault(message => message.Role == "system").Text;
            foreach (var message in messages)
            {
                if (message.Role == "system")
                    continue;

                string role = message.Role == "assistant" ? "model" : "user";
                prompt.Append("<start_of_turn>").Append(role).Append('\n');
                if (role == "user" && !string.IsNullOrWhiteSpace(system))
                {
                    prompt.Append(system.Trim()).Append("\n\n");
                    system = null;
                }
                prompt.Append(message.Text.Trim()).Append("<end_of_turn>\n");
            }

            if (addGenerationPrompt)
                prompt.Append("<start_of_turn>model\n");
            return prompt.ToString();
        }

        public override IReadOnlyList<string> ThinkingStartMarkers => ThinkStart;
        public override IReadOnlyList<string> ThinkingEndMarkers => ThinkEnd;
        public override IReadOnlyList<string> StripMarkers => Strip;
    }
}
