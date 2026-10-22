using System;
using System.Collections.Generic;
using System.Text;
using LLama;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>MiniCPM5 ChatML 与 thinking 预填适配。</summary>
    public class MiniCpm5LlamaModelProfile : DefaultLlamaModelProfile
    {
        private static readonly string[] Strip =
        [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        ];

        public override string Name => "minicpm5";

        public override bool Matches(LlamaModelInfo info) =>
            info.Name.Contains("MiniCPM5", StringComparison.OrdinalIgnoreCase) ||
            info.Metadata.GetValueOrDefault("general.basename", string.Empty)
                .Contains("MiniCPM5", StringComparison.OrdinalIgnoreCase);

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(messages);

            var prompt = new StringBuilder("<s>");
            foreach (var message in messages)
            {
                prompt.Append("<|im_start|>")
                    .Append(message.Role)
                    .Append('\n')
                    .Append(message.Text)
                    .Append("<|im_end|>\n");
            }

            if (addGenerationPrompt)
            {
                prompt.Append("<|im_start|>assistant\n<think>\n");
                if (!enableThinking)
                    prompt.Append("</think>\n\n");
            }

            return prompt.ToString();
        }

        public override IReadOnlyList<string> StripMarkers => Strip;
        public override bool StartsInsideThinking(bool enableThinking) => enableThinking;
    }
}
