using System;
using System.Collections.Generic;
using System.Text;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>ChatML 模板，覆盖 Qwen、部分 DeepSeek/MiniCPM 派生模型。</summary>
    public class ChatMlLlamaModelProfile : DefaultLlamaModelProfile
    {
        public override string Name => "chatml";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.StartsWith("qwen", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.ChatTemplate, "<|im_start|>");

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
                prompt.Append("<|im_start|>").Append(message.Role).Append('\n')
                    .Append(message.Text).Append("<|im_end|>\n");
            }
            if (addGenerationPrompt)
                prompt.Append("<|im_start|>assistant\n");
            return prompt.ToString();
        }
    }
}
