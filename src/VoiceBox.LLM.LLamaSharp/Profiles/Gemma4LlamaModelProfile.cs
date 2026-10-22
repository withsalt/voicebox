using System;
using System.Collections.Generic;
using System.Text;
using LLama;
using static VoiceBox.LLM.LLamaSharp.LlamaProfileText;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>Gemma 4 turn、thinking 与 channel 模板。</summary>
    public sealed class Gemma4LlamaModelProfile : DefaultLlamaModelProfile
    {
        private const string ChannelStart = "<|channel>";
        private const string ChannelEnd = "<channel|>";

        private static readonly string[] ThinkStart = ["<|channel>thought\n"];
        private static readonly string[] ThinkEnd = [ChannelEnd];
        private static readonly string[] Strip = ["<|turn>", "<turn|>", "<|think|>"];

        public override string Name => "gemma4";

        public override bool Matches(LlamaModelInfo info) =>
            info.Architecture.Equals("gemma4", StringComparison.OrdinalIgnoreCase) ||
            Contains(info.ChatTemplate, "Google Gemma 4") ||
            Contains(info.ChatTemplate, "<|turn>");

        public override string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            ArgumentNullException.ThrowIfNull(messages);
            var prompt = new StringBuilder();

            int messageStart = 0;
            bool hasLeadingSystem = messages.Count > 0 &&
                messages[0].Role is "system" or "developer";
            if (enableThinking || hasLeadingSystem)
            {
                prompt.Append("<|turn>system\n");
                if (enableThinking)
                    prompt.Append("<|think|>\n");
                if (hasLeadingSystem)
                {
                    prompt.Append(messages[0].Text.Trim());
                    messageStart = 1;
                }
                prompt.Append("<turn|>\n");
            }

            for (int i = messageStart; i < messages.Count; i++)
            {
                LlamaChatMessage message = messages[i];
                string role = message.Role == "assistant" ? "model" : message.Role;
                string text = role == "model"
                    ? StripThinkingChannels(message.Text)
                    : message.Text.Trim();
                prompt.Append("<|turn>").Append(role).Append('\n')
                    .Append(text).Append("<turn|>\n");
            }
            if (addGenerationPrompt)
                prompt.Append("<|turn>model\n");
            return prompt.ToString();
        }

        private static string StripThinkingChannels(string text)
        {
            string remaining = text.Trim();
            var visible = new StringBuilder(remaining.Length);
            int offset = 0;
            while (offset < remaining.Length)
            {
                int start = remaining.IndexOf(ChannelStart, offset, StringComparison.Ordinal);
                if (start < 0)
                {
                    visible.Append(remaining, offset, remaining.Length - offset);
                    break;
                }

                visible.Append(remaining, offset, start - offset);
                int end = remaining.IndexOf(
                    ChannelEnd, start + ChannelStart.Length, StringComparison.Ordinal);
                if (end < 0)
                    break;
                offset = end + ChannelEnd.Length;
            }
            return visible.ToString().Trim();
        }

        public override IReadOnlyList<string> ThinkingStartMarkers => ThinkStart;
        public override IReadOnlyList<string> ThinkingEndMarkers => ThinkEnd;
        public override IReadOnlyList<string> StripMarkers => Strip;
    }
}
