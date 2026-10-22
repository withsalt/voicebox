using System;
using System.Collections.Generic;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    /// <summary>
    /// 把 <see cref="ChatMessage"/> 列表扁平化为 (role, text) 序列并交给模型 profile 渲染。
    /// 多模态内容 (音频/图像) 在 v1 明确不支持 — llama.cpp 侧需要 mmproj + mtmd 链路,
    /// 留待后续版本; 这里直接抛出清晰的 <see cref="NotSupportedException"/>。
    /// </summary>
    internal static class LlamaPromptBuilder
    {
        public static string Build(
            LlamaModelSession session,
            IReadOnlyList<ChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            var flat = Flatten(messages);
            string prompt = session.Profile.BuildPrompt(session.Weights, flat, addGenerationPrompt, enableThinking);
            if (string.IsNullOrEmpty(prompt))
            {
                throw new InvalidOperationException(
                    $"Model profile '{session.Profile.Name}' rendered an empty prompt.");
            }
            return prompt;
        }

        /// <summary>把每条消息的多片段内容压平为纯文本; 遇到音频/图像片段直接拒绝。</summary>
        internal static List<LlamaChatMessage> Flatten(IReadOnlyList<ChatMessage> messages)
        {
            var flat = new List<LlamaChatMessage>(messages.Count);
            foreach (var message in messages)
            {
                flat.Add(new LlamaChatMessage(
                    RoleString(message.Role),
                    LlamaChatContentFormatter.Format(message)));
            }
            return flat;
        }

        private static string RoleString(ChatRole role) => role switch
        {
            ChatRole.System => "system",
            ChatRole.User => "user",
            ChatRole.Assistant => "assistant",
            ChatRole.Tool => "tool",
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
        };
    }
}
