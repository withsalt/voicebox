using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using OpenAI.Chat;
using VoiceBox.Core.Services.LLMService.Models;
using OAIChatMessage = OpenAI.Chat.ChatMessage;
using ChatMessage = VoiceBox.Core.Services.LLMService.Models.ChatMessage;

namespace VoiceBox.LLM.OpenAI.Internal
{
    /// <summary>
    /// 把 VoiceBox 的 <see cref="ChatMessage"/> 列表翻译为 OpenAI SDK 的
    /// <see cref="OAIChatMessage"/> 列表。
    ///
    /// 行为约定:
    /// <list type="bullet">
    ///   <item>多个 <see cref="TextContent"/> 按出现顺序拼接。</item>
    ///   <item><see cref="ImageContent"/> 仅在 <c>user</c> 消息上转为图片 part (要求模型本身是
    ///         vision 模型); 出现在 system/assistant 上时按文本处理 (忽略图片)。</item>
    ///   <item><see cref="AudioContent"/> 在标准 <c>/chat/completions</c> 文本协议下没有通用映射,
    ///         遇到时抛 <see cref="NotSupportedException"/> —— 明确失败比沉默丢弃更安全。</item>
    /// </list>
    /// </summary>
    internal static class OpenAIMessageConverter
    {
        public static List<OAIChatMessage> Convert(IReadOnlyList<ChatMessage> messages)
        {
            var result = new List<OAIChatMessage>(messages.Count);
            foreach (var msg in messages)
            {
                result.Add(ConvertOne(msg));
            }
            return result;
        }

        private static OAIChatMessage ConvertOne(ChatMessage msg)
        {
            ToolCallContent[] toolCalls = msg.Content.OfType<ToolCallContent>().ToArray();
            ToolResultContent[] toolResults = msg.Content.OfType<ToolResultContent>().ToArray();
            if (toolCalls.Length > 0 || toolResults.Length > 0)
            {
                if (msg.Role == ChatRole.Assistant && toolCalls.Length == msg.Content.Count)
                {
                    return new AssistantChatMessage(toolCalls.Select(static call =>
                        ChatToolCall.CreateFunctionToolCall(
                            call.CallId,
                            call.ToolName,
                            BinaryData.FromString(call.ArgumentsJson))));
                }

                if (msg.Role == ChatRole.Tool && toolResults.Length == 1 && msg.Content.Count == 1)
                {
                    ToolResultContent result = toolResults[0];
                    return new ToolChatMessage(result.CallId, result.ResultJson);
                }

                throw new NotSupportedException(
                    "OpenAI tool messages must contain only assistant tool calls or one tool result.");
            }

            // 按原始顺序收集 parts (多模态时文本与图片的相对顺序对模型有意义),
            // 同时累积纯文本; hasImage 用于决定走 part 数组还是更通用的纯字符串形式。
            var textSb = new StringBuilder();
            var parts = new List<ChatMessageContentPart>(msg.Content.Count);
            bool hasImage = false;

            foreach (var item in msg.Content)
            {
                switch (item)
                {
                    case TextContent text:
                        textSb.Append(text.Text);
                        parts.Add(ChatMessageContentPart.CreateTextPart(text.Text));
                        break;

                    case ImageContent image:
                        hasImage = true;
                        parts.Add(CreateImagePart(image));
                        break;

                    case AudioContent:
                        throw new NotSupportedException(
                            "标准 OpenAI /chat/completions 文本协议不支持音频输入。" +
                            "如需语音转写, 请改用专门的 ASR 服务 (VoiceBox.ASR.*), " +
                            "或换用本地的 VoiceBox.LLM.OnnxGenAI 多模态实现。");
                }
            }

            return msg.Role switch
            {
                // system / assistant 只取文本 —— 这两类角色不携带图片。
                ChatRole.System => new SystemChatMessage(textSb.ToString()),
                ChatRole.Assistant => new AssistantChatMessage(textSb.ToString()),

                // 含图片时按多模态 part 数组发送; 纯文本则用更通用的字符串形式 (兼容性更好、负载更小)。
                ChatRole.User => hasImage
                    ? new UserChatMessage(parts)
                    : new UserChatMessage(textSb.ToString()),
                ChatRole.Tool => throw new NotSupportedException(
                    "A tool-role message must contain one ToolResultContent value."),
                _ => throw new ArgumentOutOfRangeException(nameof(msg), msg.Role, "Unsupported chat role."),
            };
        }

        private static ChatMessageContentPart CreateImagePart(ImageContent image)
        {
            if (image.Bytes is { Length: > 0 } bytes)
            {
                return ChatMessageContentPart.CreateImagePart(
                    BinaryData.FromBytes(bytes), GuessImageMediaType(bytes, image.Path));
            }

            if (!string.IsNullOrEmpty(image.Path))
            {
                if (!File.Exists(image.Path))
                    throw new FileNotFoundException($"Image file not found: {image.Path}", image.Path);
                var fileBytes = File.ReadAllBytes(image.Path);
                return ChatMessageContentPart.CreateImagePart(
                    BinaryData.FromBytes(fileBytes), GuessImageMediaType(fileBytes, image.Path));
            }

            throw new ArgumentException("ImageContent must have either Path or Bytes set.");
        }

        /// <summary>
        /// 推断图片 MIME。先看魔数 (最可靠), 再看扩展名, 最后兜底 image/jpeg。
        /// </summary>
        private static string GuessImageMediaType(byte[] bytes, string? path)
        {
            if (bytes.Length >= 8 &&
                bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47)
                return "image/png";

            if (bytes.Length >= 3 &&
                bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF)
                return "image/jpeg";

            if (bytes.Length >= 6 &&
                bytes[0] == 0x47 && bytes[1] == 0x49 && bytes[2] == 0x46) // GIF
                return "image/gif";

            if (bytes.Length >= 12 &&
                bytes[0] == 0x52 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x46 && // RIFF
                bytes[8] == 0x57 && bytes[9] == 0x45 && bytes[10] == 0x42 && bytes[11] == 0x50) // WEBP
                return "image/webp";

            string ext = string.IsNullOrEmpty(path) ? string.Empty : Path.GetExtension(path).ToLowerInvariant();
            return ext switch
            {
                ".png" => "image/png",
                ".gif" => "image/gif",
                ".webp" => "image/webp",
                ".bmp" => "image/bmp",
                _ => "image/jpeg",
            };
        }
    }
}
