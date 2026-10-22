using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>
    /// 把 <see cref="ChatMessage"/> 列表渲染为 Gemma chat template 字符串。
    /// 依赖模型目录下的 <c>chat_template.jinja</c> + ORT-GenAI Tokenizer 自带的 jinja 渲染。
    ///
    /// 多模态内容在模板渲染前会被替换为 placeholder token (例如 <c>&lt;|audio|&gt;</c>),
    /// 由 <see cref="GenAIInputBuilder"/> 决定如何配对实际的音频/图像数据。
    /// </summary>
    internal sealed class GenAIPromptTemplate
    {
        private const string AudioPlaceholderToken = "<|audio|>";
        private const string ImagePlaceholderToken = "<|image|>";

        private readonly string _templateBody;

        public GenAIPromptTemplate(string modelPath)
        {
            string templatePath = Path.Combine(modelPath, "chat_template.jinja");
            _templateBody = File.Exists(templatePath)
                ? File.ReadAllText(templatePath, Encoding.UTF8)
                : string.Empty;
        }

        /// <summary>
        /// 把消息列表渲染为 prompt 字符串, 并返回消息中按顺序出现的音频/图像数量,
        /// 以便 InputBuilder 校验数据条数与 placeholder token 数量是否一致。
        /// </summary>
        public RenderResult Render(
            Tokenizer tokenizer,
            IReadOnlyList<ChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking)
        {
            int audioCount = 0;
            int imageCount = 0;

            var jsonMessages = new List<GenAIChatTemplateMessage>(messages.Count);
            foreach (var msg in messages)
            {
                string content = BuildContent(msg.Content, ref audioCount, ref imageCount);
                jsonMessages.Add(new GenAIChatTemplateMessage(RoleString(msg.Role), content));
            }

            // 用 source-generated 元数据序列化 (避免反射式 JsonSerializer, 利于裁剪/AOT)。
            string messagesJson = JsonSerializer.Serialize(jsonMessages, GenAIChatJsonContext.Default.ListGenAIChatTemplateMessage);

            // 思考模式: Gemma 4 chat template 通过 tools 段的 "enable_thinking" 标志位控制,
            // 这里以约定字段透传; 模板若不识别会被忽略, 不破坏现有流程。
            string toolsJson = enableThinking
                ? "{\"enable_thinking\":true}"
                : string.Empty;

            string prompt = tokenizer.ApplyChatTemplate(
                template_str: _templateBody,
                messages: messagesJson,
                tools: toolsJson,
                add_generation_prompt: addGenerationPrompt);

            return new RenderResult(prompt, audioCount, imageCount);
        }

        internal readonly record struct RenderResult(string Prompt, int AudioCount, int ImageCount);

        private static string BuildContent(IReadOnlyList<ChatContent> content, ref int audioCount, ref int imageCount)
        {
            if (content.Count == 1 && content[0] is TextContent singleText)
                return singleText.Text;

            var sb = new StringBuilder();
            foreach (var item in content)
            {
                switch (item)
                {
                    case TextContent text:
                        sb.Append(text.Text);
                        break;
                    case AudioContent:
                        sb.Append(AudioPlaceholderToken);
                        audioCount++;
                        break;
                    case ImageContent:
                        sb.Append(ImagePlaceholderToken);
                        imageCount++;
                        break;
                    case ToolCallContent:
                    case ToolResultContent:
                        throw new NotSupportedException(
                            "VoiceBox.LLM.OnnxGenAI does not support structured tool messages.");
                    default:
                        throw new NotSupportedException(
                            $"Unsupported chat content type: {item.GetType().Name}.");
                }
            }

            return sb.ToString();
        }

        private static string RoleString(ChatRole role) => role switch
        {
            ChatRole.System => "system",
            ChatRole.User => "user",
            ChatRole.Assistant => "assistant",
            ChatRole.Tool => throw new NotSupportedException(
                "VoiceBox.LLM.OnnxGenAI does not support structured tool messages."),
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
        };
    }

    internal sealed class GenAIChatTemplateMessage
    {
        public GenAIChatTemplateMessage(string role, string content)
        {
            Role = role;
            Content = content;
        }

        [JsonPropertyName("role")]
        public string Role { get; }

        [JsonPropertyName("content")]
        public string Content { get; }
    }

    /// <summary>
    /// 渲染 chat template 时传给 tokenizer 的 messages JSON 的 source-generated 序列化上下文。
    /// </summary>
    [JsonSerializable(typeof(List<GenAIChatTemplateMessage>))]
    internal sealed partial class GenAIChatJsonContext : JsonSerializerContext
    {
    }
}
