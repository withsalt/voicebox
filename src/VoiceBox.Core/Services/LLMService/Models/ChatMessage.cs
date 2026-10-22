namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 一条对话消息。<see cref="Content"/> 可以是纯文本 (单 <see cref="TextContent"/>),
    /// 也可以是文本与音频/图像混合的多片段。
    /// </summary>
    public sealed class ChatMessage(ChatRole role, IReadOnlyList<ChatContent> content)
    {
        public ChatRole Role { get; } = role;

        public IReadOnlyList<ChatContent> Content { get; } = content ?? [];

        public static ChatMessage System(string text) =>
            new(ChatRole.System, [new TextContent(text)]);

        public static ChatMessage User(string text) =>
            new(ChatRole.User, [new TextContent(text)]);

        public static ChatMessage User(params ChatContent[] content) =>
            new(ChatRole.User, content);

        public static ChatMessage Assistant(string text) =>
            new(ChatRole.Assistant, [new TextContent(text)]);

        public static ChatMessage AssistantToolCalls(params ToolCallContent[] calls) =>
            new(ChatRole.Assistant, calls);

        public static ChatMessage ToolResult(
            string callId,
            string resultJson,
            bool isError = false) =>
            new(ChatRole.Tool, [new ToolResultContent(callId, resultJson, isError)]);
    }
}
