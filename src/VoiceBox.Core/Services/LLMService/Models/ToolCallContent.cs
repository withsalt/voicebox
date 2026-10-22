namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>模型生成的一次结构化工具调用。</summary>
    public sealed class ToolCallContent : ChatContent
    {
        public ToolCallContent(string callId, string toolName, string argumentsJson)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(callId);
            ArgumentException.ThrowIfNullOrWhiteSpace(toolName);
            CallId = callId;
            ToolName = toolName;
            ArgumentsJson = string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson;
        }

        public string CallId { get; }

        public string ToolName { get; }

        public string ArgumentsJson { get; }
    }
}
