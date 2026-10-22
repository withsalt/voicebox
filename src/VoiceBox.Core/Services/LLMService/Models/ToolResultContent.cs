namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>与模型工具调用 ID 关联的结构化工具结果。</summary>
    public sealed class ToolResultContent : ChatContent
    {
        public ToolResultContent(string callId, string resultJson, bool isError = false)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(callId);
            CallId = callId;
            ResultJson = resultJson ?? string.Empty;
            IsError = isError;
        }

        public string CallId { get; }

        public string ResultJson { get; }

        public bool IsError { get; }
    }
}
