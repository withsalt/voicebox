using System.Collections.ObjectModel;

namespace VoiceBox.Core.Services.LLMService.Models;

/// <summary>一次支持结构化工具调用的 provider 请求。</summary>
public sealed class LlmToolTurnRequest
{
    public LlmToolTurnRequest(
        IEnumerable<ChatMessage> messages,
        IEnumerable<LlmToolDefinition> tools,
        LlmCompleteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(tools);
        Messages = Array.AsReadOnly(messages.ToArray());
        Tools = Array.AsReadOnly(tools.ToArray());
        Options = options;
    }

    public ReadOnlyCollection<ChatMessage> Messages { get; }

    public ReadOnlyCollection<LlmToolDefinition> Tools { get; }

    public LlmCompleteOptions? Options { get; }
}
