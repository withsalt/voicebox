using System.Text;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp.Internal;

internal static class LlamaToolPromptBuilder
{
    public static IReadOnlyList<ChatMessage> AddToolInstructions(
        IReadOnlyList<ChatMessage> messages,
        LlamaToolCallCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(catalog);

        var systemText = new StringBuilder();
        var conversation = new List<ChatMessage>(messages.Count + 1);
        foreach (ChatMessage message in messages)
        {
            if (message.Role != ChatRole.System)
            {
                conversation.Add(message);
                continue;
            }

            string text = LlamaChatContentFormatter.Format(message);
            if (string.IsNullOrWhiteSpace(text))
                continue;
            if (systemText.Length > 0)
                systemText.AppendLine().AppendLine();
            systemText.Append(text);
        }

        string instructions = catalog.SystemPrompt;
        if (systemText.Length > 0)
            systemText.AppendLine().AppendLine();
        systemText.Append(instructions);
        conversation.Insert(0, ChatMessage.System(systemText.ToString()));
        return conversation.AsReadOnly();
    }
}
