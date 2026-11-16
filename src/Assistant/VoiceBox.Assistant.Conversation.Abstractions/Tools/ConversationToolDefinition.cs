namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 一条工具目录项, 由工具来源向消费会话暴露。
/// </summary>
/// <param name="Id">工具在会话内的稳定身份。</param>
/// <param name="Descriptor">宿主受信任元数据与模型可见 schema。</param>
public sealed record ConversationToolDefinition(ConversationToolId Id, ConversationToolDescriptor Descriptor)
{
    public ConversationToolId Id { get; } = ValidateId(Id);

    public ConversationToolDescriptor Descriptor { get; } = Descriptor ?? throw new ArgumentNullException(nameof(Descriptor));

    private static ConversationToolId ValidateId(ConversationToolId id)
    {
        if (string.IsNullOrWhiteSpace(id.ServerAlias) || string.IsNullOrWhiteSpace(id.ToolName))
            throw new ArgumentException("Tool identity must contain a server alias and tool name.", nameof(id));
        return id;
    }
}
