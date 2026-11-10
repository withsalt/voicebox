namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>审批策略看到的调用上下文。</summary>
public sealed record ConversationToolApprovalContext(
    string ServerAlias,
    ConversationToolDescriptor Tool,
    ConversationToolInvocation Invocation,
    string ArgumentsHash)
{
    public string ServerAlias { get; } = string.IsNullOrWhiteSpace(ServerAlias)
        ? throw new ArgumentException("Server alias cannot be empty.", nameof(ServerAlias))
        : ServerAlias;

    public ConversationToolDescriptor Tool { get; } = Tool ?? throw new ArgumentNullException(nameof(Tool));

    public ConversationToolInvocation Invocation { get; } = Invocation ?? throw new ArgumentNullException(nameof(Invocation));

    public string ArgumentsHash { get; } = string.IsNullOrWhiteSpace(ArgumentsHash)
        ? throw new ArgumentException("Arguments hash cannot be empty.", nameof(ArgumentsHash))
        : ArgumentsHash;
}
