using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>Exposes registered in-process tools as one tool source, so they share the execution path of external providers.</summary>
internal sealed class LocalToolProvider : IConversationToolProvider
{
    internal const string LocalServerAlias = "local";

    private readonly Dictionary<string, IConversationTool> _toolsByName = new(StringComparer.Ordinal);
    private readonly IReadOnlyList<ConversationToolDefinition> _definitions;

    public LocalToolProvider(IReadOnlyList<IConversationTool> tools)
    {
        var definitions = new List<ConversationToolDefinition>(tools.Count);
        foreach (IConversationTool tool in tools)
        {
            ConversationToolDescriptor descriptor = tool.Descriptor
                ?? throw new InvalidOperationException("A local tool returned a null descriptor.");
            if (!_toolsByName.TryAdd(descriptor.Name, tool))
                throw new InvalidOperationException($"Local tool '{descriptor.Name}' is already registered.");
            definitions.Add(new ConversationToolDefinition(new ConversationToolId(LocalServerAlias, descriptor.Name), descriptor));
        }
        _definitions = definitions.AsReadOnly();
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.CompletedTask;
    }

    public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(_definitions);
    }

    public ValueTask<ConversationToolResult> InvokeAsync(ConversationToolCall call, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(call);
        return _toolsByName.TryGetValue(call.ToolId.ToolName, out IConversationTool? tool)
            ? tool.ExecuteAsync(call, cancellationToken)
            : ValueTask.FromResult(ConversationToolResult.Failure(ConversationToolResultKind.ProtocolError, "unknown_local_tool"));
    }

    public Task StopAsync() => Task.CompletedTask;

    // The tools themselves are owned and disposed by the tool runtime.
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
