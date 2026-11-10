using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>Adapts registered in-process tools onto the same execution path as external providers.</summary>
internal sealed class LocalToolProvider : IConversationToolProvider
{
    internal const string LocalServerAlias = "local";

    private readonly IReadOnlyList<IConversationTool> _tools;
    private readonly Dictionary<string, IConversationTool> _toolsByName = new(StringComparer.Ordinal);
    private IReadOnlyList<ConversationToolDefinition> _definitions = Array.Empty<ConversationToolDefinition>();
    private int _disposed;

    public LocalToolProvider(IReadOnlyList<IConversationTool> tools)
    {
        _tools = tools.ToArray();
        foreach (IConversationTool tool in _tools)
        {
            ConversationToolDescriptor descriptor = tool.Descriptor
                ?? throw new InvalidOperationException("A local tool returned a null descriptor.");
            if (_toolsByName.TryAdd(descriptor.Name, tool))
                continue;
            throw new InvalidOperationException($"Local tool '{descriptor.Name}' is already registered.");
        }
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        _definitions = Array.AsReadOnly(_tools
            .Select(tool => new ConversationToolDefinition(new ConversationToolId(LocalServerAlias, tool.Descriptor.Name), tool.Descriptor))
            .ToArray());
        return Task.CompletedTask;
    }

    public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(_definitions);
    }

    public async ValueTask<ConversationToolResult> InvokeAsync(ConversationToolCall call, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_toolsByName.TryGetValue(call.ToolId.ToolName, out IConversationTool? tool))
        {
            return new ConversationToolResult(call.CallId, ConversationToolResultKind.ProtocolError, ErrorCode: "unknown_local_tool");
        }

        using JsonDocument document = JsonDocument.Parse(string.IsNullOrWhiteSpace(call.ArgumentsJson) ? "{}" : call.ArgumentsJson);
        ConversationToolExecutionResult result = await tool.ExecuteAsync(
            new ConversationToolInvocation(call.CallId, document.RootElement, call.ExecutionPreference),
            cancellationToken).ConfigureAwait(false);

        if (!string.IsNullOrWhiteSpace(result.StructuredContentJson))
        {
            return new ConversationToolResult(
                call.CallId,
                result.IsSuccess
                    ? ConversationToolResultKind.Success
                    : ConversationToolResultKind.ToolError,
                result.StructuredContentJson,
                result.ErrorCode);
        }

        return new ConversationToolResult(
            call.CallId,
            result.IsSuccess
                ? ConversationToolResultKind.Success
                : ConversationToolResultKind.ToolError,
            JsonSerializer.Serialize(new { content = result.Content }),
            result.ErrorCode);
    }

    public Task StopAsync() => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        Interlocked.Exchange(ref _disposed, 1);
        _definitions = Array.Empty<ConversationToolDefinition>();
        return ValueTask.CompletedTask;
    }
}
