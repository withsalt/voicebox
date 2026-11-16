using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Tests.Support;

/// <summary>A local tool whose behaviour is supplied by the test.</summary>
internal sealed class ScriptedTool : IConversationTool
{
    private const string AnyObjectSchema =
        """
        {
          "type": "object",
          "properties": {},
          "additionalProperties": true
        }
        """;

    private int _executeCount;

    public ScriptedTool(
        string name,
        ConversationToolRisk risk = ConversationToolRisk.ReadOnly,
        ConversationToolExecutionPolicy policy = ConversationToolExecutionPolicy.ForegroundOnly,
        TimeSpan? timeout = null,
        string? displayName = null,
        string? executionPrompt = null,
        string? inputSchemaJson = null)
    {
        Descriptor = new ConversationToolDescriptor(
            name,
            displayName ?? name,
            name,
            inputSchemaJson ?? AnyObjectSchema,
            risk: risk,
            executionPolicy: policy,
            executionPrompt: executionPrompt,
            timeout: timeout ?? TimeSpan.FromSeconds(5));
    }

    public ConversationToolDescriptor Descriptor { get; }

    public int ExecuteCount => Volatile.Read(ref _executeCount);

    /// <summary>Runs on every call. The default succeeds with "ok".</summary>
    public Func<ConversationToolCall, CancellationToken, Task<ConversationToolResult>> OnExecute { get; set; } =
        static (_, _) => Task.FromResult(ConversationToolResult.Success("ok"));

    public Action? OnDispose { get; set; }

    public ValueTask<ConversationToolResult> ExecuteAsync(ConversationToolCall call, CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _executeCount);
        return new ValueTask<ConversationToolResult>(OnExecute(call, cancellationToken));
    }

    public void Dispose() => OnDispose?.Invoke();
}

internal sealed class AllowAllApprovalPolicy : IConversationToolApprovalPolicy
{
    public ValueTask<ConversationToolApproval> AuthorizeAsync(ConversationToolApprovalContext context, CancellationToken cancellationToken)
        => ValueTask.FromResult(ConversationToolApproval.Allow);
}
