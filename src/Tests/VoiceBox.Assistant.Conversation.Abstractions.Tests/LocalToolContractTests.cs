using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tests;

public sealed class LocalToolContractTests
{
    [Fact]
    public async Task AThirdPartyCanConstructAPublicInvocationAndCallALocalTool()
    {
        var tool = new EchoTool();
        ConversationToolInvocation invocation;
        using (JsonDocument document = JsonDocument.Parse("""{"text":"hello"}"""))
        {
            invocation = new ConversationToolInvocation("call-1", document.RootElement, ConversationToolExecutionPreference.Background);
        }

        // The JsonDocument has been disposed; the invocation must still own its arguments.
        ConversationToolExecutionResult result = await tool.ExecuteAsync(invocation, CancellationToken.None);

        Assert.True(result.IsSuccess);
        Assert.True(invocation.Arguments.TryGetProperty("text", out JsonElement text));
        Assert.Equal("hello", text.GetString());
        Assert.Equal("echo:hello", result.Content);
        Assert.Contains("hello", result.StructuredContentJson, StringComparison.Ordinal);
        Assert.Equal(ConversationToolExecutionPreference.Background, invocation.ExecutionPreference);
    }

    [Fact]
    public void InvalidInvocationInputsAreRejected()
    {
        using JsonDocument objectDocument = JsonDocument.Parse("""{"a":1}""");
        using JsonDocument arrayDocument = JsonDocument.Parse("[]");
        using JsonDocument stringDocument = JsonDocument.Parse("\"text\"");

        Assert.Throws<ArgumentException>(() => new ConversationToolInvocation(
            " ", objectDocument.RootElement, ConversationToolExecutionPreference.Foreground));
        Assert.Throws<ArgumentException>(() => new ConversationToolInvocation(
            "call", arrayDocument.RootElement, ConversationToolExecutionPreference.Foreground));
        Assert.Throws<ArgumentException>(() => new ConversationToolInvocation(
            "call", stringDocument.RootElement, ConversationToolExecutionPreference.Foreground));
        Assert.Throws<ArgumentException>(() => new ConversationToolInvocation(
            "call", default, ConversationToolExecutionPreference.Foreground));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationToolInvocation(
            "call", objectDocument.RootElement, (ConversationToolExecutionPreference)99));
    }

    [Fact]
    public async Task ToolProviderAndApprovalPolicyCanBeImplementedUsingOnlyThePublicContracts()
    {
        var provider = new EchoProvider();
        await provider.StartAsync();
        try
        {
            IReadOnlyList<ConversationToolDefinition> tools = await provider.GetToolsAsync();
            ConversationToolDefinition definition = Assert.Single(tools);
            Assert.Equal("fake", definition.Id.ServerAlias);

            var call = new ConversationToolCall("call-1", definition.Id, """{"text":"world"}""");
            ConversationToolResult result = await provider.InvokeAsync(call);
            Assert.Equal(ConversationToolResultKind.Success, result.Kind);
            Assert.Contains("world", result.ValueJson, StringComparison.Ordinal);

            using JsonDocument document = JsonDocument.Parse(call.ArgumentsJson);
            var invocation = new ConversationToolInvocation(call.CallId, document.RootElement, call.ExecutionPreference);
            var approval = new AllowAllPolicy();
            ConversationToolApproval decision = await approval.AuthorizeAsync(
                new ConversationToolApprovalContext(definition.Id.ServerAlias, definition.Descriptor, invocation, "hash"),
                CancellationToken.None);
            Assert.Equal(ConversationToolApproval.Allow, decision);
        }
        finally
        {
            await provider.StopAsync();
        }
    }

    [Fact]
    public void ToolCallAndResultValidateTheirEnums()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationToolCall(
            "call",
            new ConversationToolId("srv", "tool"),
            "{}",
            (ConversationToolExecutionPreference)99));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationToolResult("call", (ConversationToolResultKind)99));
    }

    private sealed class EchoTool : IConversationTool
    {
        public ConversationToolDescriptor Descriptor { get; } = new(
            "echo",
            "Echo",
            "Echo text",
            """
            {
              "type": "object",
              "properties": { "text": { "type": "string" } },
              "required": ["text"]
            }
            """,
            executionPolicy: ConversationToolExecutionPolicy.ForegroundOnly);

        public ValueTask<ConversationToolExecutionResult> ExecuteAsync(
            ConversationToolInvocation invocation,
            CancellationToken cancellationToken)
        {
            string text = invocation.Arguments.GetProperty("text").GetString() ?? string.Empty;
            return ValueTask.FromResult(ConversationToolExecutionResult.Success(
                $"echo:{text}",
                JsonSerializer.Serialize(new { echo = text })));
        }

        public void Dispose()
        {
        }
    }

    private sealed class EchoProvider : IConversationToolProvider, IConversationToolProviderMetadata
    {
        private bool _started;

        public IReadOnlyList<string> ServerAliases { get; } = ["fake"];

        public Task StartAsync(CancellationToken cancellationToken = default)
        {
            _started = true;
            return Task.CompletedTask;
        }

        public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(CancellationToken cancellationToken = default)
        {
            Assert.True(_started);
            return ValueTask.FromResult<IReadOnlyList<ConversationToolDefinition>>(
            [
                new ConversationToolDefinition(
                    new ConversationToolId("fake", "echo"),
                    new ConversationToolDescriptor(
                        "echo",
                        "Echo",
                        "Echo text",
                        """
                        {
                          "type": "object",
                          "properties": { "text": { "type": "string" } },
                          "required": ["text"]
                        }
                        """,
                        executionPolicy: ConversationToolExecutionPolicy.ForegroundOnly)),
            ]);
        }

        public ValueTask<ConversationToolResult> InvokeAsync(ConversationToolCall call, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new ConversationToolResult(call.CallId, ConversationToolResultKind.Success, call.ArgumentsJson));

        public Task StopAsync()
        {
            _started = false;
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class AllowAllPolicy : IConversationToolApprovalPolicy
    {
        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
            => ValueTask.FromResult(ConversationToolApproval.Allow);
    }
}
