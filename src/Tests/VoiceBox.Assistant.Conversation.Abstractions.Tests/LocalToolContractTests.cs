using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tests;

public sealed class LocalToolContractTests
{
    private static readonly ConversationToolId EchoId = new("local", "echo");

    [Fact]
    public async Task AThirdPartyCanConstructAPublicCallAndCallALocalTool()
    {
        var tool = new EchoTool();
        ConversationToolCall call;
        using (JsonDocument document = JsonDocument.Parse("""{"text":"hello"}"""))
        {
            call = new ConversationToolCall("call-1", EchoId, document.RootElement, ConversationToolExecutionPreference.Background);
        }

        // The JsonDocument has been disposed; the call must still own its arguments.
        ConversationToolResult result = await tool.ExecuteAsync(call, CancellationToken.None);

        Assert.True(result.IsSuccess);
        Assert.Equal(ConversationToolResultKind.Success, result.Kind);
        Assert.True(call.Arguments.TryGetProperty("text", out JsonElement text));
        Assert.Equal("hello", text.GetString());
        Assert.Equal("echo:hello", result.Content);
        Assert.Contains("hello", result.StructuredContentJson, StringComparison.Ordinal);
        Assert.Equal(EchoId, call.ToolId);
        Assert.Equal(ConversationToolExecutionPreference.Background, call.ExecutionPreference);
    }

    [Fact]
    public void InvalidCallInputsAreRejected()
    {
        using JsonDocument objectDocument = JsonDocument.Parse("""{"a":1}""");
        using JsonDocument arrayDocument = JsonDocument.Parse("[]");
        using JsonDocument stringDocument = JsonDocument.Parse("\"text\"");

        Assert.Throws<ArgumentException>(() => new ConversationToolCall(" ", EchoId, objectDocument.RootElement));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", default, objectDocument.RootElement));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", EchoId, arrayDocument.RootElement));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", EchoId, stringDocument.RootElement));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", EchoId, default(JsonElement)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationToolCall(
            "call", EchoId, objectDocument.RootElement, (ConversationToolExecutionPreference)99));
    }

    [Fact]
    public void ACallCanBeCreatedFromJsonText()
    {
        var call = new ConversationToolCall("call", EchoId, """{"city":"北京"}""");
        var blank = new ConversationToolCall("call", EchoId, "  ");

        Assert.Equal("北京", call.Arguments.GetProperty("city").GetString());
        Assert.Equal(ConversationToolExecutionPreference.Foreground, call.ExecutionPreference);
        Assert.Equal(JsonValueKind.Object, blank.Arguments.ValueKind);
        Assert.Empty(blank.Arguments.EnumerateObject());

        Assert.Throws<ArgumentNullException>(() => new ConversationToolCall("call", EchoId, (string)null!));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", EchoId, "{not json"));
        Assert.Throws<ArgumentException>(() => new ConversationToolCall("call", EchoId, "[1]"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationToolCall(
            "call", EchoId, "{}", (ConversationToolExecutionPreference)99));
    }

    [Fact]
    public void ResultFactoriesDescribeSuccessAndEveryKindOfFailure()
    {
        ConversationToolResult success = ConversationToolResult.Success("晴", """{"t":31}""");
        Assert.True(success.IsSuccess);
        Assert.Equal("晴", success.Content);
        Assert.Equal("""{"t":31}""", success.StructuredContentJson);
        Assert.Null(success.ErrorCode);
        Assert.False(success.IsTransient);
        Assert.Null(ConversationToolResult.Success("晴").StructuredContentJson);
        Assert.Null(ConversationToolResult.Success("晴", " ").StructuredContentJson);
        Assert.Equal(string.Empty, ConversationToolResult.Success(null!).Content);

        ConversationToolResult error = ConversationToolResult.Error("network", "稍后重试", isTransient: true);
        Assert.False(error.IsSuccess);
        Assert.Equal(ConversationToolResultKind.ToolError, error.Kind);
        Assert.Equal("network", error.ErrorCode);
        Assert.Equal("稍后重试", error.Content);
        Assert.True(error.IsTransient);
        Assert.False(ConversationToolResult.Error("invalid", "参数错误").IsTransient);

        ConversationToolResult timedOut = ConversationToolResult.Failure(ConversationToolResultKind.TimedOut, "timeout");
        Assert.Equal(ConversationToolResultKind.TimedOut, timedOut.Kind);
        Assert.Equal("timeout", timedOut.ErrorCode);
        Assert.Equal(string.Empty, timedOut.Content);
        Assert.Null(timedOut.StructuredContentJson);
        Assert.False(timedOut.IsTransient);
    }

    [Fact]
    public void AFailureNeedsANonSuccessKindAndAnErrorCode()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ConversationToolResult.Failure(ConversationToolResultKind.Success, "code"));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConversationToolResult.Failure((ConversationToolResultKind)99, "code"));
        Assert.Throws<ArgumentException>(() => ConversationToolResult.Failure(ConversationToolResultKind.ToolError, " "));
        Assert.Throws<ArgumentException>(() => ConversationToolResult.Error(string.Empty, "message"));
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

            // The same call type goes to the approval policy and to the provider.
            var call = new ConversationToolCall("call-1", definition.Id, """{"text":"world"}""");
            var approval = new AllowAllPolicy();
            ConversationToolApproval decision = await approval.AuthorizeAsync(
                new ConversationToolApprovalContext(definition.Descriptor, call, "hash"),
                CancellationToken.None);
            Assert.Equal(ConversationToolApproval.Allow, decision);
            Assert.Same(call, approval.Seen!.Call);
            Assert.Equal("fake", approval.Seen.Call.ToolId.ServerAlias);

            ConversationToolResult result = await provider.InvokeAsync(call);
            Assert.Equal(ConversationToolResultKind.Success, result.Kind);
            Assert.Contains("world", result.StructuredContentJson, StringComparison.Ordinal);
        }
        finally
        {
            await provider.StopAsync();
        }
    }

    [Fact]
    public void TheApprovalContextRequiresItsToolCallAndHash()
    {
        ConversationToolDescriptor descriptor = new EchoTool().Descriptor;
        var call = new ConversationToolCall("call", EchoId, "{}");

        Assert.Throws<ArgumentNullException>(() => new ConversationToolApprovalContext(null!, call, "hash"));
        Assert.Throws<ArgumentNullException>(() => new ConversationToolApprovalContext(descriptor, null!, "hash"));
        Assert.Throws<ArgumentException>(() => new ConversationToolApprovalContext(descriptor, call, " "));
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

        public ValueTask<ConversationToolResult> ExecuteAsync(ConversationToolCall call, CancellationToken cancellationToken)
        {
            string text = call.Arguments.GetProperty("text").GetString() ?? string.Empty;
            return ValueTask.FromResult(ConversationToolResult.Success($"echo:{text}", JsonSerializer.Serialize(new { echo = text })));
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
            => ValueTask.FromResult(ConversationToolResult.Success(string.Empty, call.Arguments.GetRawText()));

        public Task StopAsync()
        {
            _started = false;
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class AllowAllPolicy : IConversationToolApprovalPolicy
    {
        public ConversationToolApprovalContext? Seen { get; private set; }

        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
        {
            Seen = context;
            return ValueTask.FromResult(ConversationToolApproval.Allow);
        }
    }
}
