using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ToolPolicyTests
{
    private static ConversationRequest Request(string text)
        => new(Guid.NewGuid(), text);

    private static async IAsyncEnumerable<LlmTurnEvent> Events(params LlmTurnEvent[] events)
    {
        await Task.Yield();
        foreach (LlmTurnEvent item in events)
            yield return item;
    }

    private static string OnlyToolName(LlmToolTurnRequest request)
        => Assert.Single(request.Tools).Name;

    private static LlmTurnEvent Call(
        string callId,
        LlmToolTurnRequest request,
        string arguments = "{}")
        => new LlmToolCallEvent(callId, OnlyToolName(request), arguments);

    [Fact]
    public async Task ReadOnlyToolsInOneBatchExecuteConcurrently()
    {
        var llm = new RecordingLlmService();
        var first = new ProbeTool("a", ConversationToolRisk.ReadOnly);
        var second = new ProbeTool("b", ConversationToolRisk.ReadOnly);
        TaskCompletionSource bothStarted = new(TaskCreationOptions.RunContinuationsAsynchronously);
        int started = 0;
        ValueTask<ConversationToolExecutionResult> WaitAsync(CancellationToken cancellationToken)
        {
            if (Interlocked.Increment(ref started) == 2)
                bothStarted.TrySetResult();
            return new ValueTask<ConversationToolExecutionResult>(WaitAndReturnAsync());
        }
        first.OnExecute = WaitAsync;
        second.OnExecute = WaitAsync;
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(
                new LlmToolCallEvent("c1", request.Tools[0].Name, "{}"),
                new LlmToolCallEvent("c2", request.Tools[1].Name, "{}"),
                new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseOptions(new LlmConversationOptions())
            .UseTools(tools => tools
                .UseOptions(new ConversationToolOptions { MaxParallelToolCalls = 2 })
                .AddTool(first)
                .AddTool(second))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("并行")));
            await bothStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(1, first.ExecuteCount);
            Assert.Equal(1, second.ExecuteCount);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }

        async Task<ConversationToolExecutionResult> WaitAndReturnAsync()
        {
            await bothStarted.Task.WaitAsync(TimeSpan.FromSeconds(2));
            return ConversationToolExecutionResult.Success("ok");
        }
    }

    [Fact]
    public async Task SideEffectingToolsStaySerialInProviderOrder()
    {
        var llm = new RecordingLlmService();
        var order = new List<string>();
        var tool = new ProbeTool("mutate", ConversationToolRisk.SideEffecting)
        {
            OnExecute = async cancellationToken =>
            {
                lock (order) order.Add("enter");
                await Task.Delay(20, cancellationToken);
                lock (order) order.Add("exit");
                return ConversationToolExecutionResult.Success("ok");
            },
        };
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(
                new LlmToolCallEvent("c1", OnlyToolName(request), "{}"),
                new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools
                .UseApprovalPolicy(new AllowAllPolicy())
                .UseOptions(new ConversationToolOptions { MaxParallelToolCalls = 4 })
                .AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("串行")));
            Assert.Equal(["enter", "exit"], order);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task ReadOnlyTransientFailureIsRetried()
    {
        var llm = new RecordingLlmService();
        var tool = new ProbeTool("flaky", ConversationToolRisk.ReadOnly);
        tool.OnExecute = _ => ValueTask.FromResult(
            tool.ExecuteCount == 1
                ? ConversationToolExecutionResult.Error("IOException", "transient")
                : ConversationToolExecutionResult.Success("recovered"));
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(new LlmToolCallEvent("c1", OnlyToolName(request), "{}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools
                .UseOptions(new ConversationToolOptions
                {
                    MaxToolRetryAttempts = 1,
                    ToolRetryBaseDelayMs = 1,
                    MaxToolRetryBudgetMs = 5_000,
                })
                .AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("重试")));
            Assert.Equal(2, tool.ExecuteCount);
            ChatMessage result = Assert.Single(
                llm.Prompts[1], message => message.Role == ChatRole.Tool);
            string json = ((ToolResultContent)Assert.Single(result.Content)).ResultJson;
            Assert.Contains("recovered", json, StringComparison.Ordinal);
            Assert.DoesNotContain("transient", json, StringComparison.Ordinal);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task SensitiveToolWithoutAnApprovalPolicyIsDenied()
    {
        var llm = new RecordingLlmService();
        var tool = new ProbeTool("sensitive", ConversationToolRisk.Sensitive);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(new LlmToolCallEvent("c1", OnlyToolName(request), "{}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("危险")));
            Assert.Equal(0, tool.ExecuteCount);
            Assert.Contains(llm.Prompts[1],
                message => message.Content.OfType<ToolResultContent>()
                    .Any(result => result.ResultJson.Contains("Denied", StringComparison.Ordinal)));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task UnknownOutcomeIsNeverReplayedByTheModel()
    {
        var llm = new RecordingLlmService();
        var provider = new StaticToolProvider(
            "remote",
            ConversationToolRisk.SideEffecting,
            ConversationToolResultKind.UnknownOutcome);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(new LlmToolCallEvent("c1", OnlyToolName(request), "{\"x\":1}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmToolCallEvent("c2", OnlyToolName(request), "{\"x\":1}"), new LlmTurnCompletedEvent()),
            3 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools
                .UseApprovalPolicy(new AllowAllPolicy())
                .AddToolProvider(provider))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("未知")));
            Assert.Equal(1, provider.InvokeCount);
            ChatMessage[] secondRound = llm.Prompts[1];
            Assert.Contains(secondRound,
                message => message.Content.OfType<ToolResultContent>()
                    .Any(result => result.ResultJson.Contains(
                        "UnknownOutcome", StringComparison.Ordinal)));
            Assert.Contains(llm.Prompts[2],
                message => message.Content.OfType<ToolResultContent>()
                    .Any(result => result.ResultJson.Contains(
                        "retry_blocked_after_unknown_outcome", StringComparison.Ordinal)));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task ContextProviderRunsOnceAcrossToolContinuationRounds()
    {
        var llm = new RecordingLlmService();
        var tool = new ProbeTool("read", ConversationToolRisk.ReadOnly);
        var context = new CountingContextProvider();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(new LlmToolCallEvent("c1", OnlyToolName(request), "{}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .AddContextProvider(context)
            .UseTools(tools => tools.AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("工具")));
            Assert.Equal(1, context.Calls);
            Assert.Equal(2, llm.Prompts.Count);
            Assert.Contains("memory", TextOf(llm.Prompts[1][0]), StringComparison.Ordinal);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public void ToolOptionsRejectOutOfRangeValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { ToolTranscriptRetentionTurns = -1 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { MaxPersistedToolResultBytes = 0 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { MaxParallelToolCalls = 0 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { MaxToolRetryAttempts = -1 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { ToolRetryBaseDelayMs = -1 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ConversationToolOptions { MaxToolRetryBudgetMs = -1 }.Clone());
    }

    [Fact]
    public void LlmConversationOptionsRejectInvalidValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmConversationOptions { MaxHistoryTurns = 0 }.Clone());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmConversationOptions { MaxResponseTokens = 0 }.Clone());
    }

    [Fact]
    public void AProviderCannotClaimTheReservedLocalAlias()
    {
        var llm = new RecordingLlmService();
        Assert.Throws<ArgumentException>(() =>
            new ConversationBuilder(llm)
                .UseTools(tools => tools.AddToolProvider(
                    new AliasProvider("local")))
                .Build());
    }

    [Fact]
    public void DuplicateProviderAliasesFailAtBuildWithoutConnecting()
    {
        var llm = new RecordingLlmService();
        Assert.Throws<InvalidOperationException>(() =>
            new ConversationBuilder(llm)
                .UseTools(tools =>
                {
                    tools.AddToolProvider(new AliasProvider("dup"));
                    tools.AddToolProvider(new AliasProvider("dup"));
                })
                .Build());
    }

    private sealed class AliasProvider(string alias) :
        IConversationToolProvider,
        IConversationToolProviderMetadata
    {
        public IReadOnlyList<string> ServerAliases { get; } = [alias];

        public Task StartAsync(CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult<IReadOnlyList<ConversationToolDefinition>>(
                Array.Empty<ConversationToolDefinition>());

        public ValueTask<ConversationToolResult> InvokeAsync(
            ConversationToolCall call,
            CancellationToken cancellationToken = default)
            => throw new NotSupportedException();

        public Task StopAsync() => Task.CompletedTask;

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private static string TextOf(ChatMessage message)
        => string.Concat(message.Content.OfType<TextContent>().Select(content => content.Text));

    private sealed class ProbeTool(
        string name,
        ConversationToolRisk risk) : IConversationTool
    {
        public ConversationToolDescriptor Descriptor { get; } = new(
            name,
            name,
            name,
            """
            {
              "type": "object",
              "properties": {},
              "additionalProperties": true
            }
            """,
            risk: risk,
            executionPolicy: ConversationToolExecutionPolicy.ForegroundOnly,
            timeout: TimeSpan.FromSeconds(5));

        public int ExecuteCount { get; private set; }

        public Func<CancellationToken, ValueTask<ConversationToolExecutionResult>>? OnExecute { get; set; }

        public ValueTask<ConversationToolExecutionResult> ExecuteAsync(
            ConversationToolInvocation invocation,
            CancellationToken cancellationToken)
        {
            ExecuteCount++;
            return OnExecute is null
                ? ValueTask.FromResult(ConversationToolExecutionResult.Success("ok"))
                : OnExecute(cancellationToken);
        }

        public void Dispose()
        {
        }
    }

    private sealed class StaticToolProvider(
        string alias,
        ConversationToolRisk risk,
        ConversationToolResultKind resultKind) : IConversationToolProvider
    {
        private int _invokeCount;

        public int InvokeCount => Volatile.Read(ref _invokeCount);

        public Task StartAsync(CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult<IReadOnlyList<ConversationToolDefinition>>(
            [
                new ConversationToolDefinition(
                    new ConversationToolId(alias, "work"),
                    new ConversationToolDescriptor(
                        "work",
                        "work",
                        "work",
                        """
                        {
                          "type": "object",
                          "properties": {},
                          "additionalProperties": true
                        }
                        """,
                        risk: risk,
                        executionPolicy: ConversationToolExecutionPolicy.ForegroundOnly,
                        timeout: TimeSpan.FromSeconds(5))),
            ]);

        public ValueTask<ConversationToolResult> InvokeAsync(
            ConversationToolCall call,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _invokeCount);
            return ValueTask.FromResult(new ConversationToolResult(
                call.CallId,
                resultKind,
                ErrorCode: "unknown"));
        }

        public Task StopAsync() => Task.CompletedTask;

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class AllowAllPolicy : IConversationToolApprovalPolicy
    {
        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
            => ValueTask.FromResult(ConversationToolApproval.Allow);
    }

    private sealed class CountingContextProvider : IConversationContextProvider
    {
        public int Calls { get; private set; }

        public ValueTask<string> GetContextAsync(
            ConversationContext context,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            return ValueTask.FromResult("memory");
        }
    }
}
