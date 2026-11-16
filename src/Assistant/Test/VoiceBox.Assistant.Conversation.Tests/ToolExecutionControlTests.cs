using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;
using static VoiceBox.Assistant.Conversation.Tests.Support.ToolTestSupport;

namespace VoiceBox.Assistant.Conversation.Tests;

/// <summary>
/// The session, not the tool, decides how long a call may take, in which order a batch runs and what is retried.
/// </summary>
public sealed class ToolExecutionControlTests
{
    private static TaskCompletionSource Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    [Fact]
    public async Task AToolThatIgnoresItsTimeoutDoesNotHoldTheTurn()
    {
        TaskCompletionSource release = Signal();
        var tool = new ScriptedTool("slow", timeout: TimeSpan.FromMilliseconds(100))
        {
            // Never looks at its cancellation token.
            OnExecute = async (_, _) =>
            {
                await release.Task;
                return ConversationToolResult.Success("too late");
            },
        };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__slow") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseOptions(new ConversationToolOptions { MaxToolRetryAttempts = 0 })));
        try
        {
            List<ConversationEvent> events = await RunTurnAsync(session, "查一下");

            // The turn is over although the tool has not returned.
            Assert.False(release.Task.IsCompleted);
            Assert.Equal("完成", Completed(events));
            ToolResultContent result = ToolResultOf(llm.Prompts[1]);
            Assert.True(result.IsError);
            Assert.Contains("TimedOut", result.ResultJson, StringComparison.Ordinal);
        }
        finally
        {
            release.TrySetResult();
        }
    }

    [Fact]
    public async Task CancelingATurnDoesNotWaitForAToolThatIgnoresCancellation()
    {
        TaskCompletionSource entered = Signal();
        TaskCompletionSource release = Signal();
        var tool = new ScriptedTool("stuck")
        {
            OnExecute = async (_, _) =>
            {
                entered.TrySetResult();
                await release.Task;
                return ConversationToolResult.Success("too late");
            },
        };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__stuck") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)));
        using var cancellation = new CancellationTokenSource();
        Task<List<ConversationEvent>> turn = ConversationEvents.CollectAsync(
            session.StreamAsync(Request("查一下"), cancellation.Token),
            cancellation.Token);
        try
        {
            await entered.Task.WaitAsync(Patience);
            await cancellation.CancelAsync();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => turn.WaitAsync(Patience));
            Assert.False(release.Task.IsCompleted);

            // The session is free again: the next turn does not queue behind the abandoned call.
            llm.ToolTurnScript = (_, _) => Events(new LlmTurnTextDelta("下一轮"), new LlmTurnCompletedEvent());
            Assert.Equal("下一轮", Completed(await RunTurnAsync(session, "继续")));
        }
        finally
        {
            release.TrySetResult();
        }
    }

    [Fact]
    public async Task ACallLeftBehindIsGivenAMomentToFinishBeforeItsToolIsDisposed()
    {
        TaskCompletionSource entered = Signal();
        TaskCompletionSource release = Signal();
        bool finished = false;
        bool? finishedWhenDisposed = null;
        var tool = new ScriptedTool("stuck")
        {
            OnExecute = async (_, _) =>
            {
                entered.TrySetResult();
                await release.Task;
                Volatile.Write(ref finished, true);
                return ConversationToolResult.Success("too late");
            },
        };
        tool.OnDispose = () => finishedWhenDisposed = Volatile.Read(ref finished);
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__stuck") };
        IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)));
        using var cancellation = new CancellationTokenSource();
        Task<List<ConversationEvent>> turn = ConversationEvents.CollectAsync(
            session.StreamAsync(Request("查一下"), cancellation.Token),
            cancellation.Token);
        try
        {
            await entered.Task.WaitAsync(Patience);
            await cancellation.CancelAsync();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => turn.WaitAsync(Patience));

            Task dispose = session.DisposeAsync().AsTask();
            await Task.Delay(50);
            release.TrySetResult();
            await dispose.WaitAsync(Patience);

            Assert.True(finishedWhenDisposed);
        }
        finally
        {
            release.TrySetResult();
            await session.DisposeAsync();
        }
    }

    [Fact]
    public async Task TheRetryBudgetDoesNotShortenAToolsOwnTimeout()
    {
        var tool = new ScriptedTool("job", ConversationToolRisk.SideEffecting, timeout: TimeSpan.FromSeconds(30))
        {
            OnExecute = async (_, cancellationToken) =>
            {
                await Task.Delay(300, cancellationToken);
                return ConversationToolResult.Success("done");
            },
        };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__job") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools
            .AddTool(tool)
            .UseApprovalPolicy(new AllowAllApprovalPolicy())
            .UseOptions(new ConversationToolOptions { MaxToolRetryBudgetMs = 50 })));

        await RunTurnAsync(session, "执行");

        ToolResultContent result = ToolResultOf(llm.Prompts[1]);
        Assert.False(result.IsError);
        Assert.Contains("done", result.ResultJson, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABatchRunsReadsTogetherAndKeepsEveryWriteWhereTheModelPutIt()
    {
        var journal = new List<string>();
        TaskCompletionSource bothReadsStarted = Signal();
        int readsStarted = 0;

        void Note(string entry)
        {
            lock (journal)
                journal.Add(entry);
        }

        ScriptedTool OverlappingRead(string name) => new(name)
        {
            OnExecute = async (_, cancellationToken) =>
            {
                Note($"start:{name}");
                if (Interlocked.Increment(ref readsStarted) == 2)
                    bothReadsStarted.TrySetResult();
                // Only returns once the other read is running too.
                await bothReadsStarted.Task.WaitAsync(Patience, cancellationToken);
                Note($"end:{name}");
                return ConversationToolResult.Success(name);
            },
        };

        ScriptedTool Recorded(string name, ConversationToolRisk risk) => new(name, risk)
        {
            OnExecute = async (_, cancellationToken) =>
            {
                Note($"start:{name}");
                await Task.Delay(20, cancellationToken);
                Note($"end:{name}");
                return ConversationToolResult.Success(name);
            },
        };

        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round == 1
            ? Events(
                new LlmToolCallEvent("c1", "local__read_a", "{}"),
                new LlmToolCallEvent("c2", "local__read_b", "{}"),
                new LlmToolCallEvent("c3", "local__write", "{}"),
                new LlmToolCallEvent("c4", "local__read_c", "{}"),
                new LlmTurnCompletedEvent())
            : Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent());
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools
            .AddTool(OverlappingRead("read_a"))
            .AddTool(OverlappingRead("read_b"))
            .AddTool(Recorded("write", ConversationToolRisk.SideEffecting))
            .AddTool(Recorded("read_c", ConversationToolRisk.ReadOnly))
            .UseApprovalPolicy(new AllowAllApprovalPolicy())));

        await RunTurnAsync(session, "先查再写再查");

        int At(string entry) => journal.IndexOf(entry);
        Assert.Equal(8, journal.Count);
        // Reads ahead of the write overlap each other and are finished before it starts.
        Assert.True(Math.Max(At("start:read_a"), At("start:read_b")) < Math.Min(At("end:read_a"), At("end:read_b")));
        Assert.True(Math.Max(At("end:read_a"), At("end:read_b")) < At("start:write"));
        // The read behind the write starts after it has finished.
        Assert.True(At("end:write") < At("start:read_c"));
        // Results go back in call order, whatever order the calls finished in.
        Assert.Equal(["c1", "c2", "c3", "c4"], ToolResultsOf(llm.Prompts[1]).Select(result => result.CallId));
    }

    [Theory]
    [InlineData("weather_network_error", true, 2)]
    [InlineData("IOException", false, 1)]
    public async Task OnlyAFailureMarkedTransientIsRetried(string errorCode, bool isTransient, int expectedExecutions)
    {
        var tool = new ScriptedTool("fetch")
        {
            OnExecute = (_, _) => Task.FromResult(ConversationToolResult.Error(errorCode, "下游失败", isTransient)),
        };
        await RunOneCallAsync(tool, new ConversationToolOptions { MaxToolRetryAttempts = 1, ToolRetryBaseDelayMs = 1 });

        Assert.Equal(expectedExecutions, tool.ExecuteCount);
    }

    [Theory]
    [InlineData(typeof(HttpRequestException), 2)]
    [InlineData(typeof(IOException), 2)]
    [InlineData(typeof(FileNotFoundException), 1)]
    [InlineData(typeof(InvalidOperationException), 1)]
    public async Task AThrownExceptionIsRetriedOnlyForConnectionLevelFailures(Type exceptionType, int expectedExecutions)
    {
        var tool = new ScriptedTool("fetch")
        {
            OnExecute = (_, _) => throw (Exception)Activator.CreateInstance(exceptionType)!,
        };
        RecordingLlmService llm = await RunOneCallAsync(
            tool,
            new ConversationToolOptions { MaxToolRetryAttempts = 1, ToolRetryBaseDelayMs = 1 });

        Assert.Equal(expectedExecutions, tool.ExecuteCount);
        ToolResultContent result = ToolResultOf(llm.Prompts[1]);
        Assert.True(result.IsError);
        Assert.Contains(exceptionType.Name, result.ResultJson, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AToolWithSideEffectsIsNeverRetried()
    {
        var tool = new ScriptedTool("mutate", ConversationToolRisk.SideEffecting)
        {
            OnExecute = (_, _) => Task.FromResult(ConversationToolResult.Error("busy", "稍后再试", isTransient: true)),
        };
        await RunOneCallAsync(tool, new ConversationToolOptions { MaxToolRetryAttempts = 2, ToolRetryBaseDelayMs = 1 });

        Assert.Equal(1, tool.ExecuteCount);
    }

    [Fact]
    public async Task AWriteThatTimesOutIsReportedAsUnknownAndTheSameCallIsNotReplayed()
    {
        var tool = new ScriptedTool("transfer", ConversationToolRisk.SideEffecting, timeout: TimeSpan.FromMilliseconds(100))
        {
            OnExecute = async (_, cancellationToken) =>
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
                return ConversationToolResult.Success("unreachable");
            },
        };
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round switch
        {
            1 => Events(new LlmToolCallEvent("c1", "local__transfer", """{"to":"张三","amount":100}"""), new LlmTurnCompletedEvent()),
            // The same call again, written differently.
            2 => Events(new LlmToolCallEvent("c2", "local__transfer", """{ "amount": 100, "to": "张三" }"""), new LlmTurnCompletedEvent()),
            _ => Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseApprovalPolicy(new AllowAllApprovalPolicy())));

        await RunTurnAsync(session, "转账");

        Assert.Equal(1, tool.ExecuteCount);
        Assert.Contains("UnknownOutcome", ToolResultOf(llm.Prompts[1]).ResultJson, StringComparison.Ordinal);
        Assert.Contains("retry_blocked_after_unknown_outcome", ToolResultsOf(llm.Prompts[2])[^1].ResultJson, StringComparison.Ordinal);
    }

    private static async Task<RecordingLlmService> RunOneCallAsync(ScriptedTool tool, ConversationToolOptions options)
    {
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer($"local__{tool.Descriptor.Name}") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseApprovalPolicy(new AllowAllApprovalPolicy()).UseOptions(options)));
        await RunTurnAsync(session, "执行");
        return llm;
    }
}
