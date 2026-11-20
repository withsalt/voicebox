using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Tests.Support;
using Xunit;
using static VoiceBox.Assistant.Conversation.Tests.Support.ToolTestSupport;

namespace VoiceBox.Assistant.Conversation.Tests;

/// <summary>
/// Failures that used to vanish are reported through the logger factory given to the builder.
/// </summary>
public sealed class ToolDiagnosticsTests
{
    [Fact]
    public async Task AnExceptionThrownByAToolIsLoggedInFull()
    {
        var logs = new RecordingLoggerFactory();
        var tool = new ScriptedTool("broken") { OnExecute = (_, _) => throw new InvalidOperationException("数据库连接已断开") };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__broken") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseLoggerFactory(logs)
            .UseTools(tools => tools.AddTool(tool)));

        await RunTurnAsync(session, "执行");

        LogEntry entry = Assert.Single(logs.Entries, item => item.Exception is not null);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Equal(typeof(LlmToolConversationSession).FullName, entry.Category);
        Assert.Equal("数据库连接已断开", Assert.IsType<InvalidOperationException>(entry.Exception).Message);
        Assert.Contains("local/broken", entry.Message, StringComparison.Ordinal);
        Assert.Contains("c1", entry.Message, StringComparison.Ordinal);
        // The model is still told only the kind of failure.
        string modelResult = ToolResultOf(llm.Prompts[1]).ResultJson;
        Assert.Contains(nameof(InvalidOperationException), modelResult, StringComparison.Ordinal);
        Assert.DoesNotContain("数据库连接已断开", modelResult, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AFailedBackgroundRunIsLoggedAsAnError()
    {
        var logs = new RecordingLoggerFactory();
        var tool = new ScriptedTool("weather", ConversationToolRisk.Sensitive, ConversationToolExecutionPolicy.BackgroundOnly);
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__weather") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseLoggerFactory(logs)
            .UseTools(tools => tools.AddTool(tool).UseApprovalPolicy(new UnavailableApprovalPolicy())));

        await RunTurnAsync(session, "天气");
        // The user still gets a result; the cause is in the log.
        await NextBackgroundTextAsync(session);

        LogEntry entry = Assert.Single(logs.Entries, item => item.Level == LogLevel.Error);
        Assert.Equal("approval service is down", Assert.IsType<InvalidOperationException>(entry.Exception).Message);
    }

    [Fact]
    public async Task ACallThatIgnoredItsTimeoutIsReportedWhenItFinallyReturns()
    {
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var logs = new RecordingLoggerFactory();
        var tool = new ScriptedTool("slow", timeout: TimeSpan.FromMilliseconds(100))
        {
            OnExecute = async (_, _) =>
            {
                await release.Task;
                return ConversationToolResult.Success("too late");
            },
        };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__slow") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseLoggerFactory(logs)
            .UseTools(tools => tools.AddTool(tool).UseOptions(options => options.MaxToolRetryAttempts = 0)));
        bool Reported() => logs.Entries.Any(item =>
            item.Message.Contains("local/slow", StringComparison.Ordinal) && item.Message.Contains("discarded", StringComparison.Ordinal));
        try
        {
            await RunTurnAsync(session, "查一下");
            Assert.False(Reported());
        }
        finally
        {
            release.TrySetResult();
        }

        await WaitUntilAsync(Reported);
    }

    [Fact]
    public async Task LogsIdentifyACallWithoutRepeatingItsArgumentsOrItsResult()
    {
        var logs = new RecordingLoggerFactory();
        var tool = new ScriptedTool("lookup") { OnExecute = (_, _) => Task.FromResult(ConversationToolResult.Success("结果里的机密")) };
        var llm = new RecordingLlmService { ToolTurnScript = CallThenAnswer("local__lookup", """{"query":"参数里的机密"}""") };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseLoggerFactory(logs)
            .UseTools(tools => tools.AddTool(tool)));

        await RunTurnAsync(session, "用户说的机密");

        Assert.Contains(logs.Entries, item => item.Message.Contains("local/lookup", StringComparison.Ordinal));
        Assert.DoesNotContain(logs.Entries, item => item.Message.Contains("机密", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AToolThatFailsToDisposeIsReportedAndDoesNotFailDisposal()
    {
        var logs = new RecordingLoggerFactory();
        var tool = new ScriptedTool("fragile") { OnDispose = () => throw new InvalidOperationException("dispose failed") };
        IConversationSession session = await StartAsync(new ConversationBuilder(new RecordingLlmService())
            .UseLoggerFactory(logs)
            .UseTools(tools => tools.AddTool(tool)));

        await session.DisposeAsync();

        LogEntry entry = Assert.Single(logs.Entries, item => item.Exception is not null);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Equal("dispose failed", entry.Exception!.Message);
    }

    [Fact]
    public async Task APlainSessionLogsATurnItHadToRollBack()
    {
        var logs = new RecordingLoggerFactory();
        var llm = new RecordingLlmService
        {
            BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        await using IConversationSession session = new ConversationBuilder(llm).UseLoggerFactory(logs).Build();
        using var cancellation = new CancellationTokenSource();

        await using (IAsyncEnumerator<ConversationEvent> events =
            session.StreamAsync(Request("会被取消"), cancellation.Token).GetAsyncEnumerator())
        {
            Assert.True(await events.MoveNextAsync());
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await events.MoveNextAsync());
        }

        LogEntry entry = Assert.Single(logs.Entries);
        Assert.Equal(typeof(LlmConversationSession).FullName, entry.Category);
        Assert.Equal(LogLevel.Debug, entry.Level);
        Assert.Contains("rolled back", entry.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLoggerFactoryIsRegisteredOnce()
    {
        ConversationBuilder builder = new ConversationBuilder(new RecordingLlmService()).UseLoggerFactory(new RecordingLoggerFactory());

        Assert.Throws<InvalidOperationException>(() => builder.UseLoggerFactory(new RecordingLoggerFactory()));
        Assert.Throws<ArgumentNullException>(() => new ConversationBuilder(new RecordingLlmService()).UseLoggerFactory(null!));
    }

    private sealed class UnavailableApprovalPolicy : IConversationToolApprovalPolicy
    {
        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
            => throw new InvalidOperationException("approval service is down");
    }
}
