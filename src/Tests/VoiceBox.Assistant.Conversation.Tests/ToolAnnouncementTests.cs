using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;
using static VoiceBox.Assistant.Conversation.Tests.Support.ToolTestSupport;

namespace VoiceBox.Assistant.Conversation.Tests;

/// <summary>
/// The fixed texts a tool session speaks on its own come from one configurable place.
/// </summary>
public sealed class ToolAnnouncementTests
{
    private static ScriptedTool BackgroundTool(string name, string displayName, ConversationToolRisk risk = ConversationToolRisk.ReadOnly)
        => new(name, risk, ConversationToolExecutionPolicy.BackgroundOnly, displayName: displayName);

    /// <summary>The first round calls both tools; the model then fails, so no summary can be generated.</summary>
    private static Func<int, LlmToolTurnRequest, IAsyncEnumerable<LlmTurnEvent>> CallBothThenFail()
        => (round, _) => round == 1
            ? Events(
                new LlmToolCallEvent("c1", "local__weather", "{}"),
                new LlmToolCallEvent("c2", "local__calendar", "{}"),
                new LlmTurnCompletedEvent())
            : throw new HttpRequestException("LLM endpoint unavailable");

    [Fact]
    public async Task TheDefaultAcknowledgementNamesEveryToolOfTheBatch()
    {
        var llm = new RecordingLlmService { ToolTurnScript = CallBothThenFail() };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(BackgroundTool("weather", "天气查询")).AddTool(BackgroundTool("calendar", "日程"))));

        List<ConversationEvent> events = await RunTurnAsync(session, "查天气和日程");

        Assert.Equal("我正在后台执行天气查询、日程任务，等会给你结果。", Completed(events));
    }

    [Fact]
    public async Task AnnouncementsCanBeWrittenInAnotherLanguage()
    {
        var options = new ConversationToolOptions
        {
            Announcements = new ConversationToolAnnouncements
            {
                BackgroundStarted = "Working on {0}. I'll get back to you.",
                NameSeparator = " and ",
                BackgroundSucceeded = "{0} is done. ",
            },
        };
        var llm = new RecordingLlmService { ToolTurnScript = CallBothThenFail() };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools
            .AddTool(BackgroundTool("weather", "Weather"))
            .AddTool(BackgroundTool("calendar", "Calendar"))
            .UseOptions(options)));
        // The session keeps its own copy: a later change to the caller's object has no effect.
        options.Announcements.BackgroundStarted = "changed after registration";

        List<ConversationEvent> events = await RunTurnAsync(session, "weather and calendar");

        Assert.Equal("Working on Weather and Calendar. I'll get back to you.", Completed(events));
        // Without a model summary, each call is reported by a fixed sentence.
        Assert.Equal("Weather is done. Calendar is done. ", await NextBackgroundTextAsync(session));
    }

    [Theory]
    [InlineData(ConversationToolRisk.ReadOnly, false, "天气查询任务已完成。")]
    [InlineData(ConversationToolRisk.Sensitive, false, "天气查询任务未获授权，未执行。")]
    [InlineData(ConversationToolRisk.ReadOnly, true, "天气查询任务执行失败。")]
    public async Task WithoutAModelSummaryTheDefaultOutcomeTextIsPublished(ConversationToolRisk risk, bool toolFails, string expected)
    {
        ScriptedTool tool = BackgroundTool("weather", "天气查询", risk);
        if (toolFails)
            tool.OnExecute = (_, _) => Task.FromResult(ConversationToolResult.Error("upstream", "下游失败"));
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round == 1
            ? Events(new LlmToolCallEvent("c1", "local__weather", "{}"), new LlmTurnCompletedEvent())
            : throw new HttpRequestException("LLM endpoint unavailable");
        // No approval policy: a sensitive tool is denied.
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)));

        await RunTurnAsync(session, "天气");

        Assert.Equal(expected, await NextBackgroundTextAsync(session));
    }

    [Fact]
    public async Task AWriteWhoseOutcomeIsUnknownIsAnnouncedAsSuch()
    {
        var tool = new ScriptedTool(
            "transfer",
            ConversationToolRisk.SideEffecting,
            ConversationToolExecutionPolicy.BackgroundOnly,
            TimeSpan.FromMilliseconds(100),
            displayName: "转账")
        {
            OnExecute = async (_, cancellationToken) =>
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
                return ConversationToolResult.Success("unreachable");
            },
        };
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round == 1
            ? Events(new LlmToolCallEvent("c1", "local__transfer", "{}"), new LlmTurnCompletedEvent())
            : throw new HttpRequestException("LLM endpoint unavailable");
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseApprovalPolicy(new AllowAllApprovalPolicy())));

        await RunTurnAsync(session, "转账");

        Assert.Equal("转账任务执行状态暂时无法确认，请勿重复操作。", await NextBackgroundTextAsync(session));
    }

    [Fact]
    public void TemplatesAreValidatedWhenTheOptionsAreRegistered()
    {
        Reject<ArgumentException>(options => options.Announcements.BackgroundStarted = " ");
        Reject<ArgumentNullException>(options => options.Announcements.BackgroundFailed = null!);
        Reject<ArgumentException>(options => options.Announcements.BackgroundSucceeded = "{0} and {1}");
        Reject<ArgumentException>(options => options.Announcements.BackgroundDenied = "{0");
        Reject<ArgumentNullException>(options => options.Announcements.NameSeparator = null!);
        Reject<ArgumentNullException>(options => options.Announcements = null!);

        // A template may leave the tool name out, and may contain literal braces.
        var accepted = new ConversationToolOptions();
        accepted.Announcements.BackgroundStarted = "Working on it.";
        accepted.Announcements.BackgroundSucceeded = "{{done}} {0}";
        new ConversationToolsBuilder().UseOptions(accepted);

        static void Reject<TException>(Action<ConversationToolOptions> change)
            where TException : Exception
        {
            var options = new ConversationToolOptions();
            change(options);
            Assert.Throws<TException>(() => new ConversationToolsBuilder().UseOptions(options));
        }
    }
}
