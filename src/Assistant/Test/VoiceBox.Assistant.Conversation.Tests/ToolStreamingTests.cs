using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;
using static VoiceBox.Assistant.Conversation.Tests.Support.ToolTestSupport;

namespace VoiceBox.Assistant.Conversation.Tests;

/// <summary>
/// A tool-enabled session forwards visible text as the model produces it, exactly like a plain session.
/// </summary>
public sealed class ToolStreamingTests
{
    [Fact]
    public async Task AnswerTextIsForwardedWhileTheModelIsStillGenerating()
    {
        var stillGenerating = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var llm = new RecordingLlmService { ToolTurnScript = (_, _) => Answer() };
        await using IConversationSession session = await StartAsync(
            new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new ScriptedTool("unused"))));

        await using IAsyncEnumerator<ConversationEvent> events = session.StreamAsync(Request("你好")).GetAsyncEnumerator();
        try
        {
            // The first sentence arrives although the model has not finished its reply.
            Assert.True(await events.MoveNextAsync().AsTask().WaitAsync(Patience));
            Assert.Equal("第一句。", Assert.IsType<ConversationTextDelta>(events.Current).Text);
            Assert.False(stillGenerating.Task.IsCompleted);
        }
        finally
        {
            stillGenerating.TrySetResult();
        }

        Assert.True(await events.MoveNextAsync());
        Assert.Equal("第二句。", Assert.IsType<ConversationTextDelta>(events.Current).Text);
        Assert.True(await events.MoveNextAsync());
        Assert.Equal("第一句。第二句。", Assert.IsType<ConversationCompleted>(events.Current).Text);
        Assert.False(await events.MoveNextAsync());

        async IAsyncEnumerable<LlmTurnEvent> Answer()
        {
            yield return new LlmTurnTextDelta("第一句。");
            await stillGenerating.Task;
            yield return new LlmTurnTextDelta("第二句。");
            yield return new LlmTurnCompletedEvent();
        }
    }

    [Fact]
    public async Task TextSaidBeforeAToolCallGoesBackToTheModelWithThatCallAndIsCommittedOnce()
    {
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round switch
        {
            1 => Events(new LlmTurnTextDelta("我查一下。"), new LlmToolCallEvent("c1", "local__weather", "{}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("北京晴。"), new LlmTurnCompletedEvent()),
            _ => Events(new LlmTurnTextDelta("不客气。"), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = await StartAsync(
            new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new ScriptedTool("weather"))));

        List<ConversationEvent> events = await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("谢谢")));

        // What follows the tool starts on its own line instead of running into what was said before it.
        Assert.Equal(["我查一下。", "\n北京晴。"], events.OfType<ConversationTextDelta>().Select(item => item.Text));
        Assert.Equal("我查一下。\n北京晴。", Assert.IsType<ConversationCompleted>(events[^1]).Text);

        // The continuation request repeats what was already said, so the model carries on from there.
        ChatMessage continued = Assert.Single(llm.Prompts[1], message => message.Content.Any(content => content is ToolCallContent));
        Assert.Equal(ChatRole.Assistant, continued.Role);
        Assert.Equal("我查一下。", Assert.IsType<TextContent>(continued.Content[0]).Text);
        Assert.Equal("c1", Assert.IsType<ToolCallContent>(continued.Content[1]).CallId);

        // Committed history holds the spoken text once: as the turn's answer, not on the tool call.
        ChatMessage[] nextTurn = llm.Prompts[2];
        ChatMessage persistedCall = Assert.Single(nextTurn, message => message.Content.Any(content => content is ToolCallContent));
        Assert.IsType<ToolCallContent>(Assert.Single(persistedCall.Content));
        Assert.Single(nextTurn, message => TextOf(message).Contains("我查一下。", StringComparison.Ordinal));
        Assert.Contains(nextTurn, message => message.Role == ChatRole.Assistant && TextOf(message) == "我查一下。\n北京晴。");
    }

    [Fact]
    public async Task BlankTextBeforeAToolCallIsNotForwarded()
    {
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round switch
        {
            1 => Events(
                new LlmTurnTextDelta("\n"),
                new LlmTurnTextDelta("\n"),
                new LlmToolCallEvent("c1", "local__weather", "{}"),
                new LlmTurnCompletedEvent()),
            _ => Events(new LlmTurnTextDelta("北京晴。"), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = await StartAsync(
            new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new ScriptedTool("weather"))));

        List<ConversationEvent> events = await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));

        Assert.Equal("北京晴。", Assert.Single(events.OfType<ConversationTextDelta>()).Text);
        ChatMessage call = Assert.Single(llm.Prompts[1], message => message.Content.Any(content => content is ToolCallContent));
        Assert.IsType<ToolCallContent>(Assert.Single(call.Content));
    }

    [Fact]
    public async Task LeadingBlankTextOfAnAnswerIsKept()
    {
        var llm = new RecordingLlmService
        {
            ToolTurnScript = (_, _) => Events(new LlmTurnTextDelta("\n"), new LlmTurnTextDelta("答案"), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = await StartAsync(
            new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new ScriptedTool("unused"))));

        List<ConversationEvent> events = await ConversationEvents.CollectAsync(session.StreamAsync(Request("问题")));

        Assert.Equal("\n答案", Assert.IsType<ConversationCompleted>(events[^1]).Text);
    }

    [Fact]
    public async Task TextBeforeABackgroundToolIsFollowedByTheAcknowledgementOnItsOwnLine()
    {
        var tool = new ScriptedTool("weather", policy: ConversationToolExecutionPolicy.BackgroundOnly, executionPrompt: "正在查询");
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (round, _) => round switch
        {
            1 => Events(new LlmTurnTextDelta("稍等。"), new LlmToolCallEvent("c1", "local__weather", "{}"), new LlmTurnCompletedEvent()),
            _ => Events(new LlmTurnTextDelta("北京晴。"), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = await StartAsync(new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)));

        List<ConversationEvent> events = await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));

        ConversationTextDelta[] deltas = events.OfType<ConversationTextDelta>().ToArray();
        Assert.Equal(["稍等。", "\n正在查询"], deltas.Select(item => item.Text));
        // Only the host's own prompt closes a speech segment; model text never does.
        Assert.Equal([false, true], deltas.Select(item => item.EndsTextSegment));
        Assert.Equal("稍等。\n正在查询", Assert.IsType<ConversationCompleted>(events[^1]).Text);
    }
}
