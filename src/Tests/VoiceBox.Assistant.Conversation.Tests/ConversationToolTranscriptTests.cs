using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

/// <summary>
/// Cross-turn tool memory: a committed tool exchange must survive into later prompts so the model
/// can answer a follow-up from structured results instead of only its own spoken summary.
/// </summary>
public sealed class ConversationToolTranscriptTests
{
    [Fact]
    public void CommittedToolExchange_ReachesTheNextTurnPrompt()
    {
        ConversationHistoryStore history = CreateStore();
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴，31度"));

        history.AddUserMessage(2, "那穿短袖行吗", null, out ChatMessage[] prompt);

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant, ChatRole.User],
            prompt.Select(static message => message.Role));
        Assert.Equal("call-1", ToolCallOf(prompt[2]).CallId);
        Assert.Equal("{\"temperature\":31}", ToolResultOf(prompt[3]).ResultJson);
        Assert.Equal("北京晴，31度", TextOf(prompt[4]));
    }

    [Fact]
    public void ToolExchange_IsDroppedWhenItsTurnIsNeverAnswered()
    {
        ConversationHistoryStore history = CreateStore();
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));

        history.RemoveUnansweredUserMessage(1);

        Assert.Equal([ChatRole.System], history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void DiscardedToolExchange_DoesNotConsumeARetentionSlot()
    {
        ConversationHistoryStore history = CreateStore(retentionTurns: 1);
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        history.RemoveUnansweredUserMessage(1);

        history.AddUserMessage(2, "上海天气", null, out _);
        history.RecordToolExchange(2, ToolExchange("call-2", "{\"b\":2}"));
        Assert.True(history.CommitGeneratedResponse(2, "上海阴"));

        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        Assert.Equal("call-2", ToolCallOf(transcript[2]).CallId);
        Assert.Equal("{\"b\":2}", ToolResultOf(transcript[3]).ResultJson);
    }

    [Fact]
    public void ToolExchanges_AreRetainedOnlyForTheMostRecentWindow()
    {
        ConversationHistoryStore history = CreateStore(maxHistoryTurns: 10, retentionTurns: 2);
        for (int turn = 1; turn <= 3; turn++)
        {
            history.AddUserMessage(turn, $"问题{turn}", null, out _);
            history.RecordToolExchange(turn, ToolExchange($"call-{turn}", $"{{\"turn\":{turn}}}"));
            Assert.True(history.CommitGeneratedResponse(turn, $"回答{turn}"));
        }

        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        string[] retainedCallIds = transcript
            .SelectMany(static message => message.Content.OfType<ToolCallContent>())
            .Select(static call => call.CallId)
            .ToArray();

        Assert.Equal(["call-2", "call-3"], retainedCallIds);
        Assert.Equal(["问题1", "回答1", "问题2", "回答2", "问题3", "回答3"], transcript.Where(IsPlainText).Select(TextOf));
    }

    [Fact]
    public void RetentionWindow_CountsOnlyTurnsThatCarryAToolExchange()
    {
        ConversationHistoryStore history = CreateStore(maxHistoryTurns: 10, retentionTurns: 1);
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴"));

        for (int turn = 2; turn <= 4; turn++)
        {
            history.AddUserMessage(turn, $"闲聊{turn}", null, out _);
            Assert.True(history.CommitGeneratedResponse(turn, $"嗯{turn}"));
        }

        Assert.Equal(
            "call-1",
            Assert.Single(history.GetModelTranscriptSnapshot()
                .SelectMany(static message => message.Content.OfType<ToolCallContent>())).CallId);
    }

    [Fact]
    public void RetentionOfZero_KeepsToolMessagesOutOfTheTranscript()
    {
        ConversationHistoryStore history = CreateStore(retentionTurns: 0);
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴"));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void UserConversationHistory_NeverExposesToolMessages()
    {
        ConversationHistoryStore history = CreateStore();
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"secret\":\"raw tool output\"}"));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴，31度"));

        ChatMessage[] visible = history.GetUserConversationHistorySnapshot();

        Assert.Equal([ChatRole.User, ChatRole.Assistant], visible.Select(static message => message.Role));
        Assert.DoesNotContain(
            visible.SelectMany(static message => message.Content),
            static content => content is ToolCallContent or ToolResultContent);
    }

    [Fact]
    public void OversizedToolResult_IsTruncatedAndMarkedWhenPersisted()
    {
        ConversationHistoryStore history = CreateStore(maxPersistedToolResultBytes: 256);
        string oversized = $"{{\"payload\":\"{new string('x', 4096)}\"}}";
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", oversized));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴"));

        ToolResultContent persisted = ToolResultOf(history.GetModelTranscriptSnapshot()[3]);

        Assert.Contains("\"__truncated\":true", persisted.ResultJson);
        Assert.DoesNotContain(new string('x', 4096), persisted.ResultJson);
        Assert.Equal("call-1", persisted.CallId);
    }

    [Fact]
    public void ToolResultWithinBudget_IsPersistedVerbatim()
    {
        ConversationHistoryStore history = CreateStore(maxPersistedToolResultBytes: 256);
        history.AddUserMessage(1, "北京天气", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));
        Assert.True(history.CommitGeneratedResponse(1, "北京晴"));

        Assert.Equal("{\"temperature\":31}", ToolResultOf(history.GetModelTranscriptSnapshot()[3]).ResultJson);
    }

    [Fact]
    public void ToolCorrelatedTurnTrim_DropsAToolExchangeWithItsTurn()
    {
        ConversationHistoryStore history = CreateStore(maxHistoryTurns: 1, retentionTurns: 5);
        history.AddUserMessage(1, "第一轮", null, out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, "第一轮回答"));

        history.AddUserMessage(2, "第二轮", null, out _);
        Assert.True(history.CommitGeneratedResponse(2, "第二轮回答"));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void BackgroundOnlyTurn_KeepsItsToolExchangeAheadOfTheDeferredSummary()
    {
        ConversationHistoryStore history = CreateStore();
        history.AddUserMessage(1, "北京天气", null, out _);
        Assert.True(history.CommitGeneratedResponse(1, "正在查询"));
        Assert.True(history.RecordBackgroundResult(1, "查到了，北京晴", ToolExchange("call-1", "{\"temperature\":31}")));
        history.RemoveUnansweredUserMessage(1);

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void BackgroundResultsCannotCommitOrRestoreAnUnansweredTurn()
    {
        ConversationHistoryStore history = CreateStore();
        history.AddUserMessage(1, "北京天气", null, out _);
        ChatMessage[] exchange = ToolExchange("call-1", "{\"temperature\":31}");

        Assert.False(history.RecordBackgroundResult(1, "查到了，北京晴", exchange));
        history.RemoveUnansweredUserMessage(1);
        Assert.False(history.RecordBackgroundResult(1, "查到了，北京晴", exchange));
        Assert.Single(history.GetModelTranscriptSnapshot());
    }

    private static ConversationHistoryStore CreateStore(
        int maxHistoryTurns = 20,
        int retentionTurns = 3,
        int maxPersistedToolResultBytes = 8 * 1024)
        => new("system", maxHistoryTurns, retentionTurns, maxPersistedToolResultBytes);

    private static ChatMessage[] ToolExchange(string callId, string resultJson) =>
    [
        ChatMessage.AssistantToolCalls(new ToolCallContent(callId, "local__weather", "{\"city\":\"北京\"}")),
        ChatMessage.ToolResult(callId, resultJson),
    ];

    private static bool IsPlainText(ChatMessage message)
        => message.Role is ChatRole.User or ChatRole.Assistant && message.Content.Count == 1 && message.Content[0] is TextContent;

    private static string TextOf(ChatMessage message)
        => Assert.IsType<TextContent>(Assert.Single(message.Content)).Text;

    private static ToolCallContent ToolCallOf(ChatMessage message)
        => Assert.IsType<ToolCallContent>(Assert.Single(message.Content));

    private static ToolResultContent ToolResultOf(ChatMessage message)
        => Assert.IsType<ToolResultContent>(Assert.Single(message.Content));
}
