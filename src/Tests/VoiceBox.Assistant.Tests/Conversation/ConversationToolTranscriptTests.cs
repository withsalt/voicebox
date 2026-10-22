using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

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
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));
        Assert.True(history.CommitGeneratedResponse(1, user, "北京晴，31度"));

        history.AddUserMessage(2, "那穿短袖行吗", out ChatMessage[] prompt);

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
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));

        // A turn interrupted before it committed leaves no trace in the model transcript: an
        // orphaned tool call must never reach a later prompt.
        history.RemoveUnansweredUserMessage(1, user);

        Assert.Equal(
            [ChatRole.System],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void DiscardedToolExchange_DoesNotConsumeARetentionSlot()
    {
        ConversationHistoryStore history = CreateStore(retentionTurns: 1);
        ChatMessage abandoned = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        history.RemoveUnansweredUserMessage(1, abandoned);

        ChatMessage answered = history.AddUserMessage(2, "上海天气", out _);
        history.RecordToolExchange(2, ToolExchange("call-2", "{\"b\":2}"));
        Assert.True(history.CommitGeneratedResponse(2, answered, "上海阴"));

        // The single retention slot belongs to the turn that actually committed.
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
            ChatMessage user = history.AddUserMessage(turn, $"问题{turn}", out _);
            history.RecordToolExchange(turn, ToolExchange($"call-{turn}", $"{{\"turn\":{turn}}}"));
            Assert.True(history.CommitGeneratedResponse(turn, user, $"回答{turn}"));
        }

        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        string[] retainedCallIds = transcript
            .SelectMany(static message => message.Content.OfType<ToolCallContent>())
            .Select(static call => call.CallId)
            .ToArray();

        // The oldest exchange is gone, but its conversational turn is intact: the thread survives
        // while the structured payload stops accumulating.
        Assert.Equal(["call-2", "call-3"], retainedCallIds);
        Assert.Equal(
            ["问题1", "回答1", "问题2", "回答2", "问题3", "回答3"],
            transcript.Where(IsPlainText).Select(TextOf));
    }

    [Fact]
    public void RetentionWindow_CountsOnlyTurnsThatCarryAToolExchange()
    {
        ConversationHistoryStore history = CreateStore(maxHistoryTurns: 10, retentionTurns: 1);
        ChatMessage withTool = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, withTool, "北京晴"));

        // Interleaved text-only turns must not push tool context out of scope.
        for (int turn = 2; turn <= 4; turn++)
        {
            ChatMessage user = history.AddUserMessage(turn, $"闲聊{turn}", out _);
            Assert.True(history.CommitGeneratedResponse(turn, user, $"嗯{turn}"));
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
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, user, "北京晴"));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void UserConversationHistory_NeverExposesToolMessages()
    {
        ConversationHistoryStore history = CreateStore();
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"secret\":\"raw tool output\"}"));
        Assert.True(history.CommitGeneratedResponse(1, user, "北京晴，31度"));

        ChatMessage[] visible = history.GetUserConversationHistorySnapshot();

        Assert.Equal(
            [ChatRole.User, ChatRole.Assistant],
            visible.Select(static message => message.Role));
        Assert.DoesNotContain(
            visible.SelectMany(static message => message.Content),
            static content => content is ToolCallContent or ToolResultContent);
    }

    [Fact]
    public void OversizedToolResult_IsTruncatedAndMarkedWhenPersisted()
    {
        ConversationHistoryStore history = CreateStore(maxPersistedToolResultBytes: 256);
        string oversized = $"{{\"payload\":\"{new string('x', 4096)}\"}}";
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", oversized));
        Assert.True(history.CommitGeneratedResponse(1, user, "北京晴"));

        ToolResultContent persisted = ToolResultOf(history.GetModelTranscriptSnapshot()[3]);

        Assert.Contains("\"__truncated\":true", persisted.ResultJson);
        Assert.DoesNotContain(new string('x', 4096), persisted.ResultJson);
        Assert.Equal("call-1", persisted.CallId);
    }

    [Fact]
    public void ToolResultWithinBudget_IsPersistedVerbatim()
    {
        ConversationHistoryStore history = CreateStore(maxPersistedToolResultBytes: 256);
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));
        Assert.True(history.CommitGeneratedResponse(1, user, "北京晴"));

        Assert.Equal(
            "{\"temperature\":31}",
            ToolResultOf(history.GetModelTranscriptSnapshot()[3]).ResultJson);
    }

    [Fact]
    public void ToolCorrelatedTurnTrim_DropsAToolExchangeWithItsTurn()
    {
        ConversationHistoryStore history = CreateStore(maxHistoryTurns: 1, retentionTurns: 5);
        ChatMessage first = history.AddUserMessage(1, "第一轮", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"a\":1}"));
        Assert.True(history.CommitGeneratedResponse(1, first, "第一轮回答"));

        ChatMessage second = history.AddUserMessage(2, "第二轮", out _);
        Assert.True(history.CommitGeneratedResponse(2, second, "第二轮回答"));

        // Turn-atomic trim: the dropped turn takes its whole tool exchange with it, so no call can
        // survive without its result.
        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void BackgroundOnlyTurn_KeepsItsToolExchangeAheadOfTheDeferredSummary()
    {
        ConversationHistoryStore history = CreateStore();
        ChatMessage user = history.AddUserMessage(1, "北京天气", out _);
        history.RecordToolExchange(1, ToolExchange("call-1", "{\"temperature\":31}"));
        // A background tool result can arrive after the foreground answer was rolled back.
        Assert.True(history.RecordBackgroundAssistantText(1, "查到了，北京晴"));
        history.RemoveUnansweredUserMessage(1, user);

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    private static ConversationHistoryStore CreateStore(
        int maxHistoryTurns = 20,
        int retentionTurns = 3,
        int maxPersistedToolResultBytes = 8 * 1024)
        => new("system", maxHistoryTurns, retentionTurns, maxPersistedToolResultBytes);

    /// <summary>
    /// The exact message shape the orchestrator appends to a provider transcript: an assistant
    /// tool-call message followed by its correlated tool result.
    /// </summary>
    private static ChatMessage[] ToolExchange(string callId, string resultJson) =>
    [
        ChatMessage.AssistantToolCalls(new ToolCallContent(callId, "local__weather", "{\"city\":\"北京\"}")),
        ChatMessage.ToolResult(callId, resultJson),
    ];

    private static bool IsPlainText(ChatMessage message)
        => message.Role is ChatRole.User or ChatRole.Assistant &&
            message.Content.Count == 1 &&
            message.Content[0] is TextContent;

    private static string TextOf(ChatMessage message)
        => Assert.IsType<TextContent>(Assert.Single(message.Content)).Text;

    private static ToolCallContent ToolCallOf(ChatMessage message)
        => Assert.IsType<ToolCallContent>(Assert.Single(message.Content));

    private static ToolResultContent ToolResultOf(ChatMessage message)
        => Assert.IsType<ToolResultContent>(Assert.Single(message.Content));
}
