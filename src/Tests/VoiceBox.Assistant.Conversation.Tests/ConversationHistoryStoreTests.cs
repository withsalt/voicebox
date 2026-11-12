using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ConversationHistoryStoreTests
{
    [Fact]
    public void TranscriptUserAndAuditViews_StaySeparatedForTextOnlyTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(1, "question", additionalContext: null, out _);
        Assert.True(history.CommitGeneratedResponse(1, "answer"));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(message => message.Role));
        Assert.Equal([ChatRole.User, ChatRole.Assistant], history.GetUserConversationHistorySnapshot().Select(message => message.Role));
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void ToolCorrelatedTurns_AreTrimmedAtomicallyAndProjectedIntoTheAuditTrail()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 1);
        history.AddUserMessage(1, "first", null, out _);
        Assert.True(history.CommitGeneratedResponse(1, "first answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1");

        history.AddUserMessage(2, "second", null, out _);
        Assert.True(history.CommitGeneratedResponse(2, "second answer"));

        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant], transcript.Select(static message => message.Role));
        Assert.Equal("second", MessageText(transcript[1]));
        Assert.Equal("second answer", MessageText(transcript[2]));
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void ToolAuditTrail_ProjectsOneEntryPerCorrelatedResult()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 4);
        history.AddUserMessage(1, "question", null, out _);
        Assert.True(history.CommitGeneratedResponse(1, "answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1");

        ToolAuditRecord audit = Assert.Single(history.GetToolAuditTrailSnapshot());
        Assert.Equal(1, audit.TurnId);
        Assert.Equal("call-1", audit.CallId);
        Assert.Equal("srv", audit.ToolId.ServerAlias);
        Assert.Equal("lookup", audit.ToolId.ToolName);
        Assert.Equal(ConversationToolResultKind.Success, audit.ResultKind);
    }

    [Theory]
    [InlineData(ConversationToolResultKind.Accepted, ConversationToolResultKind.Success)]
    [InlineData(ConversationToolResultKind.Success, ConversationToolResultKind.Accepted)]
    public void ToolAuditTrail_KeepsTheTerminalOutcomeWhateverOrderItArrivesIn(
        ConversationToolResultKind first,
        ConversationToolResultKind second)
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 4);
        var toolId = new ConversationToolId("srv", "lookup");
        history.AddUserMessage(1, "question", null, out _);
        Assert.True(history.CommitGeneratedResponse(1, "answer"));

        // A fast background execution can finish before its acknowledgement is recorded.
        history.RecordToolActivity(1, "call-1", toolId, first);
        history.RecordToolActivity(1, "call-1", toolId, second);

        Assert.Equal(ConversationToolResultKind.Success, Assert.Single(history.GetToolAuditTrailSnapshot()).ResultKind);
    }

    [Fact]
    public void TextOnlyTurns_KeepMessageBasedTrimmingAndAreNotTrimmedByTurnCount()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(1, "first", null, out _);
        history.AddUserMessage(2, "second", null, out _);
        history.AddUserMessage(3, "third", null, out _);

        Assert.True(history.CommitGeneratedResponse(3, string.Empty));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.User, ChatRole.User],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void Rollback_RemovesPendingTurnFromBothTextViews()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(4, "pending", null, out _);

        history.RemoveUnansweredUserMessage(4);

        Assert.Single(history.GetModelTranscriptSnapshot());
        Assert.Empty(history.GetUserConversationHistorySnapshot());
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void AddUserMessage_ReturnsPromptWithSystemAndUser()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);

        ChatMessage user = history.AddUserMessage(1, "你好", null, out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("你好", MessageText(user));
        Assert.Same(user, prompt[1]);
    }

    [Fact]
    public void AdditionalContext_ChangesOnlyTheReturnedSystemMessage()
    {
        var history = new ConversationHistoryStore("base", maxHistoryTurns: 2);

        history.AddUserMessage(1, "问题", "记忆片段", out ChatMessage[] prompt);

        Assert.Contains("base", MessageText(prompt[0]), StringComparison.Ordinal);
        Assert.Contains("记忆片段", MessageText(prompt[0]), StringComparison.Ordinal);
        Assert.Equal("base", MessageText(history.GetModelTranscriptSnapshot()[0]));
    }

    [Fact]
    public void CommitGeneratedResponse_IsIdempotentAndIncludedInNextPrompt()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);
        history.AddUserMessage(1, "你好", null, out _);

        Assert.True(history.CommitGeneratedResponse(1, "好的。"));
        Assert.False(history.CommitGeneratedResponse(1, "重复。"));

        history.AddUserMessage(2, "第二轮", null, out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("好的。", MessageText(prompt[2]));
        Assert.DoesNotContain(prompt, m => MessageText(m) == "重复。");
    }

    [Fact]
    public void CommitOfAnUnknownTurn_DoesNotCreateHistory()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);

        Assert.False(history.CommitGeneratedResponse(99, "answer"));
        Assert.Single(history.GetModelTranscriptSnapshot());
    }

    [Fact]
    public void RollbackOfACommittedTurnIsANoOp()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);
        history.AddUserMessage(1, "第一轮", null, out _);
        history.CommitGeneratedResponse(1, "回答一");

        history.RemoveUnansweredUserMessage(1);
        history.AddUserMessage(2, "第二轮", null, out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("回答一", MessageText(prompt[2]));
    }

    [Fact]
    public void EmptyReplyCommitsTheUserAndCannotBeRolledBack()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(1, "thinking only", null, out _);

        Assert.True(history.CommitGeneratedResponse(1, string.Empty));
        history.RemoveUnansweredUserMessage(1);
        history.AddUserMessage(2, "next", null, out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("thinking only", MessageText(prompt[1]));
        Assert.Equal("next", MessageText(prompt[2]));
    }

    [Fact]
    public void CommitGeneratedResponse_TrimsOldestCompleteTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        for (long turn = 1; turn <= 3; turn++)
        {
            history.AddUserMessage(turn, $"用户{turn}", null, out _);
            history.CommitGeneratedResponse(turn, $"回答{turn}");
        }

        history.AddUserMessage(4, "用户4", null, out ChatMessage[] prompt);

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User, ChatRole.Assistant, ChatRole.User],
            prompt.Select(m => m.Role));
        Assert.DoesNotContain(prompt, m => MessageText(m) is "用户1" or "回答1");
        Assert.Contains(prompt, m => MessageText(m) == "用户2");
        Assert.Contains(prompt, m => MessageText(m) == "回答3");
    }

    [Fact]
    public void CommitGeneratedResponse_TrimsDeduplicationEntriesWithExpiredTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);

        for (long turn = 1; turn <= 3; turn++)
        {
            history.AddUserMessage(turn, $"用户{turn}", null, out _);
            Assert.True(history.CommitGeneratedResponse(turn, $"回答{turn}"));
        }

        Assert.Equal(2, history.CommittedResponseTurnCount);
    }

    [Fact]
    public void CommitGeneratedResponse_MaximumTurnLimitDoesNotOverflow()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: int.MaxValue);
        history.AddUserMessage(1, "第一轮", null, out _);

        bool committed = history.CommitGeneratedResponse(1, "回答一");
        history.AddUserMessage(2, "第二轮", null, out ChatMessage[] prompt);

        Assert.True(committed);
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
    }

    [Fact]
    public void Constructor_RejectsNonPositiveHistoryLimit()
    {
        ArgumentOutOfRangeException zero = Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationHistoryStore("system", 0));
        ArgumentOutOfRangeException negative = Assert.Throws<ArgumentOutOfRangeException>(() => new ConversationHistoryStore("system", -1));

        Assert.Equal("maxHistoryTurns", zero.ParamName);
        Assert.StartsWith("Max history turns must be >= 1.", zero.Message, StringComparison.Ordinal);
        Assert.Equal("maxHistoryTurns", negative.ParamName);
    }

    private static void AttachToolCorrelation(ConversationHistoryStore history, long turnId, string callId)
        => history.RecordToolActivity(turnId, callId, new ConversationToolId("srv", "lookup"), ConversationToolResultKind.Success);

    private static string MessageText(ChatMessage message) => string.Concat(message.Content.OfType<TextContent>().Select(c => c.Text));
}
