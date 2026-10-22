using System.Reflection;
using VoiceBox.Assistant.Modules.Conversation.Models;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class ConversationHistoryStoreTests
{
    [Fact]
    public void TranscriptUserAndAuditViews_StaySeparatedForCurrentTextOnlyTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        ChatMessage user = history.AddUserMessage(1, "question", out _);
        Assert.True(history.CommitGeneratedResponse(1, user, "answer"));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            history.GetModelTranscriptSnapshot().Select(message => message.Role));
        Assert.Equal(
            [ChatRole.User, ChatRole.Assistant],
            history.GetUserConversationHistorySnapshot().Select(message => message.Role));
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void ToolCorrelatedTurns_AreTrimmedAtomicallyAndProjectedIntoTheAuditTrail()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1");

        ChatMessage second = history.AddUserMessage(2, "second", out _);
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));

        // The oldest tool-correlated turn is dropped as a whole: neither its user message nor its
        // assistant message may survive, so a call can never be separated from its result.
        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant],
            transcript.Select(static message => message.Role));
        Assert.Equal("second", MessageText(transcript[1]));
        Assert.Equal("second answer", MessageText(transcript[2]));
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void ToolAuditTrail_ProjectsOneEntryPerCorrelatedResult()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 4);
        ChatMessage user = history.AddUserMessage(1, "question", out _);
        Assert.True(history.CommitGeneratedResponse(1, user, "answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1");

        ToolAuditRecord audit = Assert.Single(history.GetToolAuditTrailSnapshot());
        Assert.Equal(1, audit.TurnId);
        Assert.Equal("call-1", audit.CallId);
        Assert.Equal("srv", audit.ToolId.ServerAlias);
        Assert.Equal("lookup", audit.ToolId.ToolName);
        Assert.Equal(AssistantToolResultKind.Success, audit.ResultKind);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ToolCorrelation_SwitchesTrimmingToTurnCountRegardlessOfWhichSideIsPresent(
        bool includeCall,
        bool includeResult)
    {
        // Message-based trimming counts messages; turn-atomic trimming counts turns. Three
        // thinking-only turns are 3 messages (under the 4-message limit) but 3 records (over the
        // 2-turn limit), so only the correlated path trims here. A turn carrying just a call must
        // already switch: otherwise a pending call could be trimmed away from its future result.
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(1, "first", out _);
        history.AddUserMessage(2, "second", out _);
        ChatMessage third = history.AddUserMessage(3, "third", out _);
        AttachToolCorrelation(history, turnId: 1, callId: "call-1", includeCall, includeResult);

        Assert.True(history.CommitGeneratedResponse(3, third, string.Empty));

        ChatMessage[] transcript = history.GetModelTranscriptSnapshot();
        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.User],
            transcript.Select(static message => message.Role));
        Assert.Equal("second", MessageText(transcript[1]));
        Assert.Equal("third", MessageText(transcript[2]));
    }

    [Fact]
    public void TextOnlyTurns_KeepMessageBasedTrimmingAndAreNotTrimmedByTurnCount()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        history.AddUserMessage(1, "first", out _);
        history.AddUserMessage(2, "second", out _);
        ChatMessage third = history.AddUserMessage(3, "third", out _);

        // Same shape as above without tool correlation: 3 messages stay under the
        // message-based limit, so nothing is trimmed.
        Assert.True(history.CommitGeneratedResponse(3, third, string.Empty));

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.User, ChatRole.User],
            history.GetModelTranscriptSnapshot().Select(static message => message.Role));
    }

    [Fact]
    public void ToolAuditTrail_SkipsAResultWithoutItsCorrelatedCall()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 4);
        ChatMessage user = history.AddUserMessage(1, "question", out _);
        Assert.True(history.CommitGeneratedResponse(1, user, "answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1", includeCall: false);

        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    /// <summary>
    /// Adds tool call/result correlation to an existing turn record. Production never does this in
    /// this round; the test proves the record shape and the atomic-trim invariant already hold.
    /// </summary>
    private static void AttachToolCorrelation(
        ConversationHistoryStore history,
        long turnId,
        string callId,
        bool includeCall = true,
        bool includeResult = true)
    {
        Assert.True(includeCall || includeResult);
        var records = (List<ConversationTurnRecord>)typeof(ConversationHistoryStore)
            .GetField("_turnRecords", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(history)!;
        int index = records.FindIndex(record => record.TurnId == turnId);
        Assert.True(index >= 0);
        var toolId = new AssistantToolId("srv", "lookup");
        records[index] = records[index] with
        {
            ToolCalls = includeCall
                ? [new AssistantToolCall(callId, toolId, OwnedJsonValue.Parse("{}"))]
                : [],
            ToolResults = includeResult
                ? [new AssistantToolResult(callId, AssistantToolResultKind.Success, null)]
                : [],
        };
    }

    [Fact]
    public void Rollback_RemovesPendingTurnFromBothTextViews()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        ChatMessage user = history.AddUserMessage(4, "pending", out _);

        history.RemoveUnansweredUserMessage(4, user);

        Assert.Single(history.GetModelTranscriptSnapshot());
        Assert.Empty(history.GetUserConversationHistorySnapshot());
        Assert.Empty(history.GetToolAuditTrailSnapshot());
    }

    [Fact]
    public void AddUserMessage_ReturnsPromptWithSystemAndUser()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);

        ChatMessage user = history.AddUserMessage(1, "你好", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("你好", MessageText(user));
        Assert.Same(user, prompt[1]);
    }

    [Fact]
    public void CommitGeneratedResponse_IsIdempotentAndIncludedInNextPrompt()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);
        ChatMessage user = history.AddUserMessage(1, "你好", out _);

        Assert.True(history.CommitGeneratedResponse(1, user, "好的。"));
        Assert.False(history.CommitGeneratedResponse(1, user, "重复。"));

        history.AddUserMessage(2, "第二轮", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("好的。", MessageText(prompt[2]));
        Assert.DoesNotContain(prompt, m => MessageText(m) == "重复。");
    }

    [Fact]
    public void RemoveUnansweredUserMessage_RemovesOnlyUncommittedUser()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);
        ChatMessage first = history.AddUserMessage(1, "第一轮", out _);
        history.RemoveUnansweredUserMessage(1, first);

        history.AddUserMessage(2, "第二轮", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("第二轮", MessageText(prompt[1]));
    }

    [Fact]
    public void RemoveUnansweredUserMessage_DoesNotRemoveCommittedExchange()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 3);
        ChatMessage first = history.AddUserMessage(1, "第一轮", out _);
        history.CommitGeneratedResponse(1, first, "回答一");

        history.RemoveUnansweredUserMessage(1, first);
        history.AddUserMessage(2, "第二轮", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
        Assert.Equal("回答一", MessageText(prompt[2]));
    }

    [Fact]
    public void CommitGeneratedResponse_TrimsOldestCompleteTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);
        for (long turn = 1; turn <= 3; turn++)
        {
            ChatMessage user = history.AddUserMessage(turn, $"用户{turn}", out _);
            history.CommitGeneratedResponse(turn, user, $"回答{turn}");
            history.ForgetCommittedResponseTurn(turn);
        }

        history.AddUserMessage(4, "用户4", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
        Assert.DoesNotContain(prompt, m => MessageText(m) is "用户1" or "回答1");
        Assert.Contains(prompt, m => MessageText(m) == "用户2");
        Assert.Contains(prompt, m => MessageText(m) == "回答3");
    }

    [Fact]
    public void CommitGeneratedResponse_TrimsDeduplicationEntriesWithExpiredTextOnlyTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 2);

        for (long turn = 1; turn <= 3; turn++)
        {
            ChatMessage user = history.AddUserMessage(turn, $"用户{turn}", out _);
            Assert.True(history.CommitGeneratedResponse(turn, user, $"回答{turn}"));
        }

        Assert.Equal(2, history.CommittedResponseTurnCount);
    }

    [Fact]
    public void CommitGeneratedResponse_TrimsDeduplicationEntriesWithExpiredToolTurns()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        AttachToolCorrelation(history, turnId: 1, callId: "call-1");

        ChatMessage second = history.AddUserMessage(2, "second", out _);
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));

        Assert.Equal(1, history.CommittedResponseTurnCount);
    }

    [Fact]
    public void CommitGeneratedResponse_MaximumTurnLimitDoesNotOverflow()
    {
        var history = new ConversationHistoryStore("system", maxHistoryTurns: int.MaxValue);
        ChatMessage user = history.AddUserMessage(1, "第一轮", out _);

        bool committed = history.CommitGeneratedResponse(1, user, "回答一");
        history.AddUserMessage(2, "第二轮", out ChatMessage[] prompt);

        Assert.True(committed);
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(m => m.Role));
    }

    private static string MessageText(ChatMessage message) =>
        string.Concat(message.Content.OfType<TextContent>().Select(c => c.Text));
}
