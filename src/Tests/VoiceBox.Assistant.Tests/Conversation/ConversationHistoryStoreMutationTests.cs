using VoiceBox.Assistant.Modules.Conversation.Models;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class ConversationHistoryStoreMutationTests
{
    [Fact]
    public void Constructor_RejectsNonPositiveHistoryLimit()
    {
        ArgumentOutOfRangeException zero = Assert.Throws<ArgumentOutOfRangeException>(
            () => new ConversationHistoryStore("system", 0));
        ArgumentOutOfRangeException negative = Assert.Throws<ArgumentOutOfRangeException>(
            () => new ConversationHistoryStore("system", -1));

        Assert.Equal("maxHistoryTurns", zero.ParamName);
        Assert.StartsWith("Max history turns must be >= 1.", zero.Message, StringComparison.Ordinal);
        Assert.Equal("maxHistoryTurns", negative.ParamName);
        Assert.StartsWith("Max history turns must be >= 1.", negative.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CommitAfterUnansweredRemoval_ReinsertsTheSameUserBeforeItsResponse()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(11, "late user", out _);
        history.RemoveUnansweredUserMessage(11, user);

        Assert.True(history.CommitGeneratedResponse(11, user, "late response"));
        history.AddUserMessage(12, "next", out ChatMessage[] prompt);

        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User],
            prompt.Select(static message => message.Role));
        Assert.Equal("late user", Text(prompt[1]));
        Assert.Equal("late response", Text(prompt[2]));
        Assert.Equal("next", Text(prompt[3]));
    }

    [Fact]
    public void KernelOwnedCommit_UsesTheExactPendingTurnAndUpdatesItsRecord()
    {
        var history = new ConversationHistoryStore("system", 4);
        history.AddUserMessage(1, "first", out _);
        history.AddUserMessage(2, "second", out _);

        Assert.True(history.CommitGeneratedResponse(1, "first answer"));
        Assert.False(history.CommitGeneratedResponse(1, "duplicate"));
        Assert.False(history.CommitGeneratedResponse(99, "missing"));

        ConversationTurnRecord[] records = TurnRecords(history);
        Assert.Equal(2, records.Length);
        Assert.Equal("first answer", Text(Assert.IsType<ChatMessage>(records[0].AssistantMessage)));
        Assert.Null(records[1].AssistantMessage);
    }

    [Fact]
    public void CommitRecordCorrelation_RequiresBothTurnAndUserIdentity()
    {
        var history = new ConversationHistoryStore("system", 4);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        history.AddUserMessage(2, "second", out _);

        Assert.True(history.CommitGeneratedResponse(2, first, "cross-correlated"));

        ConversationTurnRecord[] records = TurnRecords(history);
        Assert.Equal(3, records.Length);
        Assert.Null(records[0].AssistantMessage);
        Assert.Null(records[1].AssistantMessage);
        Assert.Equal(2, records[2].TurnId);
        Assert.Same(first, records[2].UserMessage);
        Assert.Equal(
            "cross-correlated",
            Text(Assert.IsType<ChatMessage>(records[2].AssistantMessage)));
    }

    [Fact]
    public void PendingRecordRemoval_RequiresTurnUnansweredStateAndUserIdentity()
    {
        var history = new ConversationHistoryStore("system", 4);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        ChatMessage replacement = history.AddUserMessage(1, "replacement", out _);
        history.AddUserMessage(2, "other turn", out _);

        history.RemoveUnansweredUserMessage(1, first);

        ConversationTurnRecord[] records = TurnRecords(history);
        Assert.Equal(2, records.Length);
        Assert.Contains(records, record => record.TurnId == 1 && ReferenceEquals(record.UserMessage, replacement));
        Assert.Contains(records, record => record.TurnId == 2);
        Assert.DoesNotContain(records, record => ReferenceEquals(record.UserMessage, first));
    }

    [Fact]
    public void Trim_RemovesRecordsWhoseUserMessagesAreNoLongerRetained()
    {
        var history = new ConversationHistoryStore("system", 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        ChatMessage second = history.AddUserMessage(2, "second", out _);
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));

        ConversationTurnRecord record = Assert.Single(TurnRecords(history));
        Assert.Equal(2, record.TurnId);
        Assert.Same(second, record.UserMessage);
    }

    [Fact]
    public void EmptyResponse_CommitsTheUserAndPreventsSubsequentRollback()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(1, "thinking only", out _);

        Assert.True(history.CommitGeneratedResponse(1, user, string.Empty));
        history.RemoveUnansweredUserMessage(1, user);
        history.AddUserMessage(2, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("thinking only", Text(prompt[1]));
        Assert.Equal("next", Text(prompt[2]));
    }

    [Fact]
    public void EmptyResponse_ForgetsItsCommitWithoutRetainingAPendingTurnMapping()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(1, "thinking only", out _);

        Assert.Equal(1, PendingTurnCount(history));
        Assert.True(history.CommitGeneratedResponse(1, user, string.Empty));
        Assert.Equal(0, PendingTurnCount(history));

        history.ForgetCommittedResponseTurn(1);
        history.RemoveUnansweredUserMessage(1);
        history.AddUserMessage(2, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("thinking only", Text(prompt[1]));
    }

    [Fact]
    public void EmptyResponse_CanBeExplicitlyRolledBackAfterForgettingItsCommit()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(1, "thinking only", out _);

        Assert.True(history.CommitGeneratedResponse(1, user, string.Empty));
        history.ForgetCommittedResponseTurn(1);
        history.RemoveUnansweredUserMessage(1, user);
        history.AddUserMessage(2, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("next", Text(prompt[1]));
    }

    [Fact]
    public void Rollback_WithAStaleCallerMessageKeepsTheOriginalHistoryEntryAndClearsTheMapping()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage original = history.AddUserMessage(7, "original", out _);
        history.RemoveUnansweredUserMessage(7, ChatMessage.User("stale"));
        ChatMessage replacement = history.AddUserMessage(7, "replacement", out _);

        history.RemoveUnansweredUserMessage(7);
        history.AddUserMessage(8, "next", out ChatMessage[] prompt);

        Assert.Contains(prompt, message => Text(message) == Text(original));
        Assert.DoesNotContain(prompt, message => Text(message) == Text(replacement));
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("original", Text(prompt[1]));
        Assert.Equal("next", Text(prompt[2]));
    }

    [Fact]
    public void Rollback_ClearsPendingTurnMappingsOnEveryTerminalPath()
    {
        var missing = new ConversationHistoryStore("system", 2);
        missing.AddUserMessage(1, "original", out _);
        missing.RemoveUnansweredUserMessage(1, ChatMessage.User("not-the-tracked-instance"));
        Assert.Equal(0, PendingTurnCount(missing));

        var answered = new ConversationHistoryStore("system", 2);
        ChatMessage answeredUser = answered.AddUserMessage(2, "answered", out _);
        Assert.True(answered.CommitGeneratedResponse(2, answeredUser, "response"));
        answered.ForgetCommittedResponseTurn(2);
        answered.RemoveUnansweredUserMessage(2, answeredUser);
        Assert.Equal(0, PendingTurnCount(answered));

        var unanswered = new ConversationHistoryStore("system", 2);
        ChatMessage unansweredUser = unanswered.AddUserMessage(3, "unanswered", out _);
        unanswered.RemoveUnansweredUserMessage(3, unansweredUser);
        Assert.Equal(0, PendingTurnCount(unanswered));
    }

    [Fact]
    public void Rollback_ClearsThePendingMappingWhenTheUserAlreadyHasAnAdjacentResponse()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(1, "question", out _);

        Assert.True(history.CommitGeneratedResponse(2, user, "answer"));
        Assert.Equal(1, PendingTurnCount(history));
        history.RemoveUnansweredUserMessage(1, user);

        Assert.Equal(0, PendingTurnCount(history));
    }

    [Fact]
    public void Rollback_DoesNotRemoveAUserThatAlreadyHasAnAssistantResponseAfterForget()
    {
        var history = new ConversationHistoryStore("system", 2);
        ChatMessage user = history.AddUserMessage(4, "question", out _);
        Assert.True(history.CommitGeneratedResponse(4, user, "answer"));
        history.ForgetCommittedResponseTurn(4);

        history.RemoveUnansweredUserMessage(4, user);
        history.AddUserMessage(5, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("question", Text(prompt[1]));
        Assert.Equal("answer", Text(prompt[2]));
    }

    [Fact]
    public void Trim_RemovesAnOrphanedAssistantSoTheHistoryAlwaysStartsWithAUser()
    {
        var history = new ConversationHistoryStore("system", 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        history.ForgetCommittedResponseTurn(1);
        history.RemoveUnansweredUserMessage(1, first);

        ChatMessage second = history.AddUserMessage(2, "second", out _);
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));
        history.AddUserMessage(3, "next", out ChatMessage[] prompt);

        Assert.Equal(ChatRole.System, prompt[0].Role);
        Assert.Equal(ChatRole.User, prompt[1].Role);
        Assert.Equal("second", Text(prompt[1]));
        Assert.DoesNotContain(prompt, message => Text(message) == "first answer");
    }

    [Fact]
    public void OutOfOrderCommits_TrimEveryLeadingOrphanedAssistantMessage()
    {
        var history = new ConversationHistoryStore("system", 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        ChatMessage second = history.AddUserMessage(2, "second", out _);

        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));
        history.AddUserMessage(3, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.Assistant, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("second answer", Text(prompt[1]));
        Assert.Equal("next", Text(prompt[2]));
        Assert.DoesNotContain(prompt, message => Text(message) == "first answer");
    }

    [Fact]
    public void Trim_AtTheExactLimitDoesNotRunOrphanCleanup()
    {
        var history = new ConversationHistoryStore("system", 1);
        ChatMessage first = history.AddUserMessage(1, "first", out _);
        ChatMessage second = history.AddUserMessage(2, "second", out _);
        Assert.True(history.CommitGeneratedResponse(1, first, "first answer"));
        Assert.True(history.CommitGeneratedResponse(2, second, "second answer"));

        Assert.True(history.CommitGeneratedResponse(3, ChatMessage.User("third"), string.Empty));
        history.AddUserMessage(4, "next", out ChatMessage[] prompt);

        Assert.Equal([ChatRole.System, ChatRole.Assistant, ChatRole.User, ChatRole.User], prompt.Select(static message => message.Role));
        Assert.Equal("second answer", Text(prompt[1]));
        Assert.Equal("third", Text(prompt[2]));
        Assert.Equal("next", Text(prompt[3]));
    }

    private static int PendingTurnCount(ConversationHistoryStore history)
    {
        var field = typeof(ConversationHistoryStore).GetField(
            "_turnUserMessages",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var pending = Assert.IsType<Dictionary<long, ChatMessage>>(field?.GetValue(history));
        return pending.Count;
    }

    private static ConversationTurnRecord[] TurnRecords(ConversationHistoryStore history)
    {
        var field = typeof(ConversationHistoryStore).GetField(
            "_turnRecords",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var records = Assert.IsType<List<ConversationTurnRecord>>(field?.GetValue(history));
        return records.ToArray();
    }

    private static string Text(ChatMessage message)
        => string.Concat(message.Content.OfType<TextContent>().Select(static content => content.Text));
}
