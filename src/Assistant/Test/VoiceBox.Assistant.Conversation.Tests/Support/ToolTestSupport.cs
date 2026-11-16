using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests.Support;

/// <summary>Shared building blocks for tests that drive a tool-enabled session.</summary>
internal static class ToolTestSupport
{
    /// <summary>Upper bound for anything a test expects to happen. Passing tests never wait this long.</summary>
    public static readonly TimeSpan Patience = TimeSpan.FromSeconds(10);

    public static ConversationRequest Request(string text) => new(Guid.NewGuid(), text);

    public static async IAsyncEnumerable<LlmTurnEvent> Events(params LlmTurnEvent[] events)
    {
        await Task.Yield();
        foreach (LlmTurnEvent item in events)
            yield return item;
    }

    /// <summary>The first model round calls the tool; every later round answers "完成".</summary>
    public static Func<int, LlmToolTurnRequest, IAsyncEnumerable<LlmTurnEvent>> CallThenAnswer(
        string exposedToolName,
        string argumentsJson = "{}")
        => (round, _) => round == 1
            ? Events(new LlmToolCallEvent("c1", exposedToolName, argumentsJson), new LlmTurnCompletedEvent())
            : Events(new LlmTurnTextDelta("完成"), new LlmTurnCompletedEvent());

    public static async Task<IConversationSession> StartAsync(ConversationBuilder builder)
    {
        IConversationSession session = builder.Build();
        await ((IConversationLifecycle)session).StartAsync();
        return session;
    }

    public static Task<List<ConversationEvent>> RunTurnAsync(IConversationSession session, string text)
        => ConversationEvents.CollectAsync(session.StreamAsync(Request(text))).WaitAsync(Patience);

    public static string Completed(IReadOnlyList<ConversationEvent> events) => Assert.IsType<ConversationCompleted>(events[^1]).Text;

    /// <summary>Reads the next result a session published after a request's foreground stream ended.</summary>
    public static async Task<string> NextBackgroundTextAsync(IConversationSession session)
    {
        using var timeout = new CancellationTokenSource(Patience);
        var output = (IConversationBackgroundOutput)session;
        await foreach (ConversationBackgroundMessage message in output.ReadBackgroundMessagesAsync(timeout.Token))
            return message.Text;
        throw new InvalidOperationException("The background channel completed without a message.");
    }

    public static async Task WaitUntilAsync(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(Patience);
        while (!condition())
            await Task.Delay(10, timeout.Token);
    }

    public static string TextOf(ChatMessage message)
        => string.Concat(message.Content.OfType<TextContent>().Select(content => content.Text));

    public static ToolCallContent ToolCallOf(IEnumerable<ChatMessage> prompt)
        => Assert.Single(prompt.SelectMany(message => message.Content).OfType<ToolCallContent>());

    public static ToolResultContent ToolResultOf(IEnumerable<ChatMessage> prompt)
        => Assert.Single(prompt.SelectMany(message => message.Content).OfType<ToolResultContent>());

    public static ToolResultContent[] ToolResultsOf(IEnumerable<ChatMessage> prompt)
        => prompt.SelectMany(message => message.Content).OfType<ToolResultContent>().ToArray();
}
