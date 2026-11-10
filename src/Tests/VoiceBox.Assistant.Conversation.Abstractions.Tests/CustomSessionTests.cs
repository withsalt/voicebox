using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Conversation.Abstractions;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tests;

public sealed class CustomSessionTests
{
    private static async Task<List<ConversationEvent>> CollectAsync(
        IAsyncEnumerable<ConversationEvent> stream,
        CancellationToken cancellationToken = default)
    {
        var events = new List<ConversationEvent>();
        await foreach (ConversationEvent item in stream.WithCancellation(cancellationToken))
            events.Add(item);
        return events;
    }

    private static string Completed(IReadOnlyList<ConversationEvent> events)
        => Assert.IsType<ConversationCompleted>(events[^1]).Text;

    [Fact]
    public async Task ACustomSessionCanOwnItsOwnMemoryAndRunTwoTurns()
    {
        await using var session = new MemorySession();

        List<ConversationEvent> first = await CollectAsync(session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "我叫小明")));
        Assert.Equal("记住了:小明", Completed(first));
        Assert.Equal(2, first.OfType<ConversationTextDelta>().Count());

        List<ConversationEvent> second = await CollectAsync(session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "我叫什么")));
        Assert.Equal("你叫小明", Completed(second));

        Assert.Equal("memory-session", session.ConversationId);
        Assert.DoesNotContain(
            AppDomain.CurrentDomain.GetAssemblies(),
            assembly => assembly.GetName().Name == "VoiceBox.Assistant.Conversation");
    }

    [Fact]
    public async Task CancellationPropagatesThroughThePublicContract()
    {
        await using var session = new MemorySession { Delay = new TaskCompletionSource() };
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => CollectAsync(
            session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "取消"), cancellation.Token),
            cancellation.Token));
    }

    [Fact]
    public async Task TheCallerOwnsAndCanDisposeTheSession()
    {
        var session = new MemorySession();
        await CollectAsync(session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "你好")));
        Assert.False(session.Disposed);

        await session.DisposeAsync();
        Assert.True(session.Disposed);
        await session.DisposeAsync();
        Assert.True(session.Disposed);
    }

    private sealed class MemorySession : IConversationSession
    {
        private string? _name;

        public string ConversationId => "memory-session";

        public TaskCompletionSource? Delay { get; init; }

        public bool Disposed { get; private set; }

        public async IAsyncEnumerable<ConversationEvent> StreamAsync(
            ConversationRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (Delay is not null)
                await Delay.Task.WaitAsync(cancellationToken);

            cancellationToken.ThrowIfCancellationRequested();
            string answer;
            if (request.Text.Contains("我叫什么", StringComparison.Ordinal) && _name is not null)
            {
                answer = $"你叫{_name}";
            }
            else if (request.Text.Contains("我叫", StringComparison.Ordinal))
            {
                _name = request.Text["我叫".Length..];
                answer = $"记住了:{_name}";
            }
            else
            {
                answer = "你好";
            }

            yield return new ConversationTextDelta(answer[..1]);
            yield return new ConversationTextDelta(answer[1..]);
            yield return new ConversationCompleted(answer);
        }

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }
}
