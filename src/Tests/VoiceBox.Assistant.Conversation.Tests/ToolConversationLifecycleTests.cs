using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ToolConversationLifecycleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ShutdownCancelsStartupAndWaitsForItsCleanup(bool dispose)
    {
        var provider = new BlockingProvider();
        IConversationSession session = new ConversationBuilder(new RecordingLlmService())
            .UseTools(tools => tools.AddToolProvider(provider)).Build();
        var lifecycle = (IConversationLifecycle)session;
        Task startup = lifecycle.StartAsync();
        Task shutdown = Task.CompletedTask;
        try
        {
            await provider.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(5));
            shutdown = dispose ? session.DisposeAsync().AsTask() : lifecycle.StopAsync();
            await provider.CancellationObserved.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.False(shutdown.IsCompleted);
            Assert.False(provider.Disposed);
            if (!dispose)
            {
                Assert.Same(shutdown, lifecycle.StopAsync());
                await Assert.ThrowsAsync<InvalidOperationException>(() => lifecycle.StartAsync());
                Assert.Throws<InvalidOperationException>(() => session.StreamAsync(
                    new ConversationRequest(Guid.NewGuid(), "停止期间的请求")));
            }

            provider.ReleaseCleanup.TrySetResult();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => startup);
            await shutdown.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.False(provider.Active);

            if (dispose)
            {
                Assert.True(provider.Disposed);
                await Assert.ThrowsAsync<ObjectDisposedException>(() => lifecycle.StartAsync());
            }
            else
            {
                provider.ReleaseStart.TrySetResult();
                await lifecycle.StartAsync();
                Assert.True(provider.Active);
                await lifecycle.StopAsync();
                Assert.False(provider.Active);
            }
        }
        finally
        {
            provider.ReleaseStart.TrySetResult();
            provider.ReleaseCleanup.TrySetResult();
            try { await startup; } catch (OperationCanceledException) { }
            await shutdown;
            await session.DisposeAsync();
        }
    }

    [Fact]
    public async Task StopWaitsForStartupThatIgnoresCancellationAndClosesItsResources()
    {
        var provider = new BlockingProvider { IgnoreStartupCancellation = true };
        await using IConversationSession session = new ConversationBuilder(new RecordingLlmService())
            .UseTools(tools => tools.AddToolProvider(provider)).Build();
        var lifecycle = (IConversationLifecycle)session;
        Task startup = lifecycle.StartAsync();
        await provider.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Task stop = lifecycle.StopAsync();
        try
        {
            Assert.False(stop.IsCompleted);
            Assert.True(provider.Active);
            provider.ReleaseStart.TrySetResult();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => startup);
            await stop.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.False(provider.Active);

            await lifecycle.StartAsync();
            Assert.True(provider.Active);
            await lifecycle.StopAsync();
            Assert.False(provider.Active);
        }
        finally
        {
            provider.ReleaseStart.TrySetResult();
            try { await startup; } catch (OperationCanceledException) { }
            await stop;
        }
    }

    [Fact]
    public async Task CallerCanceledStartupCleansUpAndAllowsRestart()
    {
        var provider = new BlockingProvider();
        await using IConversationSession session = new ConversationBuilder(new RecordingLlmService())
            .UseTools(tools => tools.AddToolProvider(provider)).Build();
        var lifecycle = (IConversationLifecycle)session;
        using var cancellation = new CancellationTokenSource();
        Task startup = lifecycle.StartAsync(cancellation.Token);
        try
        {
            await provider.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(5));
            await cancellation.CancelAsync();
            await provider.CancellationObserved.Task.WaitAsync(TimeSpan.FromSeconds(5));
            provider.ReleaseCleanup.TrySetResult();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => startup);
            Assert.False(provider.Active);

            provider.ReleaseStart.TrySetResult();
            await lifecycle.StartAsync();
            Assert.True(provider.Active);
            await lifecycle.StopAsync();
            Assert.False(provider.Active);
        }
        finally
        {
            provider.ReleaseStart.TrySetResult();
            provider.ReleaseCleanup.TrySetResult();
            try { await startup; } catch (OperationCanceledException) { }
        }
    }

    private sealed class BlockingProvider : IConversationToolProvider
    {
        public TaskCompletionSource StartEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource CancellationObserved { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseStart { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseCleanup { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool Active { get; private set; }
        public bool Disposed { get; private set; }
        public bool IgnoreStartupCancellation { get; init; }

        public async Task StartAsync(CancellationToken cancellationToken = default)
        {
            Active = true;
            StartEntered.TrySetResult();
            try
            {
                await ReleaseStart.Task.WaitAsync(
                    IgnoreStartupCancellation ? CancellationToken.None : cancellationToken);
            }
            catch (OperationCanceledException)
            {
                CancellationObserved.TrySetResult();
                await ReleaseCleanup.Task;
                Active = false;
                throw;
            }
        }

        public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(
            CancellationToken cancellationToken = default)
            => ValueTask.FromResult<IReadOnlyList<ConversationToolDefinition>>(
                [new(new("external", "probe"), new ConversationToolDescriptor(
                    "probe", "Probe", "Probe tool", "{\"type\":\"object\"}"))]);

        public ValueTask<ConversationToolResult> InvokeAsync(
            ConversationToolCall call, CancellationToken cancellationToken = default)
            => ValueTask.FromResult(new ConversationToolResult(call.CallId, ConversationToolResultKind.Success, "{}"));

        public Task StopAsync()
        {
            Active = false;
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync()
        {
            Active = false;
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }
}
