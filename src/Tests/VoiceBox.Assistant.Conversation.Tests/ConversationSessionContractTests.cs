using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ConversationSessionContractTests
{
    private static ConversationRequest Request(string text)
        => new(Guid.NewGuid(), text);

    private static string TextOf(ChatMessage message)
        => string.Concat(message.Content.OfType<TextContent>().Select(content => content.Text));

    [Fact]
    public async Task SecondTurnPromptContainsCommittedFirstTurn()
    {
        var llm = new RecordingLlmService { ResponseText = "第一答" };
        await using IConversationSession session = new ConversationBuilder(llm)
            .UseOptions(new LlmConversationOptions { SystemPrompt = "系统" })
            .Build();

        List<ConversationEvent> first = await ConversationEvents.CollectAsync(
            session.StreamAsync(Request("第一问")));
        Assert.Equal("第一答", Assert.IsType<ConversationCompleted>(first[^1]).Text);

        List<ConversationEvent> second = await ConversationEvents.CollectAsync(
            session.StreamAsync(Request("第二问")));

        Assert.Equal(2, llm.Prompts.Count);
        ChatMessage[] prompt = llm.Prompts[1];
        Assert.Equal(4, prompt.Length);
        Assert.Equal(ChatRole.System, prompt[0].Role);
        Assert.Equal("系统", TextOf(prompt[0]));
        Assert.Equal("第一问", TextOf(prompt[1]));
        Assert.Equal(ChatRole.Assistant, prompt[2].Role);
        Assert.Equal("第一答", TextOf(prompt[2]));
        Assert.Equal("第二问", TextOf(prompt[3]));
        Assert.Equal("第一答", Assert.IsType<ConversationCompleted>(second[^1]).Text);
    }

    [Fact]
    public async Task ThinkingIsObservableButNeverEntersHistory()
    {
        var llm = new RecordingLlmService
        {
            ThinkingText = "内部推理",
            ResponseText = "最终回答",
        };
        await using IConversationSession session = new ConversationBuilder(llm)
            .UseOptions(new LlmConversationOptions { EnableThinking = true })
            .Build();

        List<ConversationEvent> updates = await ConversationEvents.CollectAsync(
            session.StreamAsync(Request("问题")));

        Assert.Contains(updates, update =>
            update is ConversationThinkingDelta { Text: "内部推理" });
        Assert.Contains(updates, update =>
            update is ConversationTextDelta { Text: "最终回答" });
        Assert.Equal("最终回答", Assert.IsType<ConversationCompleted>(updates[^1]).Text);
        Assert.DoesNotContain(updates, update =>
            update is ConversationTextDelta delta && delta.Text.Contains("内部推理", StringComparison.Ordinal));

        await ConversationEvents.CollectAsync(session.StreamAsync(Request("继续")));
        string history = string.Join("\n", llm.Prompts[1].Select(TextOf));
        Assert.DoesNotContain("内部推理", history, StringComparison.Ordinal);
        Assert.Contains("最终回答", history, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CancellationBeforeCommitRollsBackThePendingUserMessage()
    {
        var llm = new RecordingLlmService();
        llm.Chunks =
        [
            new LlmTextChunk { Text = "半截", Kind = LlmChunkKind.Text },
            new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true },
        ];
        llm.BlockAfterFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        using var cancellation = new CancellationTokenSource();
        await using (IAsyncEnumerator<ConversationEvent> enumerator =
            session.StreamAsync(Request("会被取消"), cancellation.Token).GetAsyncEnumerator())
        {
            Assert.True(await enumerator.MoveNextAsync());
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                async () => await enumerator.MoveNextAsync());
        }

        llm.BlockAfterFirstChunk = null;
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("下一轮")));

        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.DoesNotContain(prompt, message => TextOf(message) == "会被取消");
        Assert.Contains(prompt, message => TextOf(message) == "下一轮");
    }

    [Fact]
    public async Task ConsumerCanStopImmediatelyAfterCompletedAndTheTurnStaysCommitted()
    {
        var llm = new RecordingLlmService { ResponseText = "完整回答" };
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        await using (IAsyncEnumerator<ConversationEvent> enumerator =
            session.StreamAsync(Request("第一轮")).GetAsyncEnumerator())
        {
            while (await enumerator.MoveNextAsync())
            {
                if (enumerator.Current is ConversationCompleted)
                    break;
            }
        }

        await ConversationEvents.CollectAsync(session.StreamAsync(Request("第二轮")));

        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.Contains(prompt, message =>
            message.Role == ChatRole.User && TextOf(message) == "第一轮");
        Assert.Contains(prompt, message =>
            message.Role == ChatRole.Assistant && TextOf(message) == "完整回答");
    }

    [Fact]
    public async Task DisposingTheStreamEarlyRollsBackTheTurn()
    {
        var llm = new RecordingLlmService();
        llm.Chunks =
        [
            new LlmTextChunk { Text = "半截", Kind = LlmChunkKind.Text },
            new LlmTextChunk { Text = "结尾", Kind = LlmChunkKind.Text, IsFinal = true },
        ];
        llm.BlockAfterFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        await using (IAsyncEnumerator<ConversationEvent> enumerator =
            session.StreamAsync(Request("提前退出")).GetAsyncEnumerator())
        {
            Assert.True(await enumerator.MoveNextAsync());
        }

        llm.BlockAfterFirstChunk = null;
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("下一轮")));

        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.DoesNotContain(prompt, message => TextOf(message) == "提前退出");
        Assert.DoesNotContain(prompt, message => TextOf(message) == "半截");
    }

    [Fact]
    public async Task RequestsOnOneSessionAreSerialized()
    {
        var llm = new RecordingLlmService();
        llm.Chunks =
        [
            new LlmTextChunk { Text = "第一段", Kind = LlmChunkKind.Text },
            new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true },
        ];
        llm.BlockAfterFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        Task<List<ConversationEvent>> first = ConversationEvents.CollectAsync(
            session.StreamAsync(Request("串行一")));
        await WaitUntilAsync(() => llm.Prompts.Count == 1);

        Task<List<ConversationEvent>> second = ConversationEvents.CollectAsync(
            session.StreamAsync(Request("串行二")));
        await Task.Delay(50);
        Assert.Single(llm.Prompts);

        llm.BlockAfterFirstChunk.TrySetResult();
        List<ConversationEvent>[] results = await Task.WhenAll(first, second);
        Assert.Equal(2, llm.Prompts.Count);
        Assert.Contains("串行二", string.Join("\n", llm.Prompts[1].Select(TextOf)), StringComparison.Ordinal);
        Assert.Equal("第一段", Assert.IsType<ConversationCompleted>(results[0][^1]).Text);
        Assert.Equal("第一段", Assert.IsType<ConversationCompleted>(results[1][^1]).Text);
    }

    [Fact]
    public async Task SeparateSessionsDoNotShareHistory()
    {
        var firstLlm = new RecordingLlmService { ResponseText = "甲" };
        var secondLlm = new RecordingLlmService { ResponseText = "乙" };
        await using IConversationSession first = new ConversationBuilder(firstLlm)
            .UseConversationId("session-a")
            .Build();
        await using IConversationSession second = new ConversationBuilder(secondLlm)
            .UseConversationId("session-b")
            .Build();

        await ConversationEvents.CollectAsync(first.StreamAsync(Request("只问甲")));
        await ConversationEvents.CollectAsync(second.StreamAsync(Request("只问乙")));

        Assert.Equal("session-a", first.ConversationId);
        Assert.Equal("session-b", second.ConversationId);
        Assert.DoesNotContain("只问乙",
            string.Join("\n", firstLlm.Prompts[0].Select(TextOf)), StringComparison.Ordinal);
        Assert.DoesNotContain("只问甲",
            string.Join("\n", secondLlm.Prompts[0].Select(TextOf)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task EmptyReplyStillCommitsTheUserMessage()
    {
        var llm = new RecordingLlmService { ResponseText = string.Empty };
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        List<ConversationEvent> updates = await ConversationEvents.CollectAsync(
            session.StreamAsync(Request("第一问")));
        Assert.Equal(string.Empty, Assert.IsType<ConversationCompleted>(updates[^1]).Text);

        await ConversationEvents.CollectAsync(session.StreamAsync(Request("第二问")));

        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.Contains(prompt, message =>
            message.Role == ChatRole.User && TextOf(message) == "第一问");
        Assert.DoesNotContain(prompt, message => message.Role == ChatRole.Assistant);
    }

    [Fact]
    public async Task ProviderConversationIsReusedAndDisposedExactlyOnceWhileTheBorrowedServiceIsNotDisposed()
    {
        var llm = new ConversationalRecordingLlmService();
        IConversationSession session = new ConversationBuilder(llm).Build();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("一")));
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("二")));

        Assert.Equal(1, llm.Conversation.CreatedCount);
        Assert.Equal(2, llm.Conversation.StreamCount);
        Assert.Equal(0, llm.Conversation.DisposeCount);

        await session.DisposeAsync();
        Assert.Equal(1, llm.Conversation.DisposeCount);
    }

    [Fact]
    public async Task BorrowedLlmServiceIsNotDisposedWithTheSession()
    {
        var llm = new RecordingLlmService();
        IConversationSession session = new ConversationBuilder(llm).Build();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("问题")));
        await session.DisposeAsync();

        Assert.Equal(0, llm.DisposeCount);
    }

    [Fact]
    public async Task OptionsAreSnapshotAtConfigurationTime()
    {
        var llm = new RecordingLlmService();
        var options = new LlmConversationOptions
        {
            SystemPrompt = "原始提示",
            MaxResponseTokens = 32,
        };
        ConversationBuilder builder = new ConversationBuilder(llm).UseOptions(options);
        options.SystemPrompt = "被修改的提示";
        options.MaxResponseTokens = 99;

        await using IConversationSession session = builder.Build();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("问题")));

        Assert.Equal("原始提示", TextOf(llm.Prompts[0][0]));
        Assert.Equal(32, llm.Options[0].MaxTokens);
    }

    [Fact]
    public async Task ContextProvidersRunOncePerTurnAndTheirTextIsNotPersisted()
    {
        var llm = new RecordingLlmService();
        var provider = new CountingContextProvider();
        await using IConversationSession session = new ConversationBuilder(llm)
            .UseOptions(new LlmConversationOptions { SystemPrompt = "基础" })
            .AddContextProvider(provider)
            .Build();

        await ConversationEvents.CollectAsync(session.StreamAsync(Request("一")));
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("二")));

        Assert.Equal(2, provider.Calls);
        Assert.Contains("基础", TextOf(llm.Prompts[0][0]), StringComparison.Ordinal);
        Assert.Contains("上下文1", TextOf(llm.Prompts[0][0]), StringComparison.Ordinal);
        Assert.Contains("基础", TextOf(llm.Prompts[1][0]), StringComparison.Ordinal);
        Assert.Contains("上下文2", TextOf(llm.Prompts[1][0]), StringComparison.Ordinal);
        Assert.DoesNotContain("上下文1", TextOf(llm.Prompts[1][0]), StringComparison.Ordinal);
        Assert.DoesNotContain(llm.Prompts[1],
            message => TextOf(message) == "上下文1");
        Assert.Equal(2, provider.Snapshots[1].CommittedHistory.Count); // user + assistant from turn one
    }

    [Fact]
    public async Task ContextProviderFailureFailsTheTurnWithoutCommitting()
    {
        var llm = new RecordingLlmService();
        var provider = new ThrowingContextProvider();
        await using IConversationSession session = new ConversationBuilder(llm)
            .AddContextProvider(provider)
            .Build();

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (ConversationEvent _ in session.StreamAsync(Request("失败轮")))
            {
            }
        });
        Assert.Empty(llm.Prompts);

        provider.Enabled = false;
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("正常轮")));
        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.DoesNotContain(prompt, message => TextOf(message) == "失败轮");
        Assert.Contains(prompt, message => TextOf(message) == "正常轮");
    }

    [Fact]
    public async Task AReturnedStreamCanOnlyBeEnumeratedOnce()
    {
        var llm = new RecordingLlmService();
        await using IConversationSession session = new ConversationBuilder(llm).Build();
        IAsyncEnumerable<ConversationEvent> stream = session.StreamAsync(Request("一次"));

        await ConversationEvents.CollectAsync(stream);
        await Assert.ThrowsAsync<InvalidOperationException>(() => ConversationEvents.CollectAsync(stream));
    }

    [Fact]
    public async Task BuilderCanOnlyBuildOnce()
    {
        var llm = new RecordingLlmService();
        ConversationBuilder builder = new ConversationBuilder(llm);
        await using IConversationSession first = builder.Build();
        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public async Task BlankInputIsRejectedByTheDefaultSession()
    {
        var llm = new RecordingLlmService();
        await using IConversationSession session = new ConversationBuilder(llm).Build();

        await Assert.ThrowsAsync<ArgumentNullException>(() => CollectAsync(null!));
        await Assert.ThrowsAsync<ArgumentException>(() => CollectAsync(Request("   ")));

        async Task CollectAsync(ConversationRequest request)
            => await ConversationEvents.CollectAsync(session.StreamAsync(request));
    }

    [Fact]
    public async Task DisposeCancelsAnInFlightTurnAndRollsItBack()
    {
        var llm = new RecordingLlmService();
        llm.Chunks =
        [
            new LlmTextChunk { Text = "半截", Kind = LlmChunkKind.Text },
            new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true },
        ];
        llm.BlockAfterFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        IConversationSession session = new ConversationBuilder(llm).Build();

        await using IAsyncEnumerator<ConversationEvent> enumerator =
            session.StreamAsync(Request("进行中")).GetAsyncEnumerator();
        Assert.True(await enumerator.MoveNextAsync());

        Task dispose = session.DisposeAsync().AsTask();
        await Task.Delay(50);
        Assert.False(dispose.IsCompleted);
        await enumerator.DisposeAsync();
        await dispose;
        Assert.Equal(0, llm.DisposeCount);
    }

    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        for (int attempt = 0; attempt < 200 && !condition(); attempt++)
            await Task.Delay(10);
        Assert.True(condition());
    }

    private sealed class CountingContextProvider : IConversationContextProvider
    {
        public int Calls { get; private set; }

        public List<ConversationContext> Snapshots { get; } = [];

        public ValueTask<string> GetContextAsync(ConversationContext context, CancellationToken cancellationToken = default)
        {
            Calls++;
            Snapshots.Add(context);
            return ValueTask.FromResult($"上下文{Calls}");
        }
    }

    private sealed class ThrowingContextProvider : IConversationContextProvider
    {
        public bool Enabled { get; set; } = true;

        public ValueTask<string> GetContextAsync(ConversationContext context, CancellationToken cancellationToken = default)
            => Enabled
                ? throw new InvalidOperationException("memory unavailable")
                : ValueTask.FromResult(string.Empty);
    }
}

/// <summary>
/// A business session that depends only on the Abstractions assembly: no LLM, no history store.
/// </summary>
internal sealed class StaticBusinessSession(string conversationId) : IConversationSession
{
    public string ConversationId { get; } = conversationId;

    public async IAsyncEnumerable<ConversationEvent> StreamAsync(
        ConversationRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        cancellationToken.ThrowIfCancellationRequested();
        if (request.Text.Contains("订单", StringComparison.Ordinal))
        {
            yield return new ConversationTextDelta("您的订单", EndsTextSegment: true);
            yield return new ConversationTextDelta("已发货。");
            yield return new ConversationCompleted("您的订单已发货。");
            yield break;
        }

        yield return new ConversationCompleted(string.Empty);
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

public sealed class CustomBusinessSessionTests
{
    [Fact]
    public async Task AThirdPartySessionNeedsOnlyThePublicContract()
    {
        await using IConversationSession session = new StaticBusinessSession("business-1");
        var updates = await ConversationEvents.CollectAsync(
            session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "订单状态")));

        Assert.Equal("business-1", session.ConversationId);
        Assert.Equal("您的订单已发货。", Assert.IsType<ConversationCompleted>(updates[^1]).Text);
        Assert.Equal(2, updates.OfType<ConversationTextDelta>().Count());
    }
}
