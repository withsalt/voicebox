using System.Runtime.CompilerServices;
using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Tests.Support;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ToolConversationSessionTests
{
    private static ConversationRequest Request(string text)
        => new(Guid.NewGuid(), text);

    private static async IAsyncEnumerable<LlmTurnEvent> Events(params LlmTurnEvent[] events)
    {
        await Task.Yield();
        foreach (LlmTurnEvent item in events)
            yield return item;
    }

    private static LlmTurnEvent Call(string callId, LlmToolTurnRequest request, string arguments = "{}")
        => new LlmToolCallEvent(callId, Assert.Single(request.Tools).Name, arguments);

    [Fact]
    public async Task LocalToolRunsInTheBoundedLoopAndItsResultReachesTheNextRound()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("最终回答"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)).Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            List<ConversationEvent> updates = await ConversationEvents.CollectAsync(session.StreamAsync(Request("查一下")));

            Assert.Equal("最终回答", Assert.IsType<ConversationCompleted>(updates[^1]).Text);
            Assert.Equal(1, tool.ExecuteCount);
            Assert.Equal(2, llm.Prompts.Count);
            ChatMessage[] secondRound = llm.Prompts[1];
            Assert.Contains(secondRound, message => message.Role == ChatRole.Tool);
            Assert.Contains(secondRound,
                message => message.Role == ChatRole.Assistant && message.Content.Any(content => content is ToolCallContent));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task ToolThinkingIsObservableWithoutEnteringAnswersOrHistory()
    {
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(new LlmTurnTextDelta("选择工具", LlmChunkKind.Thinking), Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("整理结果", LlmChunkKind.Thinking), new LlmTurnTextDelta("最终回答"), new LlmTurnCompletedEvent()),
            3 => Events(new LlmTurnTextDelta("继续回答"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };
        await using IConversationSession session = new ConversationBuilder(llm)
            .UseOptions(new LlmConversationOptions { EnableThinking = true })
            .UseTools(tools => tools.AddTool(new EchoTool()))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();

        List<ConversationEvent> updates = await ConversationEvents.CollectAsync(session.StreamAsync(Request("查一下")));

        Assert.Equal(["选择工具", "整理结果"], updates.OfType<ConversationThinkingDelta>().Select(update => update.Text));
        Assert.Equal("最终回答", Assert.Single(updates.OfType<ConversationTextDelta>()).Text);
        Assert.Equal("最终回答", Assert.IsType<ConversationCompleted>(updates[^1]).Text);
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("继续")));
        Assert.All(llm.Prompts, prompt => Assert.DoesNotContain(prompt, message => TextOf(message)is "选择工具" or "整理结果"));
        Assert.Contains(llm.Prompts[^1], message => TextOf(message) == "最终回答");
    }

    [Fact]
    public async Task ToolEnabledConversationRequiresAStructuredProvider()
    {
        var llm = new TextOnlyLlmService();
        Assert.Throws<InvalidOperationException>(() =>
            new ConversationBuilder(llm)
                .UseTools(tools => tools.AddTool(new EchoTool()))
                .Build());
        await Task.CompletedTask;
    }

    [Fact]
    public async Task ApprovalPolicyCanDenyAToolBeforeItRuns()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(ConversationToolRisk.SideEffecting);
        var policy = new FixedApprovalPolicy(ConversationToolApproval.Deny);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("被拒绝"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools.UseApprovalPolicy(policy).AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("试一下")));
            Assert.Equal(0, tool.ExecuteCount);
            ChatMessage[] secondRound = llm.Prompts[1];
            Assert.Contains(secondRound,
                message => message.Content.OfType<ToolResultContent>()
                    .Any(result => result.ResultJson.Contains("Denied", StringComparison.Ordinal)));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task InvalidArgumentsNeverReachTheTool()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(requireCity: true);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request, "{}"), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("需要城市"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)).Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("查一下")));
            Assert.Equal(0, tool.ExecuteCount);
            Assert.Contains(llm.Prompts[1],
                message => message.Content.OfType<ToolResultContent>()
                    .Any(result => result.ResultJson.Contains("invalid_arguments", StringComparison.Ordinal)));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Theory]
    [InlineData("[]", true)]
    [InlineData("null", true)]
    [InlineData("42", true)]
    [InlineData("\"city\"", true)]
    [InlineData("{}", true)]
    [InlineData("[]", false)]
    [InlineData("null", false)]
    [InlineData("42", false)]
    [InlineData("\"city\"", false)]
    public async Task InvalidArgumentsAreCorrectedBeforeApprovalAndExecution(string arguments, bool validateSchema)
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(requireCity: true);
        var policy = new FixedApprovalPolicy(ConversationToolApproval.Allow);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("invalid", request, arguments), new LlmTurnCompletedEvent()),
            2 => Events(Call("corrected", request, """{"city":"北京"}"""), new LlmTurnCompletedEvent()),
            3 => Events(new LlmTurnTextDelta("查询完成"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools
                .AddTool(tool)
                .UseApprovalPolicy(policy)
                .UseOptions(new ConversationToolOptions { ValidateToolArguments = validateSchema }))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();

        List<ConversationEvent> updates = await ConversationEvents.CollectAsync(session.StreamAsync(Request("查一下北京")));

        Assert.Equal("查询完成", Assert.IsType<ConversationCompleted>(updates[^1]).Text);
        ToolResultContent invalid = Assert.Single(llm.Prompts[1].SelectMany(message => message.Content).OfType<ToolResultContent>());
        Assert.Equal("invalid", invalid.CallId);
        Assert.Contains("invalid_arguments", invalid.ResultJson, StringComparison.Ordinal);
        Assert.Equal("corrected", Assert.Single(policy.Calls).CallId);
        Assert.Equal(1, tool.ExecuteCount);
        Assert.Contains(llm.Prompts[2].SelectMany(message => message.Content).OfType<ToolResultContent>(),
            result => result.CallId == "corrected");
    }

    [Fact]
    public async Task BackgroundToolAcknowledgesThenPublishesOnTheBackgroundChannel()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(policy: ConversationToolExecutionPolicy.BackgroundOnly, executionPrompt: "正在查询");
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        tool.OnExecute = async cancellationToken =>
        {
            await release.Task.WaitAsync(cancellationToken);
            return ConversationToolResult.Success("晴");
        };
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("今天晴"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        IConversationSession session = new ConversationBuilder(llm)
            .UseConversationId("background-session")
            .UseTools(tools => tools.AddTool(tool))
            .Build();
        var background = (IConversationBackgroundOutput)session;
        await background.StartAsync();
        try
        {
            var request = Request("天气");
            await using IAsyncEnumerator<ConversationEvent> updates = session.StreamAsync(request).GetAsyncEnumerator();
            Assert.True(await updates.MoveNextAsync());
            var prompt = Assert.IsType<ConversationTextDelta>(updates.Current);
            Assert.Equal("正在查询", prompt.Text);
            Assert.True(prompt.EndsTextSegment);
            // The tool must not run until the prompt segment has been consumed and the stream
            // advances past it.
            Assert.Equal(0, tool.ExecuteCount);

            Assert.True(await updates.MoveNextAsync());
            Assert.Equal("正在查询", Assert.IsType<ConversationCompleted>(updates.Current).Text);

            release.TrySetResult();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            await using IAsyncEnumerator<ConversationBackgroundMessage> reader =
                background.ReadBackgroundMessagesAsync(timeout.Token).GetAsyncEnumerator();
            Assert.True(await reader.MoveNextAsync());
            ConversationBackgroundMessage message = reader.Current;
            Assert.Equal("background-session", message.ConversationId);
            Assert.Equal(request.RequestId, message.OriginatingRequestId);
            Assert.Equal("今天晴", message.Text);
        }
        finally
        {
            release.TrySetResult();
            await session.DisposeAsync();
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public async Task BackgroundResultsAreRecordedInHistoryBeforePublication(int retentionTurns)
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(policy: ConversationToolExecutionPolicy.BackgroundOnly, executionPrompt: "正在查询");
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("后台结果"), new LlmTurnCompletedEvent()),
            3 => Events(new LlmTurnTextDelta("继续回答"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseOptions(new ConversationToolOptions
            {
                ToolTranscriptRetentionTurns = retentionTurns,
            }))
            .Build();
        var background = (IConversationBackgroundOutput)session;
        await background.StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            await using (IAsyncEnumerator<ConversationBackgroundMessage> reader =
                background.ReadBackgroundMessagesAsync(timeout.Token).GetAsyncEnumerator())
            {
                Assert.True(await reader.MoveNextAsync());
                Assert.Equal("后台结果", reader.Current.Text);
            }

            await ConversationEvents.CollectAsync(session.StreamAsync(Request("继续")));
            string history = string.Join("\n", llm.Prompts[^1].Select(TextOf));
            Assert.Contains("后台结果", history, StringComparison.Ordinal);
            ChatContent[] contents = llm.Prompts[^1].SelectMany(message => message.Content).ToArray();
            if (retentionTurns == 0)
            {
                Assert.Empty(contents.OfType<ToolCallContent>());
                Assert.Empty(contents.OfType<ToolResultContent>());
            }
            else
            {
                Assert.Equal("c1", Assert.Single(contents.OfType<ToolCallContent>()).CallId);
                ToolResultContent result = Assert.Single(contents.OfType<ToolResultContent>());
                Assert.Equal("c1", result.CallId);
                using JsonDocument payload = JsonDocument.Parse(result.ResultJson);
                Assert.Equal("晴", payload.RootElement.GetProperty("content").GetString());
                Assert.Equal(
                    [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Assistant,
                        ChatRole.Tool, ChatRole.Assistant, ChatRole.User],
                    llm.Prompts[^1].Select(message => message.Role));
            }
        }
        finally
        {
            await session.DisposeAsync();
        }
    }

    [Fact]
    public async Task CanceledTurnIsRemovedWhenAnEarlierBackgroundResultArrives()
    {
        var releaseTool = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var pendingStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var pendingCancellation = new CancellationTokenSource();
        var tool = new EchoTool(policy: ConversationToolExecutionPolicy.BackgroundOnly)
        {
            OnExecute = async token =>
            {
                await releaseTool.Task.WaitAsync(token);
                return ConversationToolResult.Success("晴");
            },
        };
        async IAsyncEnumerable<LlmTurnEvent> Pending()
        {
            pendingStarted.TrySetResult();
            await Task.Delay(Timeout.Infinite, pendingCancellation.Token);
            yield return new LlmTurnCompletedEvent();
        }
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Pending(),
            3 => Events(new LlmTurnTextDelta("后台结果"), new LlmTurnCompletedEvent()),
            4 => Events(new LlmTurnTextDelta("继续回答"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };
        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)).Build();
        var background = (IConversationBackgroundOutput)session;
        await background.StartAsync();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));
        Task<List<ConversationEvent>> pending = ConversationEvents.CollectAsync(
            session.StreamAsync(Request("取消的问题"), pendingCancellation.Token));
        try
        {
            await pendingStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
            releaseTool.TrySetResult();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await using (IAsyncEnumerator<ConversationBackgroundMessage> reader =
                background.ReadBackgroundMessagesAsync(timeout.Token).GetAsyncEnumerator())
            {
                Assert.True(await reader.MoveNextAsync());
                Assert.Equal("后台结果", reader.Current.Text);
            }
            await pendingCancellation.CancelAsync();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("继续")));

            Assert.DoesNotContain(llm.Prompts[^1], message => TextOf(message) == "取消的问题");
            Assert.Contains(llm.Prompts[^1], message => TextOf(message) == "后台结果");
            Assert.Contains(llm.Prompts[^1], message => TextOf(message) == "天气");
        }
        finally
        {
            releaseTool.TrySetResult();
            await pendingCancellation.CancelAsync();
            try { await pending; } catch (OperationCanceledException) { }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task BackgroundFailureRetainsOnlyCompleteToolExchanges(int failingApproval)
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(policy: ConversationToolExecutionPolicy.BackgroundOnly);
        llm.ToolTurnScript = (_, request) => TextOf(request.Messages[^1]) switch
        {
            "天气" => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            "继续" => Events(new LlmTurnTextDelta("继续回答"), new LlmTurnCompletedEvent()),
            _ => Events(Call("c2", request), new LlmTurnCompletedEvent()),
        };
        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools.AddTool(tool).UseApprovalPolicy(new FailingApprovalPolicy(failingApproval)))
            .Build();
        var background = (IConversationBackgroundOutput)session;
        await background.StartAsync();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("天气")));
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await using (IAsyncEnumerator<ConversationBackgroundMessage> reader =
            background.ReadBackgroundMessagesAsync(timeout.Token).GetAsyncEnumerator())
        {
            Assert.True(await reader.MoveNextAsync());
            Assert.Contains("任务执行失败", reader.Current.Text, StringComparison.Ordinal);
        }
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("继续")));

        ChatContent[] contents = llm.Prompts[^1].SelectMany(message => message.Content).ToArray();
        string[] expectedCalls = failingApproval == 1 ? [] : ["c1"];
        Assert.Equal(expectedCalls, contents.OfType<ToolCallContent>().Select(call => call.CallId));
        Assert.Equal(expectedCalls, contents.OfType<ToolResultContent>().Select(result => result.CallId));
    }

    [Fact]
    public async Task TextBeforeAToolCallIsStreamedAndThinkingStaysOutOfTheAnswer()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(
                new LlmTurnTextDelta("思考中", LlmChunkKind.Thinking),
                new LlmTurnTextDelta("工具前附带文本"),
                Call("c1", request),
                new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("最终回答"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(tool)).Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            List<ConversationEvent> events = await ConversationEvents.CollectAsync(session.StreamAsync(Request("思考")));

            Assert.Contains(events, item => item is ConversationThinkingDelta { Text: "思考中" });
            // Text is forwarded as it arrives, so what the model said ahead of its tool call is part of the answer.
            Assert.Equal(["工具前附带文本", "\n最终回答"], events.OfType<ConversationTextDelta>().Select(item => item.Text));
            Assert.Equal("工具前附带文本\n最终回答", Assert.IsType<ConversationCompleted>(events[^1]).Text);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task ApprovalFailureLeavesNoDanglingToolCallInTheNextPrompt()
    {
        var llm = new RecordingLlmService();
        var tool = new EchoTool(ConversationToolRisk.SideEffecting);
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("恢复"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };

        await using IConversationSession session = new ConversationBuilder(llm)
            .UseTools(tools => tools.UseApprovalPolicy(new ThrowingApprovalPolicy()).AddTool(tool))
            .Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await ConversationEvents.CollectAsync(session.StreamAsync(Request("失败轮")));
            });

            await ConversationEvents.CollectAsync(session.StreamAsync(Request("下一轮")));
            Assert.DoesNotContain(llm.Prompts[^1],
                message => message.Content.Any(content => content is ToolCallContent or ToolResultContent));
            Assert.Contains(llm.Prompts[^1],
                message => message.Content.OfType<TextContent>()
                    .Any(content => content.Text == "下一轮"));
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    [Fact]
    public async Task ToolSessionMustBeStartedBeforeStreaming()
    {
        var llm = new RecordingLlmService();
        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new EchoTool())).Build();

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("未启动")));
        });
    }

    [Fact]
    public async Task StopAfterDisposeAndRestartRetainsHistory()
    {
        var llm = new RecordingLlmService { ResponseText = "回答" };
        IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new EchoTool())).Build();
        var lifecycle = (IConversationLifecycle)session;

        await lifecycle.StartAsync();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("第一问")));
        await lifecycle.StopAsync();

        await lifecycle.StartAsync();
        await ConversationEvents.CollectAsync(session.StreamAsync(Request("第二问")));
        await lifecycle.StopAsync();

        ChatMessage[] prompt = llm.Prompts[^1];
        Assert.Contains(prompt, message => TextOf(message) == "第一问");
        Assert.Contains(prompt, message => TextOf(message) == "回答");
        await session.DisposeAsync();
    }

    [Fact]
    public async Task StopCancelsABlockedBackgroundChannelWrite()
    {
        var llm = new RecordingLlmService();
        IConversationSession session = new ConversationBuilder(llm)
            .UseConversationId("backpressure")
            .UseTools(tools => tools.AddTool(new EchoTool()))
            .Build();
        var background = (IConversationBackgroundOutput)session;
        var publisher = (IBackgroundResultPublisher)session;
        await background.StartAsync();

        // Fill the public capacity so the next write blocks on the bounded channel.
        for (int index = 0; index < 64; index++)
        {
            await publisher.PublishAsync(
                new ConversationBackgroundMessage(Guid.NewGuid(), "backpressure", Guid.NewGuid(), $"m{index}"),
                CancellationToken.None);
        }

        Task blocked = publisher.PublishAsync(
            new ConversationBackgroundMessage(Guid.NewGuid(), "backpressure", Guid.NewGuid(), "blocked"),
            CancellationToken.None).AsTask();
        await Task.Delay(50);
        Assert.False(blocked.IsCompleted);

        Task stop = background.StopAsync();
        await stop.WaitAsync(TimeSpan.FromSeconds(10));
        await blocked.WaitAsync(TimeSpan.FromSeconds(10));
        await session.DisposeAsync();
    }

    [Fact]
    public async Task RestartDoesNotDeliverPreviousRunMessages()
    {
        var llm = new RecordingLlmService();
        IConversationSession session = new ConversationBuilder(llm)
            .UseConversationId("restart")
            .UseTools(tools => tools.AddTool(new EchoTool()))
            .Build();
        var background = (IConversationBackgroundOutput)session;
        var publisher = (IBackgroundResultPublisher)session;

        await background.StartAsync();
        await publisher.PublishAsync(
            new ConversationBackgroundMessage(Guid.NewGuid(), "restart", Guid.NewGuid(), "旧运行消息"),
            CancellationToken.None);
        await background.StopAsync();

        await background.StartAsync();
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            {
                await foreach (ConversationBackgroundMessage _ in
                    background.ReadBackgroundMessagesAsync(timeout.Token))
                {
                }
            });
        }
        finally
        {
            await session.DisposeAsync();
        }
    }

    [Fact]
    public async Task LocalToolDescriptorsExposeStableIdentityAndSchemas()
    {
        var llm = new RecordingLlmService();
        llm.ToolTurnScript = (call, request) => call switch
        {
            1 => Events(Call("c1", request), new LlmTurnCompletedEvent()),
            2 => Events(new LlmTurnTextDelta("ok"), new LlmTurnCompletedEvent()),
            _ => throw new InvalidOperationException("unexpected round"),
        };
        await using IConversationSession session = new ConversationBuilder(llm).UseTools(tools => tools.AddTool(new EchoTool())).Build();
        await ((IConversationLifecycle)session).StartAsync();
        try
        {
            await ConversationEvents.CollectAsync(session.StreamAsync(Request("查")));
            LlmToolDefinition definition = Assert.Single(llm.ToolRequests[0].Tools);
            Assert.Equal("local__echo", definition.Name);
            Assert.Contains("__execution", definition.InputSchemaJson, StringComparison.Ordinal);
        }
        finally
        {
            await ((IConversationLifecycle)session).StopAsync();
        }
    }

    private static string TextOf(ChatMessage message)
        => string.Concat(message.Content.OfType<TextContent>().Select(content => content.Text));

    private sealed class EchoTool : IConversationTool
    {
        public EchoTool(
            ConversationToolRisk risk = ConversationToolRisk.ReadOnly,
            ConversationToolExecutionPolicy policy = ConversationToolExecutionPolicy.ForegroundOnly,
            string? executionPrompt = null,
            bool requireCity = false)
        {
            Descriptor = new ConversationToolDescriptor(
                "echo",
                "回声工具",
                "返回输入内容",
                requireCity
                    ? """
                      {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"],
                        "additionalProperties": false
                      }
                      """
                    : """
                      {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                      }
                      """,
                risk: risk,
                executionPolicy: policy,
                executionPrompt: executionPrompt ?? (policy == ConversationToolExecutionPolicy.ForegroundOnly ? null : "正在处理"),
                timeout: TimeSpan.FromSeconds(5));
        }

        public ConversationToolDescriptor Descriptor { get; }

        public int ExecuteCount { get; private set; }

        public Func<CancellationToken, ValueTask<ConversationToolResult>>? OnExecute { get; set; }

        public async ValueTask<ConversationToolResult> ExecuteAsync(ConversationToolCall call, CancellationToken cancellationToken)
        {
            ExecuteCount++;
            if (OnExecute is not null)
                return await OnExecute(cancellationToken);
            return ConversationToolResult.Success("晴");
        }

        public void Dispose()
        {
        }
    }

    private sealed class FailingApprovalPolicy(int failingCall) : IConversationToolApprovalPolicy
    {
        private int _calls;

        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
        {
            if (++_calls == failingCall)
                throw new InvalidOperationException("Approval service failed.");
            return ValueTask.FromResult(ConversationToolApproval.Allow);
        }
    }

    private sealed class ThrowingApprovalPolicy : IConversationToolApprovalPolicy
    {
        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
            => throw new InvalidOperationException("approval failed");
    }

    private sealed class FixedApprovalPolicy(ConversationToolApproval decision)
        : IConversationToolApprovalPolicy
    {
        public List<ConversationToolCall> Calls { get; } = [];

        public ValueTask<ConversationToolApproval> AuthorizeAsync(
            ConversationToolApprovalContext context,
            CancellationToken cancellationToken)
        {
            Calls.Add(context.Call);
            return ValueTask.FromResult(decision);
        }
    }

    private sealed class TextOnlyLlmService : IStreamLLMService
    {
        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => Task.FromResult(new LlmCompletionResult { Text = "ok" });

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            yield return new LlmTextChunk { Text = "ok", Kind = LlmChunkKind.Text };
        }

        public void Dispose()
        {
        }
    }
}
