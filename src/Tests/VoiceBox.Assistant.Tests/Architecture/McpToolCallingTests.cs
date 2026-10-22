using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Adapters;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI;
using VoiceBox.LLM.OpenAI.Internal;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void UseMcpTools_FailsFastForTextOnlyProviderAndBuildsForStructuredProvider()
    {
        TestServices services = TestServices.Create();
        AssistantCompositionException error = Assert.Throws<AssistantCompositionException>(() =>
            VoiceAssistant.CreateBuilder()
                .UseAudio(services.Audio)
                .UseTurnBoundary(services.Vad)
                .UseSpeechRecognition(services.Asr)
                .UseResponseGeneration(services.Llm)
                .UseMcpTools(static tools => tools.AddTool(new WeatherTool()))
                .UseSpeechSynthesis(services.Tts)
                .Build());
        Assert.Equal("VBX-COMP-011", error.Code);
        Assert.Contains("IToolCallingStreamLLMService", error.Message, StringComparison.Ordinal);

        using VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(new StructuredTestLlm())
            .UseMcpTools(static tools =>
            {
                tools.AddTool(new WeatherTool());
                tools.AddStdioServer("offline_stdio", options =>
                {
                    options.Command = "this-command-does-not-exist";
                    options.AllowedTools.Add("read_file");
                });
                tools.AddStreamableHttpServer("offline_http", options =>
                {
                    options.Endpoint = new Uri("https://127.0.0.1:1/mcp");
                    options.AllowedTools.Add("lookup");
                });
            })
            .UseSpeechSynthesis(services.Tts)
            .Build();
    }

    private sealed class StructuredTestLlm : IToolCallingStreamLLMService
    {
        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default) => Task.FromResult(new LlmCompletionResult { Text = string.Empty });

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.CompletedTask;
            yield break;
        }

        public async IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
            LlmToolTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            yield return new LlmTurnCompletedEvent();
        }

        public void Dispose()
        {
        }
    }
}

public sealed class McpToolCallingTests
{
    [Theory]
    [InlineData(AssistantToolExecutionPolicy.BackgroundOnly)]
    [InlineData(AssistantToolExecutionPolicy.ModelChoice)]
    public void LocalToolDescriptor_BackgroundCapablePolicyRequiresExecutionPrompt(
        AssistantToolExecutionPolicy policy)
    {
        foreach (string? prompt in new string?[] { null, string.Empty, "   " })
        {
            Assert.ThrowsAny<ArgumentException>(() => new AssistantToolDescriptor(
                "weather",
                "查询天气",
                "查询指定城市天气",
                "{\"type\":\"object\"}",
                executionPolicy: policy,
                executionPrompt: prompt));
        }
    }

    [Fact]
    public void LocalToolDescriptor_ForegroundPolicyAllowsMissingExecutionPrompt()
    {
        var descriptor = new AssistantToolDescriptor(
            "clock",
            "查询时间",
            "查询当前时间",
            "{\"type\":\"object\"}",
            executionPolicy: AssistantToolExecutionPolicy.ForegroundOnly);

        Assert.Null(descriptor.ExecutionPrompt);
    }

    [Fact]
    public async Task ApprovalContext_ForegroundLocalToolPreservesMissingExecutionPrompt()
    {
        var approvalPolicy = new RecordingApprovalPolicy();
        var builder = new McpToolsBuilder();
        builder.AddTool(new ForegroundTool());
        builder.UseApprovalPolicy(approvalPolicy);
        IAssistantToolRuntime runtime = builder.BuildFactory().Create(new AssistantToolRuntimePolicy(ValidateArguments: true));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            AssistantToolDefinition definition = Assert.Single((await runtime.GetSnapshotAsync(
                new ToolRequestContext(1),
                CancellationToken.None)).Definitions);
            var call = new AssistantToolCall(
                "call-1",
                definition.Id,
                OwnedJsonValue.Parse("{}"));

            Assert.Equal(
                ToolApprovalDecision.Allowed,
                await runtime.AuthorizeAsync(call, CancellationToken.None));
            Assert.NotNull(approvalPolicy.Context);
            Assert.Null(approvalPolicy.Context.Tool.ExecutionPrompt);
        }
        finally
        {
            await runtime.StopAsync();
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public void OpenAIProviderExposesNativeToolContractAndConvertsCorrelatedMessages()
    {
        Assert.True(typeof(IToolCallingStreamLLMService).IsAssignableFrom(typeof(OpenAILLMService)));
        List<OpenAI.Chat.ChatMessage> converted = OpenAIMessageConverter.Convert(
        [
            ChatMessage.AssistantToolCalls(new ToolCallContent("call-1", "local__weather", "{}")),
            ChatMessage.ToolResult("call-1", "{\"temperature\":25}"),
        ]);

        Assert.IsType<OpenAI.Chat.AssistantChatMessage>(converted[0]);
        Assert.IsType<OpenAI.Chat.ToolChatMessage>(converted[1]);
    }

    [Fact]
    public void LLamaSharpProviderExposesStructuredToolContract()
    {
        Assert.True(typeof(IToolCallingStreamLLMService).IsAssignableFrom(
            typeof(VoiceBox.LLM.LLamaSharp.LLamaSharpLLMService)));
    }

    [Fact]
    public async Task AddTool_RuntimeListsApprovesAndExecutesLocalImplementation()
    {
        var builder = new McpToolsBuilder();
        var weather = new WeatherTool();
        builder.AddTool(weather);
        IAssistantToolRuntime runtime = builder.BuildFactory().Create(new AssistantToolRuntimePolicy(ValidateArguments: true));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            ToolCatalogSnapshot catalog = await runtime.GetSnapshotAsync(
                new ToolRequestContext(1),
                CancellationToken.None);
            AssistantToolDefinition definition = Assert.Single(catalog.Definitions);
            Assert.Equal(new AssistantToolId("local", "weather"), definition.Id);
            Assert.Equal("查询天气", definition.Title);
            Assert.Equal("正在查询，请稍后", definition.ExecutionPrompt);

            var call = new AssistantToolCall(
                "call-1",
                definition.Id,
                OwnedJsonValue.Parse("{\"city\":\"杭州\"}"));
            Assert.Equal(
                ToolApprovalDecision.Allowed,
                await runtime.AuthorizeAsync(call, CancellationToken.None));
            AssistantToolResult result = await runtime.ExecuteAsync(call, CancellationToken.None);

            Assert.Equal(AssistantToolResultKind.Success, result.Kind);
            Assert.Equal(
                "杭州",
                result.Value!.ToJsonElement().GetProperty("city").GetString());
            Assert.Equal(1, weather.CallCount);
        }
        finally
        {
            await runtime.StopAsync();
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public async Task Orchestrator_ForegroundToolReturnsResultToModelBeforeVisibleAnswer()
    {
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ForegroundOnly);
        var runtime = new FakeToolRuntime(definition);
        var client = new ForegroundTurnClient(definition.Id);
        using var orchestrator = new ToolUseOrchestrator(client, runtime, ToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));

        Assert.Equal("杭州晴，25度。", Assert.Single(events.OfType<VisibleTextDelta>()).Text);
        Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(1, runtime.ExecutionCount);
        Assert.True(client.SawCorrelatedToolTranscript);
    }

    [Fact]
    public async Task Orchestrator_ModelChoiceForegroundDoesNotSubmitExecutionPrompt()
    {
        AssistantToolDefinition definition = CreateDefinition(
            AssistantToolExecutionPolicy.ModelChoice,
            executionPrompt: "正在查询，请稍后");
        var runtime = new FakeToolRuntime(definition);
        using var orchestrator = new ToolUseOrchestrator(
            new ForegroundTurnClient(definition.Id),
            runtime,
            ToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));

        VisibleTextDelta answer = Assert.Single(events.OfType<VisibleTextDelta>());
        Assert.Equal("杭州晴，25度。", answer.Text);
        Assert.False(answer.ForceSpeechBoundary);
        Assert.Equal(1, runtime.ExecutionCount);
    }

    [Fact]
    public async Task Orchestrator_BackgroundToolAcknowledgesImmediatelyThenPublishesCompletedResult()
    {
        AssistantToolDefinition definition = CreateDefinition(
            AssistantToolExecutionPolicy.ModelChoice,
            executionPrompt: "正在查询，请稍后");
        var runtime = new FakeToolRuntime(definition, blockExecution: true);
        var client = new BackgroundTurnClient(definition.Id);
        var sink = new RecordingBackgroundSink();
        using var orchestrator = new ToolUseOrchestrator(client, runtime, ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] immediate = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));
        VisibleTextDelta prompt = Assert.Single(immediate.OfType<VisibleTextDelta>());
        Assert.Equal("正在查询，请稍后", prompt.Text);
        Assert.True(prompt.ForceSpeechBoundary);
        Assert.False(sink.Published.Task.IsCompleted);

        runtime.ReleaseExecution();
        string completed = await sink.Published.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal("后台天气查询完成。", completed);
        Assert.Equal(1, runtime.ExecutionCount);
    }

    [Fact]
    public async Task Orchestrator_MixedBatchSubmitsOneBackgroundPromptBeforeAnyToolExecutes()
    {
        AssistantToolDefinition firstBackground = CreateDefinition(
            AssistantToolExecutionPolicy.BackgroundOnly,
            executionPrompt: "正在查询天气，请稍后",
            toolName: "weather");
        AssistantToolDefinition foreground = CreateDefinition(
            AssistantToolExecutionPolicy.ForegroundOnly,
            toolName: "clock");
        AssistantToolDefinition secondBackground = CreateDefinition(
            AssistantToolExecutionPolicy.BackgroundOnly,
            executionPrompt: "正在查询空气质量，请稍后",
            toolName: "air_quality");
        var runtime = new BatchToolRuntime(firstBackground, foreground, secondBackground);
        using var orchestrator = new ToolUseOrchestrator(
            new MixedPreferenceBatchTurnClient(
                firstBackground.Id,
                foreground.Id,
                secondBackground.Id),
            runtime,
            ToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        await using IAsyncEnumerator<ModelTurnEvent> events = orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None).GetAsyncEnumerator();

        Assert.True(await events.MoveNextAsync());
        VisibleTextDelta prompt = Assert.IsType<VisibleTextDelta>(events.Current);
        Assert.Equal("正在查询天气，请稍后", prompt.Text);
        Assert.True(prompt.ForceSpeechBoundary);
        Assert.Equal(0, runtime.ExecutionCount);

        Assert.True(await events.MoveNextAsync());
        VisibleTextDelta answer = Assert.IsType<VisibleTextDelta>(events.Current);
        Assert.Equal("全部查询完成。", answer.Text);
        Assert.False(answer.ForceSpeechBoundary);
        Assert.Equal(3, runtime.ExecutionCount);

        Assert.True(await events.MoveNextAsync());
        Assert.IsType<ModelTurnCompleted>(events.Current);
        Assert.False(await events.MoveNextAsync());
    }

    [Fact]
    public async Task Orchestrator_BackgroundResultsPublishInAcknowledgementOrder()
    {
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ModelChoice);
        var runtime = new OrderedFakeToolRuntime(definition);
        var sink = new OrderedBackgroundSink();
        using var orchestrator = new ToolUseOrchestrator(
            new OrderedBackgroundTurnClient(definition.Id),
            runtime,
            ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest("first"),
            CancellationToken.None));
        await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest("second"),
            CancellationToken.None));

        await runtime.SecondStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        runtime.Release("second");
        await Task.Delay(25);
        Assert.Empty(sink.Values);
        runtime.Release("first");

        string[] published = await sink.Completed.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(["first完成。", "second完成。"], published);
    }

    [Fact]
    public async Task Orchestrator_SynchronousBackgroundToolCannotBlockAcknowledgement()
    {
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ModelChoice);
        var runtime = new SynchronouslyBlockingToolRuntime(definition);
        var sink = new RecordingBackgroundSink();
        using var orchestrator = new ToolUseOrchestrator(
            new BackgroundTurnClient(definition.Id),
            runtime,
            ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        await orchestrator.StartAsync(CancellationToken.None);

        try
        {
            await using IAsyncEnumerator<ModelTurnEvent> events = orchestrator.GenerateAsync(
                CreateRequest(),
                CancellationToken.None).GetAsyncEnumerator();
            Assert.True(await events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(1)));
            Assert.Equal(
                "我正在后台执行查询天气任务，等会给你结果。",
                Assert.IsType<VisibleTextDelta>(events.Current).Text);
            Assert.False(Assert.IsType<VisibleTextDelta>(events.Current).ForceSpeechBoundary);
            Assert.False(runtime.Started.Task.IsCompleted);

            Assert.True(await events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(1)));
            Assert.IsType<ModelTurnCompleted>(events.Current);
            Assert.False(runtime.Started.Task.IsCompleted);

            Assert.False(await events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(1)));
            await runtime.Started.Task.WaitAsync(TimeSpan.FromSeconds(1));
            Assert.False(runtime.ExecutionCompleted);
        }
        finally
        {
            runtime.Release();
        }

        Assert.Equal(
            "后台天气查询完成。",
            await sink.Published.Task.WaitAsync(TimeSpan.FromSeconds(2)));
    }

    [Theory]
    [InlineData(true, "我先查一下天气。")]
    [InlineData(false, "我先查一下天气。")]
    [InlineData(true, " \r\n")]
    [InlineData(false, " \r\n")]
    public async Task Orchestrator_MixedVisibleTextAndToolCallExecutesToolAndOnlyReturnsFinalAnswer(
        bool textComesFirst,
        string preamble)
    {
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ForegroundOnly);
        var runtime = new FakeToolRuntime(definition);
        var client = new MixedOutputTurnClient(definition.Id, textComesFirst, preamble);
        using var orchestrator = new ToolUseOrchestrator(
            client,
            runtime,
            ToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));

        Assert.Equal("杭州晴，25度。", Assert.Single(events.OfType<VisibleTextDelta>()).Text);
        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(new ModelUsage(20, 10, 30), completed.Usage);
        Assert.Equal("mixed-1", Assert.Single(completed.ToolCalls).CallId);
        Assert.Equal(AssistantToolResultKind.Success, Assert.Single(completed.ToolResults).Kind);
        Assert.Equal(1, runtime.ExecutionCount);
        Assert.Equal(2, client.Transcripts.Count);
        ChatMessage[] toolExchange = client.Transcripts[1].Skip(2).ToArray();
        Assert.Equal(2, toolExchange.Length);
        Assert.Equal("mixed-1", Assert.IsType<ToolCallContent>(Assert.Single(toolExchange[0].Content)).CallId);
        Assert.Equal("mixed-1", Assert.IsType<ToolResultContent>(Assert.Single(toolExchange[1].Content)).CallId);
    }

    [Theory]
    [InlineData(true, null)]
    [InlineData(false, null)]
    [InlineData(true, "正在查询，请稍后")]
    [InlineData(false, "正在查询，请稍后")]
    public async Task Orchestrator_BackgroundMixedOutputUsesTrustedPromptAndCompletesFurtherTools(
        bool textComesFirst,
        string? executionPrompt)
    {
        AssistantToolDefinition definition = CreateDefinition(
            AssistantToolExecutionPolicy.BackgroundOnly,
            executionPrompt: executionPrompt);
        var runtime = new FakeToolRuntime(definition, blockExecution: true);
        var client = new MixedOutputTurnClient(
            definition.Id,
            textComesFirst,
            "我先查一下天气。",
            toolRounds: 2);
        var sink = new RecordingBackgroundSink();
        using var orchestrator = new ToolUseOrchestrator(client, runtime, ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] immediate = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));

        VisibleTextDelta prompt = Assert.Single(immediate.OfType<VisibleTextDelta>());
        Assert.Equal(executionPrompt ?? "我正在后台执行查询天气任务，等会给你结果。", prompt.Text);
        Assert.Equal(executionPrompt is not null, prompt.ForceSpeechBoundary);
        ModelTurnCompleted accepted = Assert.Single(immediate.OfType<ModelTurnCompleted>());
        Assert.Equal(new ModelUsage(10, 5, 15), accepted.Usage);
        Assert.Equal(AssistantToolResultKind.Accepted, Assert.Single(accepted.ToolResults).Kind);
        Assert.False(sink.Published.Task.IsCompleted);

        runtime.ReleaseExecution();
        Assert.Equal("杭州晴，25度。", await sink.Published.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(2, runtime.ExecutionCount);
        Assert.Equal(3, client.Transcripts.Count);
        ChatMessage[] toolExchange = client.Transcripts[2].Skip(2).ToArray();
        Assert.Equal(4, toolExchange.Length);
        for (int index = 0; index < 2; index++)
        {
            string callId = $"mixed-{index + 1}";
            Assert.Equal(callId, Assert.IsType<ToolCallContent>(Assert.Single(toolExchange[index * 2].Content)).CallId);
            Assert.Equal(callId, Assert.IsType<ToolResultContent>(Assert.Single(toolExchange[index * 2 + 1].Content)).CallId);
        }
    }

    [Fact]
    public async Task Orchestrator_RecordsSuccessfulToolBeforeFollowingModelFailureAndRollback()
    {
        const long turnId = 41;
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ForegroundOnly);
        var runtime = new FakeToolRuntime(definition);
        var history = new ConversationHistoryStore("system", 4);
        ChatMessage user = history.AddUserMessage(turnId, "杭州天气", out ChatMessage[] prompt);
        using var orchestrator = new ToolUseOrchestrator(
            new FailingAfterToolTurnClient(definition.Id),
            runtime,
            ToolOptions());
        orchestrator.BindToolHistorySink(history);
        await orchestrator.StartAsync(CancellationToken.None);

        await Assert.ThrowsAsync<InvalidOperationException>(() => CollectAsync(
            orchestrator.GenerateAsync(
                new ModelTurnRequest(
                    prompt,
                    new ModelGenerationOptions(),
                    ToolCatalogSnapshot.Empty,
                    turnId),
                CancellationToken.None)));
        history.RemoveUnansweredUserMessage(turnId, user);

        Assert.Equal(
            AssistantToolResultKind.Success,
            Assert.Single(history.GetToolAuditTrailSnapshot()).ResultKind);
    }

    [Fact]
    public async Task Orchestrator_RecordsSuccessfulToolBeforeFollowingModelCancellationAndRollback()
    {
        const long turnId = 42;
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ForegroundOnly);
        var runtime = new FakeToolRuntime(definition);
        var client = new CancellableAfterToolTurnClient(definition.Id);
        var history = new ConversationHistoryStore("system", 4);
        ChatMessage user = history.AddUserMessage(turnId, "杭州天气", out ChatMessage[] prompt);
        using var orchestrator = new ToolUseOrchestrator(client, runtime, ToolOptions());
        orchestrator.BindToolHistorySink(history);
        await orchestrator.StartAsync(CancellationToken.None);
        using var cancellation = new CancellationTokenSource();

        Task<ModelTurnEvent[]> generation = CollectAsync(orchestrator.GenerateAsync(
            new ModelTurnRequest(
                prompt,
                new ModelGenerationOptions(),
                ToolCatalogSnapshot.Empty,
                turnId),
            cancellation.Token));
        await client.SecondRoundStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generation);
        history.RemoveUnansweredUserMessage(turnId, user);

        Assert.Equal(
            AssistantToolResultKind.Success,
            Assert.Single(history.GetToolAuditTrailSnapshot()).ResultKind);
    }

    [Fact]
    public async Task Orchestrator_BackgroundCompletionReplacesAcceptedAuditAndEntersHistoryOnce()
    {
        const long turnId = 43;
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ModelChoice);
        var runtime = new FakeToolRuntime(definition, blockExecution: true);
        var history = new ConversationHistoryStore("system", 4);
        var toolHistory = new RecordingToolHistorySink(history);
        ChatMessage user = history.AddUserMessage(turnId, "杭州天气", out ChatMessage[] prompt);
        var sink = new RecordingBackgroundSink(
            (publishedTurnId, text) => history.RecordBackgroundAssistantText(publishedTurnId, text));
        using var orchestrator = new ToolUseOrchestrator(
            new BackgroundTurnClient(definition.Id),
            runtime,
            ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        orchestrator.BindToolHistorySink(toolHistory);
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] immediate = await CollectAsync(orchestrator.GenerateAsync(
            new ModelTurnRequest(
                prompt,
                new ModelGenerationOptions(),
                ToolCatalogSnapshot.Empty,
                turnId),
            CancellationToken.None));
        string acknowledgement = Assert.Single(immediate.OfType<VisibleTextDelta>()).Text;
        Assert.Equal(
            AssistantToolResultKind.Accepted,
            Assert.Single(history.GetToolAuditTrailSnapshot()).ResultKind);
        Assert.True(history.CommitGeneratedResponse(turnId, user, acknowledgement));

        runtime.ReleaseExecution();
        string completed = await sink.Published.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("后台天气查询完成。", completed);
        Assert.Equal(
            AssistantToolResultKind.Success,
            Assert.Single(history.GetToolAuditTrailSnapshot()).ResultKind);
        Assert.Equal(
            ["杭州天气", acknowledgement, completed],
            history.GetUserConversationHistorySnapshot().Select(MessageText));

        ChatMessage[] transcript = toolHistory.Messages;
        var audit = Assert.Single(history.GetToolAuditTrailSnapshot());
        Assert.Equal(turnId, audit.TurnId);
        Assert.Equal("call-bg", audit.CallId);
        Assert.Equal(audit.CallId, Assert.Single(transcript
            .SelectMany(static message => message.Content).OfType<ToolCallContent>()).CallId);
        Assert.Equal(audit.CallId, Assert.Single(transcript
            .SelectMany(static message => message.Content).OfType<ToolResultContent>()).CallId);
    }

    [Fact]
    public async Task Orchestrator_BackgroundSummaryFailureUsesResolvedToolOutcome()
    {
        AssistantToolDefinition definition = CreateDefinition(AssistantToolExecutionPolicy.ModelChoice);
        var sink = new RecordingBackgroundSink();
        using var orchestrator = new ToolUseOrchestrator(
            new BackgroundSummaryFailureTurnClient(definition.Id),
            new FakeToolRuntime(definition),
            ToolOptions());
        orchestrator.BindBackgroundResultSink(sink);
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        Assert.Equal(
            "查询天气任务已完成。",
            await sink.Published.Task.WaitAsync(TimeSpan.FromSeconds(2)));
    }

    [Fact]
    public async Task Orchestrator_UnknownSideEffectOutcomeBlocksSameInvocationReplay()
    {
        AssistantToolDefinition definition = CreateDefinition(
            AssistantToolExecutionPolicy.ForegroundOnly,
            Modules.ModelInteraction.Models.AssistantToolRisk.SideEffecting,
            TimeSpan.FromMilliseconds(50));
        var runtime = new TimingOutToolRuntime(definition);
        using var orchestrator = new ToolUseOrchestrator(
            new UnknownOutcomeReplayTurnClient(definition.Id),
            runtime,
            ToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(orchestrator.GenerateAsync(
            CreateRequest(),
            CancellationToken.None));

        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(1, runtime.ExecutionCount);
        Assert.Equal(2, completed.ToolResults.Count);
        Assert.All(
            completed.ToolResults,
            result => Assert.Equal(AssistantToolResultKind.UnknownOutcome, result.Kind));
        Assert.Equal(
            "retry_blocked_after_unknown_outcome",
            completed.ToolResults[1].ErrorCode);
    }

    /// <summary>
    /// Production defaults. Tests that assert an execution ordering property pass an explicit
    /// options object instead, so the default stays the thing under test everywhere else.
    /// </summary>
    private static VoiceAssistantToolOptions ToolOptions() => new();

    private static ModelTurnRequest CreateRequest() => new(
        [ChatMessage.System("system"), ChatMessage.User("杭州天气")],
        new ModelGenerationOptions(),
        ToolCatalogSnapshot.Empty);

    private static ModelTurnRequest CreateRequest(string text) => new(
        [ChatMessage.System("system"), ChatMessage.User(text)],
        new ModelGenerationOptions(),
        ToolCatalogSnapshot.Empty);

    private static AssistantToolDefinition CreateDefinition(
        AssistantToolExecutionPolicy policy,
        Modules.ModelInteraction.Models.AssistantToolRisk risk =
            Modules.ModelInteraction.Models.AssistantToolRisk.ReadOnly,
        TimeSpan? timeout = null,
        string? executionPrompt = null,
        string toolName = "weather") => new(
        new AssistantToolId("local", toolName),
        "查询天气",
        "查询指定城市天气",
        OwnedJsonValue.Parse("{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}"),
        null,
        risk,
        policy,
        executionPrompt,
        timeout ?? TimeSpan.FromSeconds(2));

    private static async Task<T[]> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var result = new List<T>();
        await foreach (T item in source)
            result.Add(item);
        return result.ToArray();
    }

    private sealed class FakeToolRuntime(
        AssistantToolDefinition definition,
        bool blockExecution = false) : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog = new([definition]);
        private readonly TaskCompletionSource _release = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly bool _blockExecution = blockExecution;
        public int ExecutionCount => Volatile.Read(ref _executionCount);
        private int _executionCount;

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public async ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _executionCount);
            if (_blockExecution)
                await _release.Task.WaitAsync(cancellationToken);
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"city\":\"杭州\",\"temperature\":25}"));
        }

        public void ReleaseExecution() => _release.TrySetResult();
    }

    private sealed class BatchToolRuntime(params AssistantToolDefinition[] definitions)
        : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog = new(definitions);

        public int ExecutionCount => Volatile.Read(ref _executionCount);
        private int _executionCount;

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;

        public Task StopAsync() => Task.CompletedTask;

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;

        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);

        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _executionCount);
            return ValueTask.FromResult(new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}")));
        }
    }

    private sealed class ForegroundTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;
        public bool SawCorrelatedToolTranscript { get; private set; }

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (_round++ == 0)
            {
                yield return new ToolCallRequested(
                    "call-1",
                    toolId,
                    OwnedJsonValue.Parse("{\"city\":\"杭州\"}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            SawCorrelatedToolTranscript = request.Transcript.Any(message =>
                message.Content.OfType<ToolCallContent>().Any(call => call.CallId == "call-1")) &&
                request.Transcript.Any(message =>
                    message.Content.OfType<ToolResultContent>().Any(result => result.CallId == "call-1"));
            yield return new VisibleTextDelta("杭州晴，25度。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class MixedPreferenceBatchTurnClient(
        AssistantToolId firstBackground,
        AssistantToolId foreground,
        AssistantToolId secondBackground) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (_round++ == 0)
            {
                yield return new ToolCallRequested(
                    "background-1",
                    firstBackground,
                    OwnedJsonValue.Parse("{}"));
                yield return new ToolCallRequested(
                    "foreground",
                    foreground,
                    OwnedJsonValue.Parse("{}"));
                yield return new ToolCallRequested(
                    "background-2",
                    secondBackground,
                    OwnedJsonValue.Parse("{}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            yield return new VisibleTextDelta("全部查询完成。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class BackgroundTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (_round++ == 0)
            {
                yield return new ToolCallRequested(
                    "call-bg",
                    toolId,
                    OwnedJsonValue.Parse(
                        "{\"city\":\"杭州\",\"__voicebox_execution\":\"background\"}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            yield return new VisibleTextDelta("后台天气查询完成。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class RecordingToolHistorySink(ConversationHistoryStore history) : IToolHistorySink
    {
        private readonly List<ChatMessage> _messages = [];

        public ChatMessage[] Messages
        {
            get
            {
                lock (_messages)
                    return _messages.ToArray();
            }
        }

        public void RecordToolActivity(long turnId, AssistantToolCall call, AssistantToolResult result)
            => history.RecordToolActivity(turnId, call, result);

        public void RecordToolExchange(long turnId, IReadOnlyList<ChatMessage> toolMessages)
        {
            lock (_messages)
                _messages.AddRange(toolMessages);
            history.RecordToolExchange(turnId, toolMessages);
        }
    }

    private sealed class RecordingBackgroundSink(
        Action<long, string>? onPublished = null) : IBackgroundToolResultSink
    {
        public TaskCompletionSource<string> Published { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public Task PublishAsync(long turnId, string text, CancellationToken cancellationToken)
        {
            onPublished?.Invoke(turnId, text);
            Published.TrySetResult(text);
            return Task.CompletedTask;
        }
    }

    private sealed class SynchronouslyBlockingToolRuntime(AssistantToolDefinition definition)
        : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog = new([definition]);
        private readonly ManualResetEventSlim _release = new(false);
        public TaskCompletionSource Started { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);
        public bool ExecutionCompleted { get; private set; }

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync()
        {
            _release.Dispose();
            return ValueTask.CompletedTask;
        }
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Started.TrySetResult();
            _release.Wait(cancellationToken);
            ExecutionCompleted = true;
            return ValueTask.FromResult(new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}")));
        }

        public void Release() => _release.Set();
    }

    private sealed class MixedOutputTurnClient(
        AssistantToolId toolId,
        bool textComesFirst,
        string preamble,
        int toolRounds = 1) : ILlmTurnClient
    {
        public List<ChatMessage[]> Transcripts { get; } = [];

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            Transcripts.Add(request.Transcript.ToArray());
            if (Transcripts.Count <= toolRounds)
            {
                if (textComesFirst)
                    yield return new VisibleTextDelta(preamble);
                yield return new ToolCallRequested(
                    $"mixed-{Transcripts.Count}", toolId, OwnedJsonValue.Parse("{}"));
                if (!textComesFirst)
                    yield return new VisibleTextDelta(preamble);
            }
            else
            {
                yield return new VisibleTextDelta("杭州晴，25度。");
            }
            yield return new ModelTurnCompleted(new ModelUsage(10, 5, 15));
        }
    }

    private sealed class FailingAfterToolTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (Interlocked.Increment(ref _round) == 1)
            {
                yield return new ToolCallRequested("failure", toolId, OwnedJsonValue.Parse("{}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }
            throw new InvalidOperationException("model failed after tool execution");
        }
    }

    private sealed class CancellableAfterToolTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;
        public TaskCompletionSource SecondRoundStarted { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (Interlocked.Increment(ref _round) == 1)
            {
                yield return new ToolCallRequested("cancel", toolId, OwnedJsonValue.Parse("{}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }
            SecondRoundStarted.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
        }
    }

    private sealed class BackgroundSummaryFailureTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (Interlocked.Increment(ref _round) == 1)
            {
                yield return new ToolCallRequested(
                    "summary-failure",
                    toolId,
                    OwnedJsonValue.Parse("{\"__voicebox_execution\":\"background\"}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }
            throw new InvalidOperationException("summary failed");
        }
    }

    private sealed class TimingOutToolRuntime(AssistantToolDefinition definition)
        : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog = new([definition]);
        public int ExecutionCount => Volatile.Read(ref _executionCount);
        private int _executionCount;
        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);
        public async ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _executionCount);
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            return new AssistantToolResult(call.CallId, AssistantToolResultKind.Success, null);
        }
    }

    private sealed class UnknownOutcomeReplayTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;
        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            int round = Interlocked.Increment(ref _round);
            if (round <= 2)
            {
                yield return new ToolCallRequested(
                    $"unknown-{round}",
                    toolId,
                    OwnedJsonValue.Parse("{\"amount\":99}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }
            yield return new VisibleTextDelta("已阻止重复执行。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private static string MessageText(ChatMessage message) =>
        string.Concat(message.Content.OfType<TextContent>().Select(static value => value.Text));

    private sealed class OrderedFakeToolRuntime(AssistantToolDefinition definition) : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog = new([definition]);
        private readonly Dictionary<string, TaskCompletionSource> _releases = new(StringComparer.Ordinal);
        private readonly object _gate = new();
        public TaskCompletionSource SecondStarted { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public async ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            TaskCompletionSource release;
            lock (_gate)
            {
                release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                _releases.Add(call.CallId, release);
                if (string.Equals(call.CallId, "second", StringComparison.Ordinal))
                    SecondStarted.TrySetResult();
            }
            await release.Task.WaitAsync(cancellationToken);
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}"));
        }

        public void Release(string callId)
        {
            TaskCompletionSource release;
            lock (_gate)
                release = _releases[callId];
            release.TrySetResult();
        }
    }

    private sealed class OrderedBackgroundTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            ToolResultContent? result = request.Transcript
                .SelectMany(static message => message.Content)
                .OfType<ToolResultContent>()
                .LastOrDefault();
            if (result is null)
            {
                string callId = request.Transcript
                    .SelectMany(static message => message.Content)
                    .OfType<TextContent>()
                    .Last().Text;
                yield return new ToolCallRequested(
                    callId,
                    toolId,
                    OwnedJsonValue.Parse(
                        $"{{\"value\":\"{callId}\",\"__voicebox_execution\":\"background\"}}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            yield return new VisibleTextDelta($"{result.CallId}完成。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class OrderedBackgroundSink : IBackgroundToolResultSink
    {
        private readonly List<string> _values = [];
        public IReadOnlyList<string> Values
        {
            get
            {
                lock (_values)
                    return _values.ToArray();
            }
        }
        public TaskCompletionSource<string[]> Completed { get; } = new(
            TaskCreationOptions.RunContinuationsAsynchronously);

        public Task PublishAsync(long turnId, string text, CancellationToken cancellationToken)
        {
            lock (_values)
            {
                _values.Add(text);
                if (_values.Count == 2)
                    Completed.TrySetResult(_values.ToArray());
            }
            return Task.CompletedTask;
        }
    }

    private sealed class RecordingApprovalPolicy : IAssistantToolApprovalPolicy
    {
        public AssistantToolApprovalContext? Context { get; private set; }

        public ValueTask<AssistantToolApproval> AuthorizeAsync(
            AssistantToolApprovalContext context,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Context = context;
            return ValueTask.FromResult(AssistantToolApproval.Allow);
        }
    }
}

internal sealed class WeatherTool : IAssistantTool
{
    private int _callCount;
    public int CallCount => Volatile.Read(ref _callCount);

    public AssistantToolDescriptor Descriptor { get; } = new(
        "weather",
        "查询天气",
        "查询指定城市天气",
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}",
        risk: Api.Tools.Models.AssistantToolRisk.ReadOnly,
        executionPolicy: AssistantToolExecutionPolicy.ModelChoice,
        executionPrompt: "正在查询，请稍后");

    public ValueTask<AssistantToolExecutionResult> ExecuteAsync(
        AssistantToolInvocation invocation,
        CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _callCount);
        string city = invocation.Arguments.GetProperty("city").GetString()!;
        return ValueTask.FromResult(AssistantToolExecutionResult.Success(
            $"{city}晴，25度。",
            $"{{\"city\":\"{city}\",\"temperature\":25}}"));
    }

    public void Dispose() { }
}

internal sealed class ForegroundTool : IAssistantTool
{
    public AssistantToolDescriptor Descriptor { get; } = new(
        "clock",
        "查询时间",
        "查询当前时间",
        "{\"type\":\"object\"}",
        executionPolicy: AssistantToolExecutionPolicy.ForegroundOnly);

    public ValueTask<AssistantToolExecutionResult> ExecuteAsync(
        AssistantToolInvocation invocation,
        CancellationToken cancellationToken)
        => ValueTask.FromResult(AssistantToolExecutionResult.Success("已完成。", "{}"));

    public void Dispose() { }
}
