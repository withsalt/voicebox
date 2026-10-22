using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Assistant.Modules.Conversation;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Adapters;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;
using InternalToolRisk = VoiceBox.Assistant.Modules.ModelInteraction.Models.AssistantToolRisk;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Batch execution, transient-failure recovery and argument validation. Together these decide how
/// long a spoken turn waits on tools and what the model learns when a call cannot be satisfied.
/// </summary>
public sealed class ToolExecutionPolicyTests
{
    private static readonly TimeSpan ToolDelay = TimeSpan.FromMilliseconds(200);

    [Theory]
    [InlineData((int)ToolApprovalDecision.Allowed, 1, (int)AssistantToolResultKind.Success)]
    [InlineData((int)ToolApprovalDecision.Denied, 0, (int)AssistantToolResultKind.Denied)]
    [InlineData((int)ToolApprovalDecision.RequiresConfirmation, 0, (int)AssistantToolResultKind.Denied)]
    public async Task ToolRuntime_ExecutesOnlyAuthorizedCalls(
        int approval,
        int expectedExecutions,
        int expectedResult)
    {
        var runtime = new ConcurrencyProbeToolRuntime(InternalToolRisk.ReadOnly, TimeSpan.Zero, toolCount: 1)
        {
            ApprovalDecision = (ToolApprovalDecision)approval,
        };
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolIds[0]),
            runtime,
            new VoiceAssistantToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        AssistantToolResult result = Assert.Single(completed.ToolResults);
        Assert.Equal("call-1", result.CallId);
        Assert.Equal((AssistantToolResultKind)expectedResult, result.Kind);
        Assert.Equal(1, runtime.AuthorizationCount);
        Assert.Equal(expectedExecutions, runtime.ExecutionCount);
    }

    [Fact]
    public async Task ReadOnlyBatch_ExecutesConcurrently()
    {
        var runtime = new ConcurrencyProbeToolRuntime(InternalToolRisk.ReadOnly, ToolDelay);
        using var orchestrator = new ToolUseOrchestrator(
            new BatchTurnClient(runtime.ToolIds),
            runtime,
            new VoiceAssistantToolOptions { MaxParallelToolCalls = 4 });
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        // Three independent reads should cost about one delay, not three.
        Assert.Equal(3, runtime.ExecutionCount);
        Assert.Equal(3, runtime.MaxObservedConcurrency);
    }

    [Fact]
    public async Task SideEffectingBatch_StaysSerialInProviderCallOrder()
    {
        var runtime = new ConcurrencyProbeToolRuntime(InternalToolRisk.SideEffecting, ToolDelay);
        using var orchestrator = new ToolUseOrchestrator(
            new BatchTurnClient(runtime.ToolIds),
            runtime,
            new VoiceAssistantToolOptions { MaxParallelToolCalls = 4 });
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        // External side effects must land in a deterministic order, so overlap is not allowed even
        // when the host permits concurrency.
        Assert.Equal(1, runtime.MaxObservedConcurrency);
        Assert.Equal(["call-0", "call-1", "call-2"], runtime.CompletionOrder);
    }

    [Fact]
    public async Task MaxParallelToolCallsOfOne_RestoresFullySerialExecution()
    {
        var runtime = new ConcurrencyProbeToolRuntime(InternalToolRisk.ReadOnly, ToolDelay);
        using var orchestrator = new ToolUseOrchestrator(
            new BatchTurnClient(runtime.ToolIds),
            runtime,
            new VoiceAssistantToolOptions { MaxParallelToolCalls = 1 });
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        Assert.Equal(1, runtime.MaxObservedConcurrency);
    }

    [Fact]
    public async Task ConcurrentBatch_KeepsResultsInProviderCallOrder()
    {
        // Later calls finish first, so a positional bug would visibly reorder the transcript.
        var runtime = new ReverseLatencyToolRuntime(InternalToolRisk.ReadOnly);
        using var orchestrator = new ToolUseOrchestrator(
            new BatchTurnClient(runtime.ToolIds),
            runtime,
            new VoiceAssistantToolOptions { MaxParallelToolCalls = 4 });
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(
            ["call-0", "call-1", "call-2"],
            completed.ToolResults.Select(static result => result.CallId));
        Assert.Equal(["call-2", "call-1", "call-0"], runtime.CompletionOrder);
    }

    [Fact]
    public async Task ReadOnlyTransientFailure_IsRetriedUntilItSucceeds()
    {
        var runtime = new FlakyToolRuntime(InternalToolRisk.ReadOnly, failuresBeforeSuccess: 2);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions
            {
                MaxToolRetryAttempts = 2,
                ToolRetryBaseDelayMs = 1,
            });
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(
            AssistantToolResultKind.Success,
            Assert.Single(completed.ToolResults).Kind);
        Assert.Equal(3, runtime.AttemptCount);
    }

    [Fact]
    public async Task SideEffectingTransientFailure_IsNeverRetried()
    {
        var runtime = new FlakyToolRuntime(InternalToolRisk.SideEffecting, failuresBeforeSuccess: 1);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions
            {
                MaxToolRetryAttempts = 3,
                ToolRetryBaseDelayMs = 1,
            });
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        // Retrying a call that may already have mutated external state is unsafe, so the failure is
        // reported as-is and the model decides what to do.
        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(AssistantToolResultKind.ToolError, Assert.Single(completed.ToolResults).Kind);
        Assert.Equal(1, runtime.AttemptCount);
    }

    [Fact]
    public async Task RetryDisabled_ReportsTheFirstFailure()
    {
        var runtime = new FlakyToolRuntime(InternalToolRisk.ReadOnly, failuresBeforeSuccess: 1);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions { MaxToolRetryAttempts = 0 });
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        Assert.Equal(1, runtime.AttemptCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TotalToolBudget_CancelsWaitingInitialAttemptOrRetry(bool failFirstAttempt)
    {
        var runtime = new BudgetProbeToolRuntime(failFirstAttempt);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions
            {
                MaxToolRetryAttempts = 2,
                ToolRetryBaseDelayMs = 1,
                MaxToolRetryBudgetMs = 100,
            });
        await orchestrator.StartAsync(CancellationToken.None);
        using var cancellation = new CancellationTokenSource();
        Task<ModelTurnEvent[]> generation = CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), cancellation.Token));
        try
        {
            ModelTurnEvent[] events = await generation.WaitAsync(TimeSpan.FromSeconds(1));
            ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
            Assert.Equal(AssistantToolResultKind.TimedOut, Assert.Single(completed.ToolResults).Kind);
            Assert.Equal(failFirstAttempt ? 2 : 1, runtime.AttemptCount);
            Assert.True(runtime.WaitingAttemptReleased);
        }
        finally
        {
            cancellation.Cancel();
            try { await generation; } catch (OperationCanceledException) { }
        }
    }

    [Fact]
    public async Task NonTransientToolError_IsNotRetried()
    {
        var runtime = new FlakyToolRuntime(
            InternalToolRisk.ReadOnly,
            failuresBeforeSuccess: 1,
            errorCode: "weather_location_invalid");
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions
            {
                MaxToolRetryAttempts = 3,
                ToolRetryBaseDelayMs = 1,
            });
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        // A tool that rejected the request on its merits will reject it again; retrying only wastes
        // the caller's turn.
        Assert.Equal(1, runtime.AttemptCount);
    }

    [Fact]
    public async Task InvalidArguments_AreRejectedBeforeTheToolRuns()
    {
        var tool = new SchemaCheckedTool();
        var builder = new McpToolsBuilder();
        builder.AddTool(tool);
        IAssistantToolRuntime runtime = builder.BuildFactory()
            .Create(new AssistantToolRuntimePolicy(ValidateArguments: true));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            AssistantToolResult result = await runtime.ExecuteAsync(
                new AssistantToolCall(
                    "call-1",
                    new AssistantToolId("local", "get_forecast"),
                    OwnedJsonValue.Parse("{\"days\":\"tomorrow\"}")),
                CancellationToken.None);

            Assert.Equal(AssistantToolResultKind.InputRequired, result.Kind);
            Assert.Equal("schema_validation_failed", result.ErrorCode);
            Assert.Equal(0, tool.CallCount);

            // The model needs to know which argument was wrong to correct itself.
            string detail = result.Value!.CanonicalJson;
            Assert.Contains("invalid_arguments", detail);
            Assert.Contains("days", detail);
        }
        finally
        {
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public async Task MissingRequiredArgument_IsRejected()
    {
        var tool = new SchemaCheckedTool();
        var builder = new McpToolsBuilder();
        builder.AddTool(tool);
        IAssistantToolRuntime runtime = builder.BuildFactory()
            .Create(new AssistantToolRuntimePolicy(ValidateArguments: true));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            AssistantToolResult result = await runtime.ExecuteAsync(
                new AssistantToolCall(
                    "call-1",
                    new AssistantToolId("local", "get_forecast"),
                    OwnedJsonValue.Parse("{\"days\":3}")),
                CancellationToken.None);

            Assert.Equal(AssistantToolResultKind.InputRequired, result.Kind);
            Assert.Equal(0, tool.CallCount);
        }
        finally
        {
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public async Task ValidArguments_ReachTheTool()
    {
        var tool = new SchemaCheckedTool();
        var builder = new McpToolsBuilder();
        builder.AddTool(tool);
        IAssistantToolRuntime runtime = builder.BuildFactory()
            .Create(new AssistantToolRuntimePolicy(ValidateArguments: true));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            AssistantToolResult result = await runtime.ExecuteAsync(
                new AssistantToolCall(
                    "call-1",
                    new AssistantToolId("local", "get_forecast"),
                    OwnedJsonValue.Parse("{\"city\":\"杭州\",\"days\":3}")),
                CancellationToken.None);

            Assert.Equal(AssistantToolResultKind.Success, result.Kind);
            Assert.Equal(1, tool.CallCount);
        }
        finally
        {
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public async Task ValidationDisabled_LetsTheToolSeeInvalidArguments()
    {
        var tool = new SchemaCheckedTool();
        var builder = new McpToolsBuilder();
        builder.AddTool(tool);
        IAssistantToolRuntime runtime = builder.BuildFactory()
            .Create(new AssistantToolRuntimePolicy(ValidateArguments: false));
        await runtime.StartAsync(CancellationToken.None);
        try
        {
            AssistantToolResult result = await runtime.ExecuteAsync(
                new AssistantToolCall(
                    "call-1",
                    new AssistantToolId("local", "get_forecast"),
                    OwnedJsonValue.Parse("{\"days\":\"tomorrow\"}")),
                CancellationToken.None);

            Assert.NotEqual(AssistantToolResultKind.InputRequired, result.Kind);
            Assert.Equal(1, tool.CallCount);
        }
        finally
        {
            await runtime.DisposeAsync();
        }
    }

    [Fact]
    public async Task OrchestratorBoundToHistory_CarriesItsToolExchangeIntoTheNextPrompt()
    {
        const long turnId = 71;
        var runtime = new PayloadEchoToolRuntime("31C humidity 62 percent");
        var history = new ConversationHistoryStore(
            "system",
            maxHistoryTurns: 4,
            toolTranscriptRetentionTurns: 3);
        ChatMessage user = history.AddUserMessage(turnId, "北京天气", out ChatMessage[] prompt);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions());
        orchestrator.BindToolHistorySink(history);
        await orchestrator.StartAsync(CancellationToken.None);

        await CollectAsync(orchestrator.GenerateAsync(
            new ModelTurnRequest(prompt, new ModelGenerationOptions(), ToolCatalogSnapshot.Empty, turnId),
            CancellationToken.None));
        Assert.True(history.CommitGeneratedResponse(turnId, user, "北京晴，31度"));

        history.AddUserMessage(72, "那穿短袖行吗", out ChatMessage[] followUp);

        // The follow-up turn sees the structured result, not just the spoken summary.
        Assert.Equal(
            [ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant, ChatRole.User],
            followUp.Select(static message => message.Role));
        Assert.Equal(
            "call-1",
            Assert.IsType<ToolCallContent>(Assert.Single(followUp[2].Content)).CallId);
        Assert.Contains(
            "31C humidity 62 percent",
            Assert.IsType<ToolResultContent>(Assert.Single(followUp[3].Content)).ResultJson);
        var audit = Assert.Single(history.GetToolAuditTrailSnapshot());
        Assert.Equal(turnId, audit.TurnId);
        Assert.Equal("call-1", audit.CallId);
        Assert.Equal(AssistantToolResultKind.Success, audit.ResultKind);
        // The user-facing projection stays free of tool plumbing.
        Assert.Equal(
            [ChatRole.User, ChatRole.Assistant, ChatRole.User],
            history.GetUserConversationHistorySnapshot().Select(static message => message.Role));
    }

    [Fact]
    public async Task WideConcurrentBatch_ResolvesEveryCallExactlyOnce()
    {
        // Results are written positionally from many tasks at once. A lost or duplicated write
        // would leave a dangling call id in the provider transcript.
        var runtime = new ConcurrencyProbeToolRuntime(
            InternalToolRisk.ReadOnly,
            TimeSpan.FromMilliseconds(20),
            toolCount: 12);
        using var orchestrator = new ToolUseOrchestrator(
            new BatchTurnClient(runtime.ToolIds),
            runtime,
            new VoiceAssistantToolOptions { MaxParallelToolCalls = 8 });
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        ModelTurnCompleted completed = Assert.Single(events.OfType<ModelTurnCompleted>());
        Assert.Equal(
            Enumerable.Range(0, 12).Select(static index => $"call-{index}"),
            completed.ToolResults.Select(static result => result.CallId));
        Assert.All(
            completed.ToolResults,
            static result => Assert.Equal(AssistantToolResultKind.Success, result.Kind));
        Assert.Equal(12, runtime.ExecutionCount);
        // The host cap is honoured even though more calls were eligible to overlap.
        Assert.True(
            runtime.MaxObservedConcurrency <= 8,
            $"Observed {runtime.MaxObservedConcurrency} concurrent executions, expected at most 8.");
    }

    [Fact]
    public async Task ToolPayloadsAndValidationDetail_NeverReachSpeakableText()
    {
        const string secret = "RAW_TOOL_PAYLOAD_MUST_NOT_BE_SPOKEN";
        var runtime = new PayloadEchoToolRuntime(secret);
        using var orchestrator = new ToolUseOrchestrator(
            new SingleCallTurnClient(runtime.ToolId),
            runtime,
            new VoiceAssistantToolOptions());
        await orchestrator.StartAsync(CancellationToken.None);

        ModelTurnEvent[] events = await CollectAsync(
            orchestrator.GenerateAsync(CreateRequest(), CancellationToken.None));

        // Only the model's own answer is speakable. Tool results, error codes and validation detail
        // are model-facing plumbing and must not leak into the speech path.
        string speakable = string.Concat(events.OfType<VisibleTextDelta>().Select(static d => d.Text));
        Assert.Equal("完成。", speakable);
        Assert.DoesNotContain(secret, speakable, StringComparison.Ordinal);
        Assert.DoesNotContain("schema_validation_failed", speakable, StringComparison.Ordinal);
        Assert.DoesNotContain("invalid_arguments", speakable, StringComparison.Ordinal);
    }

    private static ModelTurnRequest CreateRequest() => new(
        [ChatMessage.System("system"), ChatMessage.User("批量查询")],
        new ModelGenerationOptions(),
        ToolCatalogSnapshot.Empty);

    private static async Task<T[]> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var result = new List<T>();
        await foreach (T item in source)
            result.Add(item);
        return result.ToArray();
    }

    private static AssistantToolDefinition CreateDefinition(
        string toolName,
        InternalToolRisk risk) => new(
        new AssistantToolId("local", toolName),
        "批量工具",
        "批量工具",
        OwnedJsonValue.Parse("{\"type\":\"object\"}"),
        null,
        risk,
        AssistantToolExecutionPolicy.ForegroundOnly,
        null,
        TimeSpan.FromSeconds(5));

    /// <summary>Records peak overlap so a concurrency claim is measured, not inferred from timing.</summary>
    private sealed class ConcurrencyProbeToolRuntime : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog;
        private readonly TimeSpan _delay;
        private readonly object _gate = new();
        private readonly List<string> _completionOrder = [];
        private int _active;
        private int _maxObserved;
        private int _executionCount;
        private int _authorizationCount;

        public ConcurrencyProbeToolRuntime(InternalToolRisk risk, TimeSpan delay, int toolCount = 3)
        {
            AssistantToolDefinition[] definitions = Enumerable.Range(0, toolCount)
                .Select(index => CreateDefinition($"tool_{index}", risk))
                .ToArray();
            _catalog = new ToolCatalogSnapshot(definitions);
            _delay = delay;
            ToolIds = definitions.Select(static definition => definition.Id).ToArray();
        }

        public AssistantToolId[] ToolIds { get; }

        public ToolApprovalDecision ApprovalDecision { get; init; } = ToolApprovalDecision.Allowed;

        public int AuthorizationCount => Volatile.Read(ref _authorizationCount);

        public int ExecutionCount => Volatile.Read(ref _executionCount);

        public int MaxObservedConcurrency => Volatile.Read(ref _maxObserved);

        public string[] CompletionOrder
        {
            get { lock (_gate) return _completionOrder.ToArray(); }
        }

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken) => ValueTask.FromResult(_catalog);
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _authorizationCount);
            return ValueTask.FromResult(ApprovalDecision);
        }

        public async ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _executionCount);
            int active = Interlocked.Increment(ref _active);
            int observed = Volatile.Read(ref _maxObserved);
            while (active > observed &&
                Interlocked.CompareExchange(ref _maxObserved, active, observed) != observed)
            {
                observed = Volatile.Read(ref _maxObserved);
            }

            try
            {
                await Task.Delay(_delay, cancellationToken);
            }
            finally
            {
                Interlocked.Decrement(ref _active);
            }

            lock (_gate)
                _completionOrder.Add(call.CallId);
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}"));
        }
    }

    /// <summary>Finishes calls in reverse request order to expose any completion-order coupling.</summary>
    private sealed class ReverseLatencyToolRuntime : IAssistantToolRuntime
    {
        private readonly ToolCatalogSnapshot _catalog;
        private readonly object _gate = new();
        private readonly List<string> _completionOrder = [];

        public ReverseLatencyToolRuntime(InternalToolRisk risk)
        {
            AssistantToolDefinition[] definitions =
            [
                CreateDefinition("tool_0", risk),
                CreateDefinition("tool_1", risk),
                CreateDefinition("tool_2", risk),
            ];
            _catalog = new ToolCatalogSnapshot(definitions);
            ToolIds = definitions.Select(static definition => definition.Id).ToArray();
        }

        public AssistantToolId[] ToolIds { get; }

        public string[] CompletionOrder
        {
            get { lock (_gate) return _completionOrder.ToArray(); }
        }

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
            int index = call.CallId switch
            {
                "call-0" => 2,
                "call-1" => 1,
                _ => 0,
            };
            await Task.Delay(TimeSpan.FromMilliseconds(60 * (index + 1)), cancellationToken);
            lock (_gate)
                _completionOrder.Add(call.CallId);
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}"));
        }
    }

    /// <summary>Fails a fixed number of times before succeeding, to exercise retry policy.</summary>
    private sealed class FlakyToolRuntime(
        InternalToolRisk risk,
        int failuresBeforeSuccess,
        string errorCode = nameof(HttpRequestException)) : IAssistantToolRuntime
    {
        private readonly AssistantToolDefinition _definition = CreateDefinition("flaky", risk);
        private int _attemptCount;

        public AssistantToolId ToolId => _definition.Id;

        public int AttemptCount => Volatile.Read(ref _attemptCount);

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken)
            => ValueTask.FromResult(new ToolCatalogSnapshot([_definition]));
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            int attempt = Interlocked.Increment(ref _attemptCount);
            if (attempt <= failuresBeforeSuccess)
            {
                return ValueTask.FromResult(new AssistantToolResult(
                    call.CallId,
                    AssistantToolResultKind.ToolError,
                    null,
                    errorCode));
            }

            return ValueTask.FromResult(new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse("{\"ok\":true}")));
        }
    }

    /// <summary>Waits for cancellation to expose the total budget on the first call and retries.</summary>
    private sealed class BudgetProbeToolRuntime(bool failFirstAttempt) : IAssistantToolRuntime
    {
        private readonly AssistantToolDefinition _definition = CreateDefinition("budget_probe", InternalToolRisk.ReadOnly);
        public AssistantToolId ToolId => _definition.Id;
        public int AttemptCount { get; private set; }
        public bool WaitingAttemptReleased { get; private set; }
        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(ToolRequestContext context, CancellationToken cancellationToken)
            => ValueTask.FromResult(new ToolCatalogSnapshot([_definition]));
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(AssistantToolCall call, CancellationToken cancellationToken)
            => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public async ValueTask<AssistantToolResult> ExecuteAsync(AssistantToolCall call, CancellationToken cancellationToken)
        {
            AttemptCount++;
            if (failFirstAttempt && AttemptCount == 1)
                return new AssistantToolResult(call.CallId, AssistantToolResultKind.ToolError, null, "transport_error");
            try
            {
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                throw new InvalidOperationException("Waiting tool should only finish through cancellation.");
            }
            finally
            {
                WaitingAttemptReleased = true;
            }
        }
    }

    /// <summary>Returns a recognizable payload so a leak into the speech path is unambiguous.</summary>
    private sealed class PayloadEchoToolRuntime(string secret) : IAssistantToolRuntime
    {
        private readonly AssistantToolDefinition _definition =
            CreateDefinition("echo", InternalToolRisk.ReadOnly);

        public AssistantToolId ToolId => _definition.Id;

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken)
            => ValueTask.FromResult(new ToolCatalogSnapshot([_definition]));
        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken) => ValueTask.FromResult(ToolApprovalDecision.Allowed);

        public ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
            => ValueTask.FromResult(new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Success,
                OwnedJsonValue.Parse($"{{\"payload\":\"{secret}\"}}")));
    }

    private sealed class BatchTurnClient(AssistantToolId[] toolIds) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (_round++ == 0)
            {
                for (int index = 0; index < toolIds.Length; index++)
                {
                    yield return new ToolCallRequested(
                        $"call-{index}",
                        toolIds[index],
                        OwnedJsonValue.Parse("{}"));
                }
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            yield return new VisibleTextDelta("全部完成。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class SingleCallTurnClient(AssistantToolId toolId) : ILlmTurnClient
    {
        private int _round;

        public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
            ModelTurnRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.CompletedTask;
            if (_round++ == 0)
            {
                yield return new ToolCallRequested("call-1", toolId, OwnedJsonValue.Parse("{}"));
                yield return new ModelTurnCompleted(default);
                yield break;
            }

            yield return new VisibleTextDelta("完成。");
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class SchemaCheckedTool : IAssistantTool
    {
        private int _callCount;

        public int CallCount => Volatile.Read(ref _callCount);

        public AssistantToolDescriptor Descriptor { get; } = new(
            "get_forecast",
            "查询天气预报",
            "查询指定城市未来若干天的天气",
            """
            {
              "type": "object",
              "properties": {
                "city": { "type": "string" },
                "days": { "type": "integer", "minimum": 1, "maximum": 7 }
              },
              "required": ["city", "days"],
              "additionalProperties": false
            }
            """,
            executionPolicy: AssistantToolExecutionPolicy.ForegroundOnly);

        public ValueTask<AssistantToolExecutionResult> ExecuteAsync(
            AssistantToolInvocation invocation,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _callCount);
            return ValueTask.FromResult(AssistantToolExecutionResult.Success("晴。", "{\"ok\":true}"));
        }

        public void Dispose() { }
    }
}
