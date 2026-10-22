using Microsoft.Extensions.Logging.Abstractions;
using System.Reflection;
using System.Text.Json;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Adapters;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Response.Adapters;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.Core.Services.TTSService;
using Xunit;
using AssistantToolExecutionPolicy = VoiceBox.Assistant.Api.Tools.Models.AssistantToolExecutionPolicy;

namespace VoiceBox.Assistant.Tests;

public sealed class ResponseModuleContractTests
{
    [Fact]
    public void ModelTurnRequest_CopiesTranscriptOptionsAndToolSnapshot()
    {
        ChatMessage[] transcript = [ChatMessage.System("system"), ChatMessage.User("question")];
        var options = new ModelGenerationOptions(MaxTokens: 128, EnableThinking: true);
        var request = new ModelTurnRequest(
            transcript,
            options,
            ToolCatalogSnapshot.Empty,
            turnId: 17);

        transcript[1] = ChatMessage.User("mutated");

        Assert.Equal("question", TextOf(request.Transcript[1]));
        Assert.Equal(128, request.Options.MaxTokens);
        Assert.True(request.Options.EnableThinking);
        Assert.Empty(request.ToolCatalog.Definitions);
        Assert.Equal(17, request.TurnId);
    }

    [Fact]
    public async Task TextOnlyAdapter_PreservesPromptOptionsAndTypedEventOrder()
    {
        var llm = new RecordingLlmService(
            new LlmTextChunk { Kind = LlmChunkKind.Thinking, Text = "think", IsFinal = false },
            new LlmTextChunk { Kind = LlmChunkKind.Text, Text = "answer", IsFinal = true });
        using var adapter = new TextOnlyLlmTurnClientAdapter(llm);
        var request = new ModelTurnRequest(
            [ChatMessage.System("system"), ChatMessage.User("question")],
            new ModelGenerationOptions(
                MaxTokens: 64,
                DoSample: false,
                Temperature: 0.2f,
                TopK: 7,
                TopP: 0.8f,
                RepetitionPenalty: 1.1f,
                EnableThinking: true),
            ToolCatalogSnapshot.Empty);

        List<ModelTurnEvent> events = await CollectAsync(adapter.RunAsync(request, CancellationToken.None));

        Assert.Collection(
            events,
            value => Assert.Equal("think", Assert.IsType<ThinkingDelta>(value).Text),
            value => Assert.Equal("answer", Assert.IsType<VisibleTextDelta>(value).Text),
            value => Assert.IsType<ModelTurnCompleted>(value));
        Assert.Equal(["system", "question"], Assert.Single(llm.Prompts).Select(TextOf));
        LlmCompleteOptions actual = Assert.Single(llm.Options);
        Assert.Equal(64, actual.MaxTokens);
        Assert.False(actual.DoSample);
        Assert.Equal(0.2f, actual.Temperature);
        Assert.Equal(7, actual.TopK);
        Assert.Equal(0.8f, actual.TopP);
        Assert.Equal(1.1f, actual.RepetitionPenalty);
        Assert.True(actual.EnableThinking);
    }

    [Fact]
    public async Task TextOnlyAdapter_UsesSerializedConversationAndOwnsOnlyConversation()
    {
        var llm = new ConversationalRecordingLlmService();
        using (var adapter = new TextOnlyLlmTurnClientAdapter(llm))
        {
            var request = new ModelTurnRequest(
                [ChatMessage.System("system"), ChatMessage.User("one")],
                new ModelGenerationOptions(),
                ToolCatalogSnapshot.Empty);

            await CollectAsync(adapter.RunAsync(request, CancellationToken.None));
            await CollectAsync(adapter.RunAsync(request, CancellationToken.None));
        }

        Assert.Equal(1, llm.CreateConversationCalls);
        Assert.Equal(2, llm.Conversation.StreamCalls);
        Assert.Equal(1, llm.Conversation.MaxConcurrentCalls);
        Assert.Equal(1, llm.Conversation.DisposeCalls);
        Assert.Equal(0, llm.DisposeCalls);
    }

    [Fact]
    public void ToolContracts_OwnJsonAndProduceStableSortedCollisionCheckedSnapshots()
    {
        OwnedJsonValue firstArguments;
        using (JsonDocument document = JsonDocument.Parse("{\"b\":2,\"a\":1}"))
            firstArguments = new OwnedJsonValue(document.RootElement);
        var reorderedArguments = OwnedJsonValue.Parse("{\"a\":1,\"b\":2}");

        Assert.Equal("{\"a\":1,\"b\":2}", firstArguments.CanonicalJson);
        Assert.Equal(firstArguments.CanonicalHash, reorderedArguments.CanonicalHash);

        AssistantToolDefinition[] definitions =
        [
            CreateDefinition("server-b", "lookup"),
            CreateDefinition("server-a", "lookup"),
        ];
        var snapshot = new ToolCatalogSnapshot(definitions);
        definitions[0] = CreateDefinition("mutated", "mutated");

        Assert.Equal(["server-a", "server-b"], snapshot.Definitions.Select(value => value.Id.ServerAlias));
        Assert.Equal(snapshot.Hash, new ToolCatalogSnapshot(snapshot.Definitions.Reverse()).Hash);
        Assert.Equal(2, snapshot.ExposedNames.Count);

        AssistantToolDefinition promptA = CreateDefinition(
            "local",
            "weather",
            AssistantToolExecutionPolicy.BackgroundOnly,
            "正在查询天气");
        AssistantToolDefinition promptB = CreateDefinition(
            "local",
            "weather",
            AssistantToolExecutionPolicy.BackgroundOnly,
            "正在获取天气");
        Assert.NotEqual(
            new ToolCatalogSnapshot([promptA]).Hash,
            new ToolCatalogSnapshot([promptB]).Hash);

        Assert.Throws<InvalidOperationException>(() => new ToolCatalogSnapshot(
        [
            CreateDefinition("server-a", "same-name"),
            CreateDefinition("server_a", "same_name"),
        ]));
    }

    [Fact]
    public void ResponseAndPlaybackBoundaries_DoNotOwnProvidersKernelOrPublicDispatcher()
    {
        Type responseModule = typeof(ResponseModule);
        string[] responseFields = responseModule
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .Select(field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();

        Assert.Contains(typeof(IResponseGenerator).FullName, responseFields);
        Assert.Contains(typeof(IResponsePlaybackSessionFactory).FullName, responseFields);
        Assert.DoesNotContain(typeof(IStreamLLMService).FullName, responseFields);
        Assert.DoesNotContain(typeof(IStreamTTSService).FullName, responseFields);
        Assert.DoesNotContain(typeof(AssistantEventDispatcher).FullName, responseFields);
        Assert.DoesNotContain(responseFields, value => value.Contains("TurnKernel", StringComparison.Ordinal));

        MethodInfo play = typeof(IResponsePlaybackSession).GetMethod("PlayAsync")!;
        Assert.Equal(typeof(IAsyncEnumerable<SpeakableTextDelta>), play.GetParameters()[0].ParameterType);

        string[] pumpFields = typeof(TtsPlaybackPump)
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .Select(field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();
        Assert.Contains(typeof(ISpeechSynthesisStreamPort).FullName, pumpFields);
        Assert.Contains(typeof(BoundedPlaybackWriter).FullName, pumpFields);
        Assert.Contains(typeof(PlaybackDrainWaiter).FullName, pumpFields);
        Assert.DoesNotContain(typeof(IStreamTTSService).FullName, pumpFields);
    }

    [Fact]
    public void ToolUseSurface_KeepsProtocolSdkOutsideCoreAssembly()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type[] types = assembly.GetTypes();

        Assert.DoesNotContain(
            types,
            type => type.IsClass && !type.IsAbstract && typeof(IAssistantToolRuntime).IsAssignableFrom(type));
        Assert.DoesNotContain(
            assembly.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("ModelContextProtocol.", StringComparison.Ordinal) == true);
    }

    [Theory]
    [InlineData("")]
    [InlineData("...")]
    public async Task ResponseModule_UnusableTextReleasesTurnResourcesExactlyOnce(string userText)
    {
        // The unusable-text short circuit returns before generation. It must hand the borrowed
        // response context back exactly once: a second hand-off for the same turn corrupts the
        // pending-release ledger, and none at all leaks the turn's CTS for the process lifetime.
        var generator = new FixedResponseGenerator(new ModelTurnCompleted(default));
        var control = new RecordingResponseControl { CurrentTurnId = 7 };
        var resources = new RecordingTurnResourcePort();
        using ResponseModule module = CreateResponseModule(
            generator,
            new RecordingPlaybackFactory(),
            control,
            new RecordingTextSink(),
            new ConversationHistoryStore("system", 4));

        ResponseTurnResult result = await module.RunRecognizedTextTurnAsync(
            7,
            userText,
            userText,
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 707,
            turnResources: resources);

        Assert.Null(result.UserMessage);
        Assert.Equal(1, resources.AttachCount);
        module.ReleaseOperationContext(7);
        Assert.Equal(1, resources.ReleaseCount);
    }

    [Fact]
    public async Task ResponseModule_EndConversationKeywordReleasesTurnResourcesExactlyOnce()
    {
        var generator = new FixedResponseGenerator(new ModelTurnCompleted(default));
        var control = new RecordingResponseControl { CurrentTurnId = 8 };
        var resources = new RecordingTurnResourcePort();
        using ResponseModule module = CreateResponseModule(
            generator,
            new RecordingPlaybackFactory(),
            control,
            new RecordingTextSink(),
            new ConversationHistoryStore("system", 4));

        ResponseTurnResult result = await module.RunRecognizedTextTurnAsync(
            8,
            "结束对话",
            "结束对话",
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 808,
            canEndContinuousConversation: true,
            turnResources: resources);

        Assert.True(result.EndConversationRequested);
        Assert.Equal(1, resources.AttachCount);
        module.ReleaseOperationContext(8);
        Assert.Equal(1, resources.ReleaseCount);
    }

    [Fact]
    public async Task ResponseModule_LateWrongTurnOperationCannotPublishTextCommitOrCompletion()
    {
        var firstStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFirst = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var generator = new SequencedResponseGenerator(firstStarted, releaseFirst);
        var playback = new RecordingPlaybackFactory();
        var control = new RecordingResponseControl { CurrentTurnId = 1 };
        var text = new RecordingTextSink();
        var history = new ConversationHistoryStore("system", 4);
        using ResponseModule module = CreateResponseModule(generator, playback, control, text, history);

        Task<ResponseTurnResult> stale = module.RunRecognizedTextTurnAsync(
            1,
            "first",
            "first",
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 101);
        await firstStarted.Task.WaitAsync(TimeSpan.FromSeconds(1));

        control.CurrentTurnId = 2;
        ResponseTurnResult current = await module.RunRecognizedTextTurnAsync(
            2,
            "second",
            "second",
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 202);
        releaseFirst.TrySetResult();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stale);
        Assert.NotNull(current.UserMessage);
        Assert.Equal(["current"], text.Visible);
        Assert.Equal(["current"], playback.Deltas);
        Assert.Equal([(2L, 202L, "current")], control.ResponseCompletions);
        Assert.Equal([(2L, 202L, "current")], control.PlaybackCompletions);
    }

    [Fact]
    public async Task ResponseModule_ToolCallWithoutCapabilityFailsWithoutTextTtsOrAssistantHistory()
    {
        var generator = new ToolCallResponseGenerator();
        var playback = new RecordingPlaybackFactory();
        var control = new RecordingResponseControl { CurrentTurnId = 3 };
        var text = new RecordingTextSink();
        var history = new ConversationHistoryStore("system", 4);
        using ResponseModule module = CreateResponseModule(generator, playback, control, text, history);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            module.RunRecognizedTextTurnAsync(
                3,
                "use tool",
                "use tool",
                TimeSpan.Zero,
                CancellationToken.None,
                operationId: 303));

        Assert.Contains("without tool capability", error.Message, StringComparison.Ordinal);
        Assert.Empty(text.Visible);
        Assert.Empty(playback.Deltas);
        Assert.DoesNotContain(
            history.GetUserConversationHistorySnapshot(),
            message => message.Role == ChatRole.Assistant);
        Assert.Equal([(3L, 303L, ResponseOperationOutcomeKind.Failed)], control.TerminalOutcomes);
    }

    [Fact]
    public async Task ResponseModule_PropagatesForcedSpeechBoundaryToPlayback()
    {
        var playback = new RecordingPlaybackFactory();
        var control = new RecordingResponseControl { CurrentTurnId = 4 };
        var text = new RecordingTextSink();
        var history = new ConversationHistoryStore("system", 4);
        using ResponseModule module = CreateResponseModule(
            new FixedResponseGenerator(
                new VisibleTextDelta("正在查询，请稍后", ForceSpeechBoundary: true),
                new ModelTurnCompleted(default)),
            playback,
            control,
            text,
            history);

        await module.RunRecognizedTextTurnAsync(
            4,
            "查询天气",
            "查询天气",
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 404);

        Assert.Equal(["正在查询，请稍后"], playback.Deltas);
        Assert.Equal([true], playback.ForcedBoundaries);
        Assert.Equal(["正在查询，请稍后"], text.Visible);
    }

    [Fact]
    public async Task ToolRuntime_RoundTripsStrongIdentityCorrelationApprovalAndDistinctOutcomes()
    {
        AssistantToolDefinition definition = CreateDefinition("server", "lookup");
        await using var runtime = new InMemoryToolRuntime(
            new ToolCatalogSnapshot([definition]),
            ToolApprovalDecision.Allowed,
            new AssistantToolResult(
                "call-7",
                AssistantToolResultKind.UnknownOutcome,
                OwnedJsonValue.Parse("{\"accepted\":true}")));
        var call = new AssistantToolCall(
            "call-7",
            definition.Id,
            OwnedJsonValue.Parse("{\"query\":\"value\"}"));

        await runtime.StartAsync(CancellationToken.None);
        ToolCatalogSnapshot snapshot = await runtime.GetSnapshotAsync(
            new ToolRequestContext(7),
            CancellationToken.None);
        ToolApprovalDecision decision = await runtime.AuthorizeAsync(call, CancellationToken.None);
        AssistantToolResult result = await runtime.ExecuteAsync(call, CancellationToken.None);
        await runtime.StopAsync();

        Assert.Same(definition, Assert.Single(snapshot.Definitions));
        Assert.Equal(ToolApprovalDecision.Allowed, decision);
        Assert.Equal(call.CallId, result.CallId);
        Assert.Equal(call.ToolId, definition.Id);
        Assert.Equal(call.Arguments.CanonicalHash, call.ArgumentsHash);
        Assert.Equal(10, Enum.GetValues<AssistantToolResultKind>().Distinct().Count());
        Assert.Equal(1, runtime.CatalogCalls);
        Assert.Equal(1, runtime.ApprovalCalls);
        Assert.Equal(1, runtime.ExecutionCalls);
    }

    [Fact]
    public async Task SameIngressResponseSchedule_ReplaysOneHundredTimesWithIdenticalTrace()
    {
        string? expected = null;
        for (int iteration = 0; iteration < 100; iteration++)
        {
            var generator = new FixedResponseGenerator(
                new ThinkingDelta("think"),
                new VisibleTextDelta("answer"),
                new ModelTurnCompleted(default));
            var playback = new RecordingPlaybackFactory();
            var control = new RecordingResponseControl { CurrentTurnId = 1 };
            var text = new RecordingTextSink();
            var history = new ConversationHistoryStore("system", 4);
            using ResponseModule module = CreateResponseModule(generator, playback, control, text, history);

            await module.RunRecognizedTextTurnAsync(
                1,
                "question",
                "question",
                TimeSpan.FromMilliseconds(20),
                CancellationToken.None,
                operationId: 404);
            string trace = string.Join(
                '|',
                control.Signals
                    .Concat(text.Visible.Select(value => $"text:{value}"))
                    .Concat(playback.Deltas.Select(value => $"tts:{value}")));
            expected ??= trace;
            Assert.Equal(expected, trace);
        }
    }

    [Fact]
    public async Task VisibleChunkPartitioning_DoesNotIncreaseControlBoundaryCount()
    {
        string[] oneChunk = await RunChunkingTraceAsync("complete answer");
        string[] threeChunks = await RunChunkingTraceAsync("complete", " ", "answer");

        Assert.Equal(oneChunk, threeChunks);
        Assert.Equal(
            ["recognition:1:0", "response-start:1:505", "speakable-start:1:505", "response-complete:1:505", "playback-complete:1:505"],
            oneChunk);
    }

    private static async Task<string[]> RunChunkingTraceAsync(params string[] chunks)
    {
        ModelTurnEvent[] events = chunks
            .Select(static chunk => (ModelTurnEvent)new VisibleTextDelta(chunk))
            .Append(new ModelTurnCompleted(default))
            .ToArray();
        var playback = new RecordingPlaybackFactory();
        var control = new RecordingResponseControl { CurrentTurnId = 1 };
        var history = new ConversationHistoryStore("system", 4);
        using ResponseModule module = CreateResponseModule(
            new FixedResponseGenerator(events),
            playback,
            control,
            new RecordingTextSink(),
            history);

        await module.RunRecognizedTextTurnAsync(
            1,
            "question",
            "question",
            TimeSpan.Zero,
            CancellationToken.None,
            operationId: 505);
        return control.Signals.ToArray();
    }

    private static ResponseModule CreateResponseModule(
        IResponseGenerator generator,
        IResponsePlaybackSessionFactory playback,
        IResponseControlSignalPort control,
        IAssistantTextStreamSink text,
        ConversationHistoryStore history)
        => new(
            new VoiceAssistantOptions(),
            bargeInInstalled: false,
            generator,
            playback,
            history,
            control,
            text,
            new PlaybackStartupGuard(),
            NullLogger.Instance,
            NullBargeInDecisionResetPort.Instance,
            new PlaybackTextEchoGuard());

    private static AssistantToolDefinition CreateDefinition(
        string serverAlias,
        string toolName,
        AssistantToolExecutionPolicy executionPolicy = AssistantToolExecutionPolicy.ForegroundOnly,
        string? executionPrompt = null)
        => new(
            new AssistantToolId(serverAlias, toolName),
            title: "Lookup",
            description: "Looks up a value.",
            inputSchema: OwnedJsonValue.Parse("{\"type\":\"object\"}"),
            outputSchema: null,
            risk: AssistantToolRisk.ReadOnly,
            executionPolicy,
            executionPrompt,
            timeout: null);

    private static string TextOf(ChatMessage message)
        => Assert.IsType<TextContent>(Assert.Single(message.Content)).Text;

    private static async Task<List<T>> CollectAsync<T>(IAsyncEnumerable<T> source)
    {
        var values = new List<T>();
        await foreach (T value in source)
            values.Add(value);
        return values;
    }

    /// <summary>
    /// Records the borrow/return protocol of the runtime-owned response context. It mirrors the
    /// production rule that one turn attaches at most one context and returns it exactly once.
    /// </summary>
    private sealed class RecordingTurnResourcePort : IResponseTurnResourcePort
    {
        private readonly HashSet<long> _attached = [];

        public int AttachCount { get; private set; }

        public int ReleaseCount { get; private set; }

        public CancellationToken RunToken => CancellationToken.None;

        public bool IsShuttingDown => false;

        public bool TryAttachResponseContext(long turnId, CancellationTokenSource responseCts)
        {
            if (!_attached.Add(turnId))
                return false;
            AttachCount++;
            return true;
        }

        public void ReleaseResponseContext(long turnId, CancellationTokenSource responseCts)
        {
            ReleaseCount++;
            _attached.Remove(turnId);
            responseCts.Dispose();
        }

        public ValueTask PublishTurnCompletedAsync(long turnId, bool endConversationRequested)
            => ValueTask.CompletedTask;
    }

    private sealed class RecordingLlmService(params LlmTextChunk[] chunks) : IStreamLLMService
    {
        public List<ChatMessage[]> Prompts { get; } = [];
        public List<LlmCompleteOptions> Options { get; } = [];

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            Prompts.Add(messages.ToArray());
            Options.Add(options ?? new LlmCompleteOptions());
            foreach (LlmTextChunk chunk in chunks)
            {
                ct.ThrowIfCancellationRequested();
                yield return chunk;
                await Task.Yield();
            }
        }

        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => throw new NotSupportedException();

        public void Dispose() { }
    }

    private sealed class ConversationalRecordingLlmService : IConversationalLLMService
    {
        public RecordingConversation Conversation { get; } = new();
        public int CreateConversationCalls { get; private set; }
        public int DisposeCalls { get; private set; }

        public ILLMConversation CreateConversation()
        {
            CreateConversationCalls++;
            return Conversation;
        }

        public IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => throw new InvalidOperationException("The stateful conversation must be used.");

        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => throw new NotSupportedException();

        public void Dispose() => DisposeCalls++;
    }

    private sealed class RecordingConversation : ILLMConversation
    {
        private int _concurrentCalls;
        public int StreamCalls { get; private set; }
        public int MaxConcurrentCalls { get; private set; }
        public int DisposeCalls { get; private set; }

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            StreamCalls++;
            int concurrent = Interlocked.Increment(ref _concurrentCalls);
            MaxConcurrentCalls = Math.Max(MaxConcurrentCalls, concurrent);
            try
            {
                await Task.Yield();
                ct.ThrowIfCancellationRequested();
                yield return new LlmTextChunk { Kind = LlmChunkKind.Text, Text = "ok", IsFinal = true };
            }
            finally
            {
                Interlocked.Decrement(ref _concurrentCalls);
            }
        }

        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => throw new NotSupportedException();

        public void Dispose() => DisposeCalls++;
    }

    private sealed class SequencedResponseGenerator(
        TaskCompletionSource firstStarted,
        TaskCompletionSource releaseFirst) : IResponseGenerator
    {
        private int _callCount;

        public async IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
            ModelTurnRequest request,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            int call = Interlocked.Increment(ref _callCount);
            if (call == 1)
            {
                firstStarted.TrySetResult();
                await releaseFirst.Task.WaitAsync(cancellationToken);
                yield return new VisibleTextDelta("late");
            }
            else
            {
                yield return new VisibleTextDelta("current");
            }
            yield return new ModelTurnCompleted(default);
        }
    }

    private sealed class ToolCallResponseGenerator : IResponseGenerator
    {
        public async IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
            ModelTurnRequest request,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            yield return new ToolCallRequested(
                "call-1",
                new AssistantToolId("server", "tool"),
                OwnedJsonValue.Parse("{\"value\":1}"));
        }
    }

    private sealed class FixedResponseGenerator(params ModelTurnEvent[] events) : IResponseGenerator
    {
        public async IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
            ModelTurnRequest request,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            foreach (ModelTurnEvent modelEvent in events)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return modelEvent;
                await Task.Yield();
            }
        }
    }

    private sealed class RecordingPlaybackFactory : IResponsePlaybackSessionFactory
    {
        public List<string> Deltas { get; } = [];

        public List<bool> ForcedBoundaries { get; } = [];

        public IResponsePlaybackSession Create(long turnId) =>
            new Session(Deltas, ForcedBoundaries);

        private sealed class Session(
            List<string> deltas,
            List<bool> forcedBoundaries) : IResponsePlaybackSession
        {
            public async Task<string> PlayAsync(
                IAsyncEnumerable<SpeakableTextDelta> text,
                CancellationToken cancellationToken)
            {
                var result = new System.Text.StringBuilder();
                await foreach (SpeakableTextDelta delta in text.WithCancellation(cancellationToken))
                {
                    deltas.Add(delta.Text);
                    forcedBoundaries.Add(delta.ForceSpeechBoundary);
                    result.Append(delta.Text);
                }
                return result.ToString();
            }

            public void Dispose() { }
        }
    }

    private sealed class RecordingTextSink : IAssistantTextStreamSink
    {
        public List<string> Visible { get; } = [];

        public long RaiseAssistantText(AssistantTextEventArgs args)
        {
            if (!args.IsThinking)
                Visible.Add(args.Delta);
            return Visible.Count;
        }

        public Task WaitUntilDeliveredAsync(
            long publicationSequence,
            CancellationToken cancellationToken = default)
            => Task.CompletedTask;
    }

    private sealed class RecordingResponseControl : IResponseControlSignalPort
    {
        public long CurrentTurnId { get; set; }
        public List<(long TurnId, long OperationId, string Text)> ResponseCompletions { get; } = [];
        public List<(long TurnId, long OperationId, string Text)> PlaybackCompletions { get; } = [];
        public List<(long TurnId, long OperationId, ResponseOperationOutcomeKind Outcome)> TerminalOutcomes { get; } = [];
        public List<string> Signals { get; } = [];

        public bool IsCurrentTurn(long turnId) => CurrentTurnId == turnId;

        public void OpenInputForBargeIn(long turnId)
            => Signals.Add($"open-input:{turnId}");

        public ValueTask PublishRecognitionCompletedAsync(
            long turnId,
            string text,
            TimeSpan speechDuration,
            CancellationToken cancellationToken)
        {
            Signals.Add($"recognition:{turnId}:0");
            return ValueTask.CompletedTask;
        }

        public ValueTask<ResponseControlTransition> StartResponseAsync(
            long turnId,
            long operationId,
            CancellationToken cancellationToken)
        {
            Signals.Add($"response-start:{turnId}:{operationId}");
            return ValueTask.FromResult(new ResponseControlTransition(IsCurrentTurn(turnId), false, 0));
        }

        public ValueTask<ResponseControlTransition> StartSpeakableResponseAsync(
            long turnId,
            long operationId,
            CancellationToken cancellationToken)
        {
            Signals.Add($"speakable-start:{turnId}:{operationId}");
            return ValueTask.FromResult(new ResponseControlTransition(IsCurrentTurn(turnId), true, 1));
        }

        public ValueTask PublishResponseCompletedAsync(
            long turnId,
            long operationId,
            string text,
            CancellationToken cancellationToken)
        {
            Signals.Add($"response-complete:{turnId}:{operationId}");
            ResponseCompletions.Add((turnId, operationId, text));
            return ValueTask.CompletedTask;
        }

        public ValueTask PublishPlaybackCompletedAsync(
            long turnId,
            long operationId,
            string text,
            TimeSpan elapsed,
            CancellationToken cancellationToken)
        {
            Signals.Add($"playback-complete:{turnId}:{operationId}");
            PlaybackCompletions.Add((turnId, operationId, text));
            return ValueTask.CompletedTask;
        }

        public ValueTask PublishResponseTerminatedAsync(
            long turnId,
            long operationId,
            ResponseOperationOutcomeKind outcome,
            string? failure,
            CancellationToken cancellationToken)
        {
            Signals.Add($"terminal:{turnId}:{operationId}:{outcome}");
            TerminalOutcomes.Add((turnId, operationId, outcome));
            return ValueTask.CompletedTask;
        }
    }

    private sealed class InMemoryToolRuntime(
        ToolCatalogSnapshot snapshot,
        ToolApprovalDecision decision,
        AssistantToolResult result) : IAssistantToolRuntime
    {
        public int CatalogCalls { get; private set; }
        public int ApprovalCalls { get; private set; }
        public int ExecutionCalls { get; private set; }

        public Task StartAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        public Task StopAsync() => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;

        public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
            ToolRequestContext context,
            CancellationToken cancellationToken)
        {
            CatalogCalls++;
            return ValueTask.FromResult(snapshot);
        }

        public ValueTask<ToolApprovalDecision> AuthorizeAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            ApprovalCalls++;
            return ValueTask.FromResult(decision);
        }

        public ValueTask<AssistantToolResult> ExecuteAsync(
            AssistantToolCall call,
            CancellationToken cancellationToken)
        {
            ExecutionCalls++;
            return ValueTask.FromResult(result);
        }
    }
}
