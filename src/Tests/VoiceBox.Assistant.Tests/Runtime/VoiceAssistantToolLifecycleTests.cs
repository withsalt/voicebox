using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Core.Services.LLMService;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task LocalTools_DisposeWithAssistantOnlyOnce(bool start, bool stop)
    {
        var tool = new ToolLifetimeProbe();
        using var assistant = CreateToolLifetimeAssistant(TestServices.Create(), [tool]);
        if (start)
            await assistant.StartAsync();
        if (stop)
            await assistant.StopAsync();

        Assert.Equal(0, tool.DisposeCount);
        assistant.Dispose();
        assistant.Dispose();

        Assert.Equal(1, tool.DisposeCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LocalTools_DisposeFailureDoesNotPreventOtherToolsFromBeingReleased(bool start)
    {
        var first = new ToolLifetimeProbe("first");
        var throwing = new ToolLifetimeProbe("throwing", throwOnDispose: true);
        var last = new ToolLifetimeProbe("last");
        using var assistant = CreateToolLifetimeAssistant(TestServices.Create(), [first, throwing, last]);
        if (start)
            await assistant.StartAsync();

        assistant.Dispose();
        assistant.Dispose();

        Assert.Equal(1, first.DisposeCount);
        Assert.Equal(1, throwing.DisposeCount);
        Assert.Equal(1, last.DisposeCount);
    }

    [Fact]
    public async Task LocalTools_StartupFailureStillAllowsAllToolsToBeReleased()
    {
        var first = new ToolLifetimeProbe("first");
        var failing = new ToolLifetimeProbe("failing");
        var last = new ToolLifetimeProbe("last");
        using var assistant = CreateToolLifetimeAssistant(TestServices.Create(), [first, failing, last]);
        failing.FailDiscovery = true;

        await Assert.ThrowsAsync<InvalidOperationException>(() => assistant.StartAsync());
        Assert.Equal(0, first.DisposeCount);
        Assert.Equal(0, failing.DisposeCount);
        Assert.Equal(0, last.DisposeCount);
        assistant.Dispose();

        Assert.Equal(1, first.DisposeCount);
        Assert.Equal(1, failing.DisposeCount);
        Assert.Equal(1, last.DisposeCount);
    }

    [Theory]
    [InlineData(AssistantToolExecutionPolicy.ForegroundOnly)]
    [InlineData(AssistantToolExecutionPolicy.BackgroundOnly)]
    public async Task LocalTools_DisposeCancelsAndWaitsForActiveCalls(AssistantToolExecutionPolicy policy)
    {
        var services = TestServices.Create();
        var tool = new ToolLifetimeProbe(policy: policy, blockExecution: true);
        using var assistant = CreateToolLifetimeAssistant(services, [tool], new BackgroundWeatherLlm());
        await assistant.StartAsync();
        StartWeatherQuestion(services);
        await tool.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Task disposal = Task.Run(assistant.Dispose);
        try
        {
            await tool.CancellationRequested.Task.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.False(disposal.IsCompleted);
            Assert.Equal(0, tool.DisposeCount);
        }
        finally
        {
            tool.AllowExecutionExit.TrySetResult();
            await disposal.WaitAsync(TimeSpan.FromSeconds(2));
        }

        Assert.Equal(1, tool.DisposeCount);
        Assert.False(tool.DisposedDuringExecution);
    }

    private static VoiceAssistant CreateToolLifetimeAssistant(
        TestServices services,
        IAssistantTool[] tools,
        IToolCallingStreamLLMService? llm = null)
        => VoiceAssistant.CreateBuilder()
            .UseOptions(new VoiceAssistantOptions
            {
                MinSentenceChars = 1,
                PlaybackPrerollMs = 0,
                PlaybackTailBufferMs = 0,
                Listening = { MinDurationMs = 0, MinRms = 0 },
            })
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(llm ?? new StructuredTestLlm())
            .UseMcpTools(builder =>
            {
                foreach (IAssistantTool tool in tools)
                    builder.AddTool(tool);
            })
            .UseSpeechSynthesis(services.Tts)
            .Build();

    private sealed class ToolLifetimeProbe(
        string name = "weather",
        AssistantToolExecutionPolicy policy = AssistantToolExecutionPolicy.ForegroundOnly,
        bool blockExecution = false,
        bool throwOnDispose = false) : IAssistantTool
    {
        private readonly AssistantToolDescriptor _descriptor = new(
            name, name, "生命周期测试工具", "{\"type\":\"object\"}",
            executionPolicy: policy, executionPrompt: "正在查询，请稍后。");
        private int _disposeCount;
        private int _activeCalls;

        public int DisposeCount => Volatile.Read(ref _disposeCount);
        public bool DisposedDuringExecution { get; private set; }
        public bool FailDiscovery { get; set; }
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource CancellationRequested { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource AllowExecutionExit { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public AssistantToolDescriptor Descriptor => FailDiscovery
            ? throw new InvalidOperationException("Tool discovery failed.")
            : _descriptor;

        public async ValueTask<AssistantToolExecutionResult> ExecuteAsync(
            AssistantToolInvocation invocation, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _activeCalls);
            try
            {
                Started.TrySetResult();
                if (blockExecution)
                {
                    try
                    {
                        await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        CancellationRequested.TrySetResult();
                        await AllowExecutionExit.Task.ConfigureAwait(false);
                        throw;
                    }
                }
                return AssistantToolExecutionResult.Success("已完成。");
            }
            finally
            {
                Interlocked.Decrement(ref _activeCalls);
            }
        }

        public void Dispose()
        {
            DisposedDuringExecution |= Volatile.Read(ref _activeCalls) != 0;
            Interlocked.Increment(ref _disposeCount);
            if (throwOnDispose)
                throw new InvalidOperationException("Tool disposal failed.");
        }
    }
}
