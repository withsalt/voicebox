using System.Reflection;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BackgroundResult_PlaybackCanBeInterrupted(bool completeDuringAcknowledgement)
    {
        var services = TestServices.Create();
        var weather = new DelayedBackgroundWeatherTool();
        var llm = new BackgroundWeatherLlm();
        if (completeDuringAcknowledgement)
            services.Tts.StreamCompleted = null;
        using var assistant = CreateBackgroundAssistant(services, weather, llm);
        var resultPublished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantText += (_, e) =>
        {
            if (e.Delta == BackgroundWeatherLlm.WeatherText)
                resultPublished.TrySetResult();
        };
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.BargeInValidation += (_, e) =>
        {
            if (e.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        await assistant.StartAsync();
        StartWeatherQuestion(services);
        await weather.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        if (!completeDuringAcknowledgement)
            await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        var resultGate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockAfterFirstChunk = resultGate;
        weather.Release.TrySetResult();
        await resultPublished.Task.WaitAsync(TimeSpan.FromSeconds(2));
        if (completeDuringAcknowledgement)
        {
            Assert.DoesNotContain(BackgroundWeatherLlm.WeatherText, services.Tts.SynthesizedTexts);
            Assert.Equal(AssistantState.Speaking, assistant.State);
            services.Tts.StreamCompleted = services.Audio.CompletePlayback;
            services.Audio.CompletePlayback();
        }
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Contains(BackgroundWeatherLlm.WeatherText) &&
            services.Audio.IsPlaying);
        Assert.Equal(AssistantState.Speaking, assistant.State);

        services.Tts.BlockAfterFirstChunk = null;
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "给我讲个故事吧");
        TaskCompletionSource recognitionGate = services.Asr.BlockFinish("session-2");
        int previousPushCount = services.PlaybackDetector.PushCount;
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));

        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.True(services.PlaybackDetector.PushCount > previousPushCount);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.False(resultGate.Task.IsCompleted);
        Assert.Equal(AssistantState.Listening, assistant.State);

        recognitionGate.TrySetResult();
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Contains(BackgroundWeatherLlm.StoryText) &&
            assistant.State == AssistantState.Idle);
        Assert.Equal(1, services.Audio.ClearQueueCount);
        Assert.Single(services.Tts.SynthesizedTexts, text => text == BackgroundWeatherLlm.WeatherText);
        await AssertNoBackgroundTurnResourcesAsync(assistant);
    }

    [Fact]
    public async Task BackgroundResult_WaitsForUserTurnThenCompletesAndReleasesResources()
    {
        var services = TestServices.Create();
        var weather = new DelayedBackgroundWeatherTool();
        var llm = new BackgroundWeatherLlm();
        using var assistant = CreateBackgroundAssistant(services, weather, llm);
        var resultPublished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantText += (_, e) =>
        {
            if (e.Delta == BackgroundWeatherLlm.WeatherText)
                resultPublished.TrySetResult();
        };
        await assistant.StartAsync();
        StartWeatherQuestion(services);
        await weather.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "给我讲个故事吧");
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        weather.Release.TrySetResult();
        await resultPublished.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.DoesNotContain(BackgroundWeatherLlm.WeatherText, services.Tts.SynthesizedTexts);

        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Contains(BackgroundWeatherLlm.WeatherText) &&
            assistant.State == AssistantState.Idle);
        Assert.Equal(
            [DelayedBackgroundWeatherTool.Prompt, BackgroundWeatherLlm.StoryText, BackgroundWeatherLlm.WeatherText],
            services.Tts.SynthesizedTexts);
        Assert.Equal(0, services.Audio.ClearQueueCount);
        await AssertNoBackgroundTurnResourcesAsync(assistant);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BackgroundResult_StopCancelsWaitingOrPlayingResult(bool waitForUserTurn)
    {
        var services = TestServices.Create();
        var weather = new DelayedBackgroundWeatherTool();
        using var assistant = CreateBackgroundAssistant(services, weather, new BackgroundWeatherLlm());
        var resultPublished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantText += (_, e) =>
        {
            if (e.Delta == BackgroundWeatherLlm.WeatherText)
                resultPublished.TrySetResult();
        };
        await assistant.StartAsync();
        StartWeatherQuestion(services);
        await weather.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        if (waitForUserTurn)
        {
            services.Asr.CompleteStart("session-2");
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        }
        else
        {
            services.Tts.BlockAfterFirstChunk = new(TaskCreationOptions.RunContinuationsAsynchronously);
        }
        weather.Release.TrySetResult();
        await resultPublished.Task.WaitAsync(TimeSpan.FromSeconds(2));
        if (!waitForUserTurn)
            await WaitUntilAsync(() => services.Audio.IsPlaying && assistant.State == AssistantState.Speaking);

        await assistant.StopAsync().WaitAsync(TimeSpan.FromSeconds(2));
        if (waitForUserTurn)
            Assert.DoesNotContain(BackgroundWeatherLlm.WeatherText, services.Tts.SynthesizedTexts);
        Assert.False(services.Audio.IsPlaying);
        if (!waitForUserTurn)
            await AssertNoBackgroundTurnResourcesAsync(assistant);
    }

    [Fact]
    public async Task BackgroundResult_SynthesisFailureReturnsToIdleAndAllowsNextQuestion()
    {
        var services = TestServices.Create();
        var weather = new DelayedBackgroundWeatherTool();
        using var assistant = CreateBackgroundAssistant(services, weather, new BackgroundWeatherLlm());
        await assistant.StartAsync();
        StartWeatherQuestion(services);
        await weather.Started.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        var returnedToIdle = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.StateChanged += (_, e) =>
        {
            if (e.To == AssistantState.Idle)
                returnedToIdle.TrySetResult();
        };
        services.Tts.StreamException = new InvalidOperationException("background synthesis failed");
        weather.Release.TrySetResult();
        await returnedToIdle.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await AssertNoBackgroundTurnResourcesAsync(assistant);

        services.Tts.StreamException = null;
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "给我讲个故事吧");
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Contains(BackgroundWeatherLlm.StoryText) &&
            assistant.State == AssistantState.Idle);
        await AssertNoBackgroundTurnResourcesAsync(assistant);
    }

    private static VoiceAssistant CreateBackgroundAssistant(
        TestServices services, DelayedBackgroundWeatherTool weather, BackgroundWeatherLlm llm)
    {
        var options = new VoiceAssistantOptions
        {
            MinSentenceChars = 1,
            PlaybackPrerollMs = 0,
            PlaybackTailBufferMs = 0,
            Listening = { MinDurationMs = 0, MinRms = 0 },
            BargeIn = { Enable = true, MinDurationMs = 0, MinRms = 0, PlaybackStartupGuardMs = 0 },
        };
        return VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(llm)
            .UseMcpTools(tools => tools.AddTool(weather))
            .UseSpeechSynthesis(services.Tts)
            .AddBargeIn(options.BargeIn, TestAcousticPathDelayEstimatorFactory.Instance)
            .BuildForBenchmark(TimeProvider.System, new VoiceAssistantBenchmarkProbe(),
                playbackBargeInDetectorFactory: new FakePlaybackBargeInDetectorFactory(services.PlaybackDetector));
    }

    private static void StartWeatherQuestion(TestServices services)
    {
        services.Asr.CompleteStart("session-1");
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
    }

    private static async Task AssertNoBackgroundTurnResourcesAsync(VoiceAssistant assistant)
    {
        var runtime = (AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(assistant)!;
        await WaitUntilAsync(() => runtime.ActiveTurnContextCount == 0);
    }

    private sealed class DelayedBackgroundWeatherTool : IAssistantTool
    {
        public const string Prompt = "正在查询请稍后。";
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public AssistantToolDescriptor Descriptor { get; } = new(
            "weather", "天气", "查询天气", "{\"type\":\"object\"}",
            executionPolicy: AssistantToolExecutionPolicy.BackgroundOnly, executionPrompt: Prompt);

        public async ValueTask<AssistantToolExecutionResult> ExecuteAsync(
            AssistantToolInvocation invocation, CancellationToken cancellationToken)
        {
            Started.TrySetResult();
            await Release.Task.WaitAsync(cancellationToken);
            return AssistantToolExecutionResult.Success("成都轻雾，22度。");
        }

        public void Dispose() { }
    }

    private sealed class BackgroundWeatherLlm : IToolCallingStreamLLMService
    {
        public const string WeatherText = "成都现在是22度的轻雾天气。";
        public const string StoryText = "从前有一座山。";
        private int _calls;
        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages, LlmCompleteOptions? options = null,
            CancellationToken ct = default) => throw new NotSupportedException();
        public IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages, LlmCompleteOptions? options = null,
            CancellationToken ct = default) => throw new NotSupportedException();

        public async IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
            LlmToolTurnRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            if (Interlocked.Increment(ref _calls) == 1)
                yield return new LlmToolCallEvent("weather-1", Assert.Single(request.Tools).Name, "{}");
            else
                yield return new LlmTurnTextDelta(
                    request.Messages[^1].Content.OfType<ToolResultContent>().Any() ? WeatherText : StoryText);
            yield return new LlmTurnCompletedEvent();
        }

        public void Dispose() { }
    }
}
