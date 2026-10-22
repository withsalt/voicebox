using System.Security.Cryptography;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task GoldenTrace_NormalTurn_IsExactAndStableAcrossReplay()
    {
        AssistantScenarioTrace first = await CaptureNormalTurnTraceAsync();
        string actual = GoldenTraceStore.Serialize(first);
        for (int replay = 0; replay < 10; replay++)
        {
            AssistantScenarioTrace next = await CaptureNormalTurnTraceAsync();
            Assert.Equal(actual, GoldenTraceStore.Serialize(next));
        }

        AssistantScenarioTrace expected = GoldenTraceStore.Read<AssistantScenarioTrace>("normal-turn.json");
        Assert.Equal(GoldenTraceStore.Serialize(expected), actual);
    }

    [Fact]
    public async Task KernelSemanticProjection_NormalTurn_MatchesTheGoldenProjection()
    {
        var probe = new VoiceAssistantBenchmarkProbe();
        AssistantScenarioTrace trace = await CaptureNormalTurnTraceAsync(probe);
        KernelSemanticProjection actual = ProjectNormalTurn(trace, probe.GetKernelDecisions());
        KernelSemanticProjection expected = GoldenTraceStore.Read<KernelSemanticProjection>("kernel-semantic-baseline.json");

        Assert.Equal(GoldenTraceStore.Serialize(expected), GoldenTraceStore.Serialize(actual));
    }

    [Fact]
    public async Task CompositionReference_BasicSplitAudioQa_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureBasicSplitAudioQaTraceAsync();
        const string relativePath = "CompositionReferenceTraces/basic-split-audio-qa.json";
        if (string.Equals(
                Environment.GetEnvironmentVariable("VOICEBOX_REGENERATE_GOLDENS"),
                "1",
                StringComparison.Ordinal))
        {
            string root = FindRepositoryRoot();
            string path = Path.Combine(
                root,
                "src",
                "Tests",
                "VoiceBox.Assistant.Tests",
                "GoldenTraces",
                relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, GoldenTraceStore.Serialize(actual));
            return;
        }

        AssistantScenarioTrace expected = GoldenTraceStore.Read<AssistantScenarioTrace>(relativePath);
        Assert.Equal(GoldenTraceStore.Serialize(expected), GoldenTraceStore.Serialize(actual));
    }

    [Fact]
    public async Task GoldenTrace_EmptyAsr_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureEmptyAsrTraceAsync();
        await AssertOrRegenerateTraceAsync("empty-asr.json", actual);
    }

    [Fact]
    public async Task GoldenTrace_WakeupTurn_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureWakeupTurnTraceAsync();
        await AssertOrRegenerateTraceAsync("wakeup-turn.json", actual);
    }

    [Fact]
    public async Task GoldenTrace_LlmFault_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureFaultTraceAsync(
            "llm-fault",
            services => services.Llm.StreamException = new InvalidOperationException("test-llm-fault"));
        await AssertOrRegenerateTraceAsync("llm-fault.json", actual);
    }

    [Fact]
    public async Task GoldenTrace_TtsFault_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureFaultTraceAsync(
            "tts-fault",
            services => services.Tts.StreamException = new InvalidOperationException("test-tts-fault"));
        await AssertOrRegenerateTraceAsync("tts-fault.json", actual);
    }

    private static async Task<AssistantScenarioTrace> CaptureNormalTurnTraceAsync(
        VoiceAssistantBenchmarkProbe? benchmarkProbe = null)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant(benchmarkProbe: benchmarkProbe);
        var events = new List<ScenarioEvent>();
        var reply = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        void AddEvent(ScenarioEvent value)
        {
            lock (events)
                events.Add(value with { PublicationSeq = events.Count + 1 });
        }

        assistant.StateChanged += (_, value) => AddEvent(new ScenarioEvent(
            0,
            "StateChanged",
            value.From.ToString(),
            value.To.ToString()));
        assistant.UserSpoke += (_, value) => AddEvent(new ScenarioEvent(
            0,
            "UserSpoke",
            Text: value.Text));
        assistant.AssistantText += (_, value) => AddEvent(new ScenarioEvent(
            0,
            "AssistantText",
            Text: value.Delta,
            IsThinking: value.IsThinking));
        assistant.AssistantResponseGenerated += (_, value) => AddEvent(new ScenarioEvent(
            0,
            "AssistantResponseGenerated",
            Text: value.Text));
        assistant.AssistantReplied += (_, value) =>
        {
            AddEvent(new ScenarioEvent(0, "AssistantReplied", Text: value.Text));
            reply.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await reply.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        ChatMessage[] prompt = Assert.Single(services.Llm.Prompts);
        float[] fed = Assert.Single(services.Asr.FedAudio["session-1"]);
        return new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            "normal-turn",
            "ExactTrace",
            [
                new ScenarioInput(1, "runtime.start", "full-stack-duplex"),
                new ScenarioInput(2, "vad.speech.started", "source=manual"),
                new ScenarioInput(3, "vad.speaking", "samples=3"),
                new ScenarioInput(4, "vad.speech.ended", "durationMs=120"),
                new ScenarioInput(5, "runtime.stop", "caller=host"),
            ],
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            [
                new ScenarioProviderCall(1, "audio.start", $"count={services.Audio.StartCount}"),
                new ScenarioProviderCall(2, "asr.feed", $"samples={fed.Length};sha256={HashFloats(fed)}"),
                new ScenarioProviderCall(3, "asr.finish", $"count={services.Asr.FinishCount}"),
                new ScenarioProviderCall(4, "llm.stream", $"messages={prompt.Length};roles={string.Join(',', prompt.Select(static value => value.Role))}"),
                new ScenarioProviderCall(5, "tts.synthesize", $"text={Assert.Single(services.Tts.SynthesizedTexts)}"),
                new ScenarioProviderCall(6, "playback.enqueue", $"count={services.Audio.EnqueueCalls};samples={services.Audio.AcceptedSamples}"),
                new ScenarioProviderCall(7, "audio.stop", $"count={services.Audio.StopCount}"),
            ],
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));
    }

    private static async Task<AssistantScenarioTrace> CaptureBasicSplitAudioQaTraceAsync()
    {
        var services = TestServices.Create();
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        services.Tts.StreamCompleted = playback.CompletePlayback;
        services.Asr.CompleteStart("split-session-1");
        var options = new VoiceAssistantOptions
        {
            MinSentenceChars = 1,
            PlaybackPrerollMs = 0,
            PlaybackTailBufferMs = 0,
            Listening = { MinDurationMs = 0, MinRms = 0 },
        };
        using VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .Build();

        var events = new List<ScenarioEvent>();
        var reply = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        void Add(ScenarioEvent value)
        {
            lock (events)
                events.Add(value with { PublicationSeq = events.Count + 1 });
        }

        assistant.StateChanged += (_, value) => Add(new ScenarioEvent(
            0, "StateChanged", value.From.ToString(), value.To.ToString()));
        assistant.AssistantReplied += (_, value) =>
        {
            Add(new ScenarioEvent(0, "AssistantReplied", Text: value.Text));
            reply.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        try
        {
            await reply.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        catch (TimeoutException exception)
        {
            throw new Xunit.Sdk.XunitException(
                $"Split reference timed out: state={assistant.State}; " +
                $"asrFinish={services.Asr.FinishCount}; prompts={services.Llm.Prompts.Count}; " +
                $"tts={services.Tts.SynthesizedTexts.Count}; record={record.State}; playback={playback.State}; " +
                $"isPlaying={playback.IsPlaying}; queue={playback.CurrentQueueSize}.",
                exception);
        }
        await assistant.StopAsync();

        return new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            "basic-split-audio-qa",
            "ExactTrace",
            [
                new ScenarioInput(1, "runtime.start", "BasicSplitAudioQa"),
                new ScenarioInput(2, "vad.speech.started", "source=manual"),
                new ScenarioInput(3, "vad.speaking", "samples=3"),
                new ScenarioInput(4, "vad.speech.ended", "durationMs=120"),
                new ScenarioInput(5, "runtime.stop", "caller=host"),
            ],
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            [
                new ScenarioProviderCall(1, "audio.playback.start", $"count={playback.StartCount}"),
                new ScenarioProviderCall(2, "audio.capture.start", $"count={record.StartCount}"),
                new ScenarioProviderCall(3, "asr.finish", $"count={services.Asr.FinishCount}"),
                new ScenarioProviderCall(4, "audio.capture.stop", $"count={record.StopCount}"),
                new ScenarioProviderCall(5, "audio.playback.stop", $"count={playback.StopCount}"),
            ],
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));
    }

    private static async Task<AssistantScenarioTrace> CaptureEmptyAsrTraceAsync()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("empty-session");
        services.Asr.SetResult("empty-session", "   ");
        using VoiceAssistant assistant = services.CreateAssistant();
        var events = new List<ScenarioEvent>();
        void Add(ScenarioEvent value)
        {
            lock (events)
                events.Add(value with { PublicationSeq = events.Count + 1 });
        }
        assistant.StateChanged += (_, value) => Add(new ScenarioEvent(
            0, "StateChanged", value.From.ToString(), value.To.ToString()));
        assistant.UserSpoke += (_, value) => Add(new ScenarioEvent(0, "UserSpoke", Text: value.Text));

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        await assistant.StopAsync();

        return new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            "empty-asr",
            "ExactTrace",
            [
                new ScenarioInput(1, "runtime.start", "full-stack-duplex"),
                new ScenarioInput(2, "vad.speech.started", "source=manual"),
                new ScenarioInput(3, "vad.speaking", "samples=3"),
                new ScenarioInput(4, "vad.speech.ended", "durationMs=120"),
                new ScenarioInput(5, "runtime.stop", "caller=host"),
            ],
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            [
                new ScenarioProviderCall(1, "asr.finish", $"count={services.Asr.FinishCount}"),
                new ScenarioProviderCall(2, "llm.stream", $"count={services.Llm.Prompts.Count}"),
                new ScenarioProviderCall(3, "tts.synthesize", $"count={services.Tts.SynthesizedTexts.Count}"),
            ],
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));
    }

    private static async Task<AssistantScenarioTrace> CaptureWakeupTurnTraceAsync()
    {
        var services = TestServices.Create();
        using VoiceAssistant assistant = services.CreateAssistant(enableWakeWord: true);
        var events = new List<ScenarioEvent>();
        var wakeup = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        void Add(ScenarioEvent value)
        {
            lock (events)
                events.Add(value with { PublicationSeq = events.Count + 1 });
        }

        assistant.WakeModeChanged += (_, value) => Add(new ScenarioEvent(
            0, "WakeModeChanged", value.From.ToString(), value.To.ToString()));
        assistant.WakeupDetected += (_, value) =>
        {
            Add(new ScenarioEvent(0, "WakeupDetected", Text: value.Wakeup.Keyword));
            wakeup.TrySetResult();
        };

        await assistant.StartAsync();
        services.Audio.Emit(new float[1600]);
        services.Wakeup.EmitWakeup(startTimeMs: 0, durationMs: 10);
        services.Audio.Emit(new float[1600]);
        await wakeup.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        return new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            "wakeup-turn",
            "ExactTrace",
            [
                new ScenarioInput(1, "runtime.start", "full-stack-duplex+wakeup"),
                new ScenarioInput(2, "capture.frame", "samples=1600"),
                new ScenarioInput(3, "kws.wakeup", "keyword=小爱同学"),
                new ScenarioInput(4, "capture.frame", "samples=1600"),
                new ScenarioInput(5, "runtime.stop", "caller=host"),
            ],
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            [
                new ScenarioProviderCall(1, "wakeup.start", $"count={services.Wakeup.StartCount}"),
                new ScenarioProviderCall(2, "wakeup.feed", $"count={services.Wakeup.FedAudio.Count}"),
                new ScenarioProviderCall(3, "wakeup.stop", $"count={services.Wakeup.StopCount}"),
            ],
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));
    }

    private static async Task<AssistantScenarioTrace> CaptureFaultTraceAsync(
        string scenarioId,
        Action<TestServices> configure)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("fault-session");
        configure(services);
        using VoiceAssistant assistant = services.CreateAssistant();
        var events = new List<ScenarioEvent>();
        void Add(ScenarioEvent value)
        {
            lock (events)
                events.Add(value with { PublicationSeq = events.Count + 1 });
        }
        assistant.StateChanged += (_, value) => Add(new ScenarioEvent(
            0, "StateChanged", value.From.ToString(), value.To.ToString()));
        assistant.Faulted += (_, value) => Add(new ScenarioEvent(
            0, "Faulted", Text: value.Component));

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        await assistant.StopAsync();

        return new AssistantScenarioTrace(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            scenarioId,
            "ExactTrace",
            [
                new ScenarioInput(1, "runtime.start", "full-stack-duplex"),
                new ScenarioInput(2, "vad.speech.started", "source=manual"),
                new ScenarioInput(3, "vad.speaking", "samples=3"),
                new ScenarioInput(4, "vad.speech.ended", "durationMs=120"),
                new ScenarioInput(5, "runtime.stop", "caller=host"),
            ],
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            [
                new ScenarioProviderCall(1, "asr.finish", $"count={services.Asr.FinishCount}"),
                new ScenarioProviderCall(2, "llm.stream", $"count={services.Llm.Prompts.Count}"),
                new ScenarioProviderCall(3, "tts.synthesize", $"count={services.Tts.SynthesizedTexts.Count}"),
            ],
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));
    }

    private static async Task AssertOrRegenerateTraceAsync(
        string relativePath,
        AssistantScenarioTrace actual)
    {
        if (string.Equals(
                Environment.GetEnvironmentVariable("VOICEBOX_REGENERATE_GOLDENS"),
                "1",
                StringComparison.Ordinal))
        {
            string root = FindRepositoryRoot();
            string path = Path.Combine(
                root,
                "src",
                "Tests",
                "VoiceBox.Assistant.Tests",
                "GoldenTraces",
                relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await File.WriteAllTextAsync(path, GoldenTraceStore.Serialize(actual));
            return;
        }

        AssistantScenarioTrace expected = GoldenTraceStore.Read<AssistantScenarioTrace>(relativePath);
        Assert.Equal(GoldenTraceStore.Serialize(expected), GoldenTraceStore.Serialize(actual));
    }

    private static KernelSemanticProjection ProjectNormalTurn(
        AssistantScenarioTrace trace,
        IReadOnlyList<KernelDecisionObservation> observations)
    {
        Assert.Equal("normal-turn", trace.ScenarioId);
        Assert.Collection(
            trace.PublicEvents,
            value => Assert.Equal(("Idle", "Listening"), (value.From, value.To)),
            value => Assert.Equal(("Listening", "Recognizing"), (value.From, value.To)),
            value => Assert.Equal("UserSpoke", value.Type),
            value => Assert.Equal(("Recognizing", "Thinking"), (value.From, value.To)),
            value => Assert.Equal(("Thinking", "Speaking"), (value.From, value.To)),
            value => Assert.Equal("AssistantText", value.Type),
            value => Assert.Equal("AssistantResponseGenerated", value.Type),
            value => Assert.Equal("AssistantReplied", value.Type),
            value => Assert.Equal(("Speaking", "Idle"), (value.From, value.To)),
            value => Assert.Equal(("Idle", "Stopped"), (value.From, value.To)));

        TurnSignalKind[] projectedKinds =
        [
            TurnSignalKind.SpeechStarted,
            TurnSignalKind.RecognitionSessionReady,
            TurnSignalKind.RecognitionCompleted,
            TurnSignalKind.PlaybackStarted,
            TurnSignalKind.ResponseGenerated,
            TurnSignalKind.PlaybackCompleted,
        ];
        KernelDecisionObservation[] projected = observations
            .Where(static observation => observation.Accepted)
            .Where(observation => projectedKinds.Contains(observation.Signal.Kind))
            .ToArray();
        Assert.Equal(projectedKinds, projected.Select(static observation => observation.Signal.Kind));

        KernelDecisionObservation[] modulePrivate = observations
            .Where(static observation => observation.Accepted)
            .Where(observation => !projectedKinds.Contains(observation.Signal.Kind))
            .ToArray();
        KernelDecisionObservation speechEnded = Assert.Single(modulePrivate);
        Assert.Equal(TurnSignalKind.SpeechEnded, speechEnded.Signal.Kind);
        Assert.IsType<FinishRecognitionEffect>(Assert.Single(speechEnded.Effects));

        return new KernelSemanticProjection(
            "voicebox.assistant/kernel-semantic-trace/v1",
            "v1",
            trace.ScenarioId,
            trace.OracleKind,
            projected.Select(ProjectKernelStep).ToArray());
    }

    private static KernelSemanticStep ProjectKernelStep(
        KernelDecisionObservation observation,
        int index)
        => new(
            index + 1,
            SignalDiscriminant(observation.Signal.Kind),
            observation.Previous.ConversationState.ToString(),
            observation.Next.ConversationState.ToString(),
            observation.Effects.Select(ProjectKernelEffect).ToArray());

    private static KernelSemanticEffect ProjectKernelEffect(AssistantEffect effect, int index)
        => effect switch
        {
            PublishStateChangedEffect value => new(
                index,
                "effect.public-event.publish.v1",
                $"StateChanged:{value.From}->{value.To}"),
            PublishUserSpokeEffect value => new(
                index,
                "effect.public-event.publish.v1",
                $"UserSpoke:{value.Text}"),
            PublishAssistantResponseGeneratedEffect value => new(
                index,
                "effect.public-event.publish.v1",
                $"AssistantResponseGenerated:{value.Text}"),
            PublishAssistantRepliedEffect value => new(
                index,
                "effect.public-event.publish.v1",
                $"AssistantReplied:{value.Text}"),
            PrepareRecognitionEffect => new(
                index,
                "effect.recognition.prepare.v1",
                "turnId=T1;operationId=O1"),
            GenerateResponseEffect => new(
                index,
                "effect.response.generate.v1",
                "turnId=T1;operationId=O2"),
            CommitHistoryEffect => new(index, "effect.history.commit.v1", "turnId=T1"),
            CompleteTurnEffect => new(index, "effect.turn.complete.v1", "turnId=T1"),
            _ => throw new Xunit.Sdk.XunitException(
                $"Unexpected effect in canonical normal-turn projection: {effect.GetType().Name}"),
        };

    private static string SignalDiscriminant(TurnSignalKind kind)
        => kind switch
        {
            TurnSignalKind.SpeechStarted => "turn.speech.started.v1",
            TurnSignalKind.RecognitionSessionReady => "turn.recognition.session-ready.v1",
            TurnSignalKind.RecognitionCompleted => "turn.recognition.completed.v1",
            TurnSignalKind.PlaybackStarted => "turn.playback.started.v1",
            TurnSignalKind.ResponseGenerated => "turn.response.generated.v1",
            TurnSignalKind.PlaybackCompleted => "turn.playback.completed.v1",
            _ => throw new Xunit.Sdk.XunitException($"Unexpected projected signal: {kind}"),
        };

    private static string HashFloats(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

}
