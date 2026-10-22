using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task GoldenTrace_TextOnlyModelInteraction_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureTextOnlyModelInteractionTraceAsync();
        await AssertOrRegenerateSpecTraceAsync("text-only-model-interaction.json", actual);
    }

    [Fact]
    public async Task GoldenTrace_ContinuousConversationTimeout_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureContinuousConversationTimeoutTraceAsync();
        await AssertOrRegenerateSpecTraceAsync("continuous-conversation-timeout.json", actual);
    }

    /// <summary>KWS disabled: the acoustic confirmation stops the reply before any ASR text; valid text then starts the new turn.</summary>
    [Fact]
    public async Task GoldenTrace_BargeInAccepted_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureBargeInValidationTraceAsync(
            "barge-in-accepted",
            validationText: "停一下",
            expectResponse: true);
        await AssertOrRegenerateSpecTraceAsync("barge-in-accepted.json", actual);
    }

    /// <summary>KWS disabled: the stop is identical; an empty whole-utterance result only means no response follows.</summary>
    [Fact]
    public async Task GoldenTrace_BargeInEmptyText_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureBargeInValidationTraceAsync(
            "barge-in-empty-text",
            validationText: string.Empty,
            expectResponse: false);
        await AssertOrRegenerateSpecTraceAsync("barge-in-empty-text.json", actual);
    }

    /// <summary>KWS enabled: another speaker's candidate is denied by the identity gate and never reaches the Kernel.</summary>
    [Fact]
    public async Task GoldenTrace_BargeInDeniedImpostor_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureBargeInDeniedImpostorTraceAsync();
        await AssertOrRegenerateSpecTraceAsync("barge-in-denied-impostor.json", actual);
    }

    [Fact]
    public async Task GoldenTrace_SpeechEndBeforeAsrReady_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureSpeechEndScheduleTraceAsync(asrReadyFirst: false);
        await AssertOrRegenerateSpecTraceAsync(
            "Schedules/speech-end-vs-asr-start/runtime-speech-end-first.json",
            actual);
    }

    [Fact]
    public async Task GoldenTrace_AsrReadyBeforeSpeechEnd_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureSpeechEndScheduleTraceAsync(asrReadyFirst: true);
        await AssertOrRegenerateSpecTraceAsync(
            "Schedules/speech-end-vs-asr-start/runtime-asr-ready-first.json",
            actual);
    }

    [Fact]
    public async Task GoldenTrace_BargeInBeforeResponseGenerated_IsExact()
    {
        AssistantScenarioTrace actual = await CaptureResponseBargeInScheduleTraceAsync();
        await AssertOrRegenerateSpecTraceAsync(
            "Schedules/response-complete-vs-barge-in/runtime-barge-in-before-response-generated.json",
            actual);
    }

    private static async Task<AssistantScenarioTrace> CaptureTextOnlyModelInteractionTraceAsync()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("text-session");
        services.Llm.ResponseText = "纯文本回答。";
        using VoiceAssistant assistant = services.CreateAssistant();
        List<ScenarioEvent> events = [];
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        AddPublicTurnEvents(assistant, events, replied);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        return CreateTrace(
            "text-only-model-interaction",
            [
                new(1, "runtime.start", "full-stack-duplex"),
                new(2, "vad.speech.started", "source=manual"),
                new(3, "vad.speaking", "samples=3"),
                new(4, "vad.speech.ended", "durationMs=120"),
                new(5, "runtime.stop", "caller=host"),
            ],
            events,
            [
                new(1, "audio.start", $"count={services.Audio.StartCount}"),
                new(2, "asr.finish", $"count={services.Asr.FinishCount}"),
                new(3, "llm.stream", $"count={services.Llm.Prompts.Count}"),
                new(4, "tts.synthesize", $"text={Assert.Single(services.Tts.SynthesizedTexts)}"),
                new(5, "audio.stop", $"count={services.Audio.StopCount}"),
            ],
            assistant);
    }

    private static async Task<AssistantScenarioTrace> CaptureContinuousConversationTimeoutTraceAsync()
    {
        var services = TestServices.Create();
        var time = new ManualTimerTimeProvider();
        using VoiceAssistant assistant = services.CreateAssistant(
            enableWakeWord: true,
            continuousConversationIdleTimeout: TimeSpan.FromMilliseconds(20),
            timeProvider: time);
        List<ScenarioEvent> events = [];
        AddWakeEvents(assistant, events);

        await assistant.StartAsync();
        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup(startTimeMs: 0, durationMs: 10);
        services.Audio.Emit([0.2f]);
        await WaitUntilAsync(() => assistant.WakeMode == AssistantWakeMode.ContinuousConversation);
        time.Advance(TimeSpan.FromMilliseconds(25));
        services.Audio.Emit([0.3f]);
        await WaitUntilAsync(() => assistant.WakeMode == AssistantWakeMode.WaitingForWakeWord);
        await assistant.StopAsync();

        return CreateTrace(
            "continuous-conversation-timeout",
            [
                new(1, "runtime.start", "full-stack-duplex+wakeup"),
                new(2, "capture.frame", "samples=1"),
                new(3, "kws.wakeup", "keyword=小爱同学"),
                new(4, "capture.frame", "samples=1"),
                new(5, "time.advance", "25ms"),
                new(6, "capture.frame", "samples=1"),
                new(7, "runtime.stop", "caller=host"),
            ],
            events,
            [
                new(1, "wakeup.start", $"count={services.Wakeup.StartCount}"),
                new(2, "wakeup.feed", $"count={services.Wakeup.FedAudio.Count}"),
                new(3, "wakeup.stop", $"count={services.Wakeup.StopCount}"),
            ],
            assistant);
    }

    private static async Task<AssistantScenarioTrace> CaptureBargeInValidationTraceAsync(
        string scenarioId,
        string validationText,
        bool expectResponse)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", validationText);
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        using VoiceAssistant assistant = services.CreateAssistant(
            enableBargeIn: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        List<ScenarioEvent> events = [];
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var accepted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        AddPublicTurnEvents(assistant, events, replied);
        assistant.BargeInValidation += (_, value) =>
        {
            AddEvent(events, new ScenarioEvent(
                0,
                "BargeInValidation",
                Text: value.Kind == BargeInValidationKind.Heard
                    ? value.Kind.ToString()
                    : $"{value.Kind}:{value.Reason}"));
            if (value.Kind == BargeInValidationKind.Accepted)
                accepted.TrySetResult();
        };

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        // The acoustic confirmation is the whole authorization with KWS disabled: the owner-scoped
        // stop is issued (and Accepted published) while the whole-utterance ASR result is still blocked.
        await accepted.Task.WaitAsync(TimeSpan.FromSeconds(2));
        int clearsBeforeAsrText = services.Audio.ClearQueueCount;
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        validationFinish.TrySetResult();

        if (expectResponse)
        {
            services.Tts.BlockAfterFirstChunk = null;
            await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        await assistant.StopAsync();
        return CreateTrace(
            scenarioId,
            [
                new(1, "runtime.start", "full-stack-duplex"),
                new(2, "vad.speech.started", "response-turn"),
                new(3, "vad.speech.ended", "response-turn"),
                new(4, "vad.speech", "barge-in;kws=off"),
                new(5, "barge-in.validation", "accepted:audio-confirmed;before-asr-text"),
                new(6, "asr.finish", $"text={validationText}"),
                new(7, "runtime.stop", "caller=host"),
            ],
            events,
            [
                new(1, "playback.clear", $"count={clearsBeforeAsrText};before-asr-text"),
                new(2, "asr.finish", $"count={services.Asr.FinishCount}"),
                new(3, "playback.clear", $"count={services.Audio.ClearQueueCount}"),
                new(4, "playback.gain", $"gain={services.Audio.PlaybackGain:0.##};changes={services.Audio.PlaybackGainChanges.Count}"),
                new(5, "llm.stream", $"count={services.Llm.Prompts.Count}"),
            ],
            assistant);
    }

    private static async Task<AssistantScenarioTrace> CaptureBargeInDeniedImpostorTraceAsync()
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        try
        {
            services.Asr.CompleteStart("session-1");
            var firstTtsBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Tts.BlockAfterFirstChunk = firstTtsBlock;
            using VoiceAssistant assistant = services.CreateAssistant(
                enableBargeIn: true,
                enableWakeWord: true,
                useRealSpeaker: true,
                configure: options =>
                {
                    options.Speaker.ProfilePath = profilePath;
                    options.PlaybackPrerollMs = 0;
                    options.BargeIn.MinDurationMs = 120;
                    options.BargeIn.PlaybackStartupGuardMs = 0;
                });
            List<ScenarioEvent> events = [];
            var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            AddPublicTurnEvents(assistant, events, replied);
            assistant.BargeInValidation += (_, value) => AddEvent(events, new ScenarioEvent(
                0,
                "BargeInValidation",
                Text: $"{value.Kind}:{value.Reason}"));

            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);

            // Another speaker talks over the reply: the candidate is denied by the identity gate and
            // produces no public event and no playback side effect.
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(1);
            float[] impostor = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(impostor);
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(impostor);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(800));
            await WaitUntilAsync(() => services.Speaker.FeedCount > 0);
            await Task.Delay(150);
            int clearsAfterImpostor = services.Audio.ClearQueueCount;

            firstTtsBlock.TrySetResult();
            await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
            await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
            await assistant.StopAsync();

            return CreateTrace(
                "barge-in-denied-impostor",
                [
                    new(1, "runtime.start", "full-stack-duplex+wakeup+speaker"),
                    new(2, "kws.wakeup", "owner"),
                    new(3, "vad.speech.started", "response-turn"),
                    new(4, "vad.speech.ended", "response-turn"),
                    new(5, "vad.speech", "impostor;kws=on"),
                    new(6, "runtime.stop", "caller=host"),
                ],
                events,
                [
                    new(1, "speaker.extract", $"observed={services.Speaker.FeedCount > 0}"),
                    new(2, "playback.clear", $"count={clearsAfterImpostor};after-impostor"),
                    new(3, "playback.gain", $"gain={services.Audio.PlaybackGain:0.##};changes={services.Audio.PlaybackGainChanges.Count}"),
                    new(4, "llm.stream", $"count={services.Llm.Prompts.Count}"),
                ],
                assistant);
        }
        finally
        {
            try { File.Delete(profilePath); } catch { }
        }
    }

    private static async Task<AssistantScenarioTrace> CaptureSpeechEndScheduleTraceAsync(bool asrReadyFirst)
    {
        var services = TestServices.Create();
        if (asrReadyFirst)
            services.Asr.CompleteStart("session-1");
        using VoiceAssistant assistant = services.CreateAssistant();
        List<ScenarioEvent> events = [];
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        AddPublicTurnEvents(assistant, events, replied);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        if (!asrReadyFirst)
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        else
            await WaitUntilAsync(() => services.Asr.FedAudio.ContainsKey("session-1"));
        if (!asrReadyFirst)
            services.Asr.CompleteStart("session-1");
        else
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await assistant.StopAsync();

        AssistantScenarioTrace trace = CreateTrace(
            "speech-end-vs-asr-start",
            asrReadyFirst
                ? [
                    new(1, "vad.speech.started", "turn=T1"),
                    new(2, "asr.session.ready", "session-1"),
                    new(3, "vad.speech.ended", "after session ready"),
                ]
                : [
                    new(1, "vad.speech.started", "turn=T1"),
                    new(2, "vad.speech.ended", "before session ready"),
                    new(3, "asr.session.ready", "session-1"),
                ],
            events,
            [new(1, "asr.finish", $"count={services.Asr.FinishCount}")],
            assistant);
        return trace with
        {
            ScheduleId = asrReadyFirst
                ? "runtime-asr-ready-first"
                : "runtime-speech-end-first",
        };
    }

    private static async Task<AssistantScenarioTrace> CaptureResponseBargeInScheduleTraceAsync()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Asr.SetResult("session-2", "停一下");
        var thinkingBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Llm.EmitThinkingWhenEnabled = true;
        services.Llm.BlockAfterThinkingChunk = thinkingBlock;
        using VoiceAssistant assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableThinking: true,
            configure: options => options.BargeIn.PlaybackStartupGuardMs = 0);
        List<ScenarioEvent> events = [];
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        AddPublicTurnEvents(assistant, events, replied);
        var thinkingPublished = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantText += (_, value) =>
        {
            if (value.IsThinking)
            {
                thinkingPublished.TrySetResult();
            }
        };
        assistant.BargeInValidation += (_, value) => AddEvent(events, new ScenarioEvent(
            0,
            "BargeInValidation",
            Text: value.Kind == BargeInValidationKind.Accepted
                ? $"{value.Kind}:{value.Reason}"
                : value.Kind.ToString()));

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await thinkingPublished.Task.WaitAsync(TimeSpan.FromSeconds(2));
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        await WaitUntilAsync(() => services.Asr.FinishCount >= 2);
        thinkingBlock.TrySetResult();
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        await assistant.StopAsync();

        AssistantScenarioTrace trace = CreateTrace(
            "response-complete-vs-barge-in",
            [
                new(1, "response.started", "T1"),
                new(2, "barge-in", "before response.generated"),
                new(3, "asr.finish", "session-2"),
            ],
            events,
            [
                new(1, "llm.stream", $"count={services.Llm.Prompts.Count}"),
                new(2, "asr.finish", $"count={services.Asr.FinishCount}"),
            ],
            assistant);
        return trace with { ScheduleId = "runtime-barge-in-before-response-generated" };
    }

    private static void AddPublicTurnEvents(
        VoiceAssistant assistant,
        List<ScenarioEvent> events,
        TaskCompletionSource replied)
    {
        assistant.StateChanged += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "StateChanged", value.From.ToString(), value.To.ToString()));
        assistant.WakeModeChanged += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "WakeModeChanged", value.From.ToString(), value.To.ToString()));
        assistant.WakeupDetected += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "WakeupDetected", Text: value.Wakeup.Keyword));
        assistant.UserSpoke += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "UserSpoke", Text: value.Text));
        assistant.AssistantText += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "AssistantText", Text: value.Delta, IsThinking: value.IsThinking));
        assistant.AssistantResponseGenerated += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "AssistantResponseGenerated", Text: value.Text));
        assistant.AssistantReplied += (_, value) =>
        {
            AddEvent(events, new ScenarioEvent(0, "AssistantReplied", Text: value.Text));
            replied.TrySetResult();
        };
    }

    private static void AddWakeEvents(VoiceAssistant assistant, List<ScenarioEvent> events)
    {
        assistant.WakeModeChanged += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "WakeModeChanged", value.From.ToString(), value.To.ToString()));
        assistant.WakeupDetected += (_, value) => AddEvent(events, new ScenarioEvent(
            0, "WakeupDetected", Text: value.Wakeup.Keyword));
    }

    private static void AddEvent(List<ScenarioEvent> events, ScenarioEvent value)
    {
        lock (events)
            events.Add(value with { PublicationSeq = events.Count + 1 });
    }

    private static AssistantScenarioTrace CreateTrace(
        string scenarioId,
        IReadOnlyList<ScenarioInput> inputs,
        IReadOnlyList<ScenarioEvent> events,
        IReadOnlyList<ScenarioProviderCall> providerCalls,
        VoiceAssistant assistant)
        => new(
            "voicebox.assistant/scenario-trace/v1",
            "v1",
            scenarioId,
            "ExactTrace",
            inputs,
            events.OrderBy(static value => value.PublicationSeq).ToArray(),
            providerCalls,
            new ScenarioFinalState(assistant.State.ToString(), assistant.WakeMode.ToString()));

    private static async Task AssertOrRegenerateSpecTraceAsync(
        string relativePath,
        AssistantScenarioTrace actual)
    {
        if (string.Equals(
                Environment.GetEnvironmentVariable("VOICEBOX_REGENERATE_GOLDENS"),
                "1",
                StringComparison.Ordinal))
        {
            string path = Path.Combine(
                FindRepositoryRoot(),
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
}
