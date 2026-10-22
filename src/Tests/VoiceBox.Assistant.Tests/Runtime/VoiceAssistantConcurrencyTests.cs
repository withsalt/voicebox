using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Core.Services.LLMService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task ScheduleCr004_StaleAsrStartAfterWakeupInterrupt_CleansOnlyTheOldSession()
    {
        var services = TestServices.Create();
        services.Asr.FinishEntered = new TaskCompletionSource<string>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(enableWakeWord: true);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, value) => replied.TrySetResult(value.Text);

        await assistant.StartAsync();
        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        await WaitUntilAsync(() => assistant.WakeMode == AssistantWakeMode.ContinuousConversation);

        // First turn starts an ASR session but leaves its provider completion pending.
        Task firstListening = WaitForAssistantStateAsync(assistant, AssistantState.Listening);
        services.Vad.EmitSpeechStarted();
        await firstListening;
        services.Vad.EmitSpeaking([0.1f]);

        // The second wakeup interrupts the first turn before its StartSession completion.
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.3f]);
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        Task secondListening = WaitForAssistantStateAsync(assistant, AssistantState.Listening);
        services.Vad.EmitSpeechStarted();
        await secondListening;
        services.Vad.EmitSpeaking([0.2f, 0.3f]);

        // The first queued completion arrives after the new turn exists and must be cleaned up.
        services.Asr.CompleteStart("stale-session-1");
        Assert.Equal(
            "stale-session-1",
            await services.Asr.FinishEntered.Task.WaitAsync(TimeSpan.FromSeconds(2)));

        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("好的。", reply);
        Assert.False(services.Asr.FedAudio.ContainsKey("stale-session-1"));
        Assert.True(services.Asr.FedAudio.ContainsKey("session-2"));
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        Assert.Equal(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task ScheduleCr007_FatalDetectorAndHostStop_PublishFaultBeforeSingleTerminalStop()
    {
        var services = TestServices.Create();
        services.Audio.StopEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StopBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(enableBargeIn: true);
        var observed = new List<string>();
        var faultEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var faultRelease = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.Faulted += (_, _) =>
        {
            lock (observed)
                observed.Add("Faulted");
            faultEntered.TrySetResult();
            faultRelease.Task.GetAwaiter().GetResult();
        };
        assistant.StateChanged += (_, value) =>
        {
            if (value.To == AssistantState.Stopped)
            {
                lock (observed)
                    observed.Add("Stopped");
            }
        };

        await assistant.StartAsync();
        services.PlaybackDetector.PushException = new InvalidOperationException("fatal detector schedule");
        Task callback = Task.Run(() => services.Audio.EmitFrame(
            new float[320], new float[320], new float[320], new float[320]));

        await faultEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Task hostStop = assistant.StopAsync();
        await services.Audio.StopEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.False(hostStop.IsCompleted);

        faultRelease.TrySetResult();
        services.Audio.StopBlock.TrySetResult();
        await Task.WhenAll(callback, hostStop).WaitAsync(TimeSpan.FromSeconds(2));

        lock (observed)
            Assert.Equal(["Faulted", "Stopped"], observed);
        Assert.Equal(1, services.Audio.StopCount);
        Assert.Equal(AssistantState.Stopped, assistant.State);
    }

    [Fact]
    public async Task ScheduleCr008_BlockedSubscriber_DefersLaterSubscriberUntilBarrierRelease()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var later = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        assistant.StateChanged += (_, value) =>
        {
            if (value.To != AssistantState.Listening)
                return;
            entered.TrySetResult();
            release.Task.GetAwaiter().GetResult();
        };
        assistant.StateChanged += (_, value) =>
        {
            if (value.To == AssistantState.Listening)
                later.TrySetResult();
        };

        await assistant.StartAsync();
        Task producer = Task.Run(services.Vad.EmitSpeechStarted);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.False(later.Task.IsCompleted);

        release.TrySetResult();
        await Task.WhenAll(producer, later.Task).WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(AssistantState.Listening, assistant.State);
    }

    [Fact]
    public async Task SameIngressSequence_ReplaysOneHundredTimesWithTheSameTerminalTrace()
    {
        const int replayCount = 100;
        string? expected = null;

        for (int replay = 0; replay < replayCount; replay++)
        {
            var services = TestServices.Create();
            services.Asr.CompleteStart("session-1");
            using var assistant = services.CreateAssistant();
            var events = new List<string>();
            var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
            assistant.StateChanged += (_, value) => events.Add($"state:{value.From}->{value.To}");
            assistant.AssistantReplied += (_, value) => replied.TrySetResult(value.Text);

            await assistant.StartAsync();
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
            string response;
            try
            {
                response = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
            }
            catch (TimeoutException exception)
            {
                throw new InvalidOperationException(
                    $"Replay {replay} timed out: state={assistant.State}, " +
                    $"asrFinish={services.Asr.FinishCount}, prompts={services.Llm.Prompts.Count}, " +
                    $"tts={services.Tts.SynthesizedTexts.Count}, acceptedSamples={services.Audio.AcceptedSamples}.",
                    exception);
            }
            await assistant.StopAsync();

            string trace = string.Join('|', events) + $"|reply:{response}|finish:{services.Asr.FinishCount}";
            expected ??= trace;
            Assert.Equal(expected, trace);
        }
    }

    [Fact]
    public async Task LifecycleStartStop_OneThousandDeterministicIterationsConverge()
    {
        const int iterationCount = 1_000;

        for (int iteration = 0; iteration < iterationCount; iteration++)
        {
            var services = TestServices.Create();
            using var assistant = services.CreateAssistant();

            await assistant.StartAsync();
            await assistant.StopAsync();

            Assert.Equal(1, services.Audio.StartCount);
            Assert.Equal(1, services.Audio.StopCount);
            Assert.Equal(AssistantState.Stopped, assistant.State);
        }
    }

    [Fact]
    public async Task MetamorphicAudioChunking_EquivalentInputAndEmptyChunksPreserveTurnOutput()
    {
        float[] source = Enumerable.Range(0, 960).Select(static value => value / 960f).ToArray();
        AudioChunkingResult baseline = await RunAudioChunksAsync(source, [960], includeEmptyChunk: false);

        foreach (int[] chunks in new[] { new[] { 160, 160, 160, 160, 160, 160 }, new[] { 256, 256, 256, 192 } })
        {
            AudioChunkingResult variant = await RunAudioChunksAsync(source, chunks, includeEmptyChunk: true);
            Assert.Equal(baseline.Reply, variant.Reply);
            Assert.Equal(baseline.FedSamples, variant.FedSamples);
        }
    }

    [Fact]
    public async Task MetamorphicTextChunking_PreservesVisibleTextTtsAndConversationHistory()
    {
        TextChunkingResult baseline = await RunTextChunksAsync(
            [new LlmTextChunk { Text = "好的。", Kind = LlmChunkKind.Text, IsFinal = true }]);
        TextChunkingResult variant = await RunTextChunksAsync(
            [
                new LlmTextChunk { Text = "好", Kind = LlmChunkKind.Text },
                new LlmTextChunk { Text = "的。", Kind = LlmChunkKind.Text, IsFinal = true },
            ]);

        Assert.Equal(baseline, variant);
    }

    [Fact]
    public async Task SynchronousIngress_CallbackMeasurementsMeetTheOnePeriodObservationBudget()
    {
        IReadOnlyList<CallbackMeasurement> measurements = await CaptureCurrentCallbackMeasurementsAsync();

        Assert.Equal(3, measurements.Count);
        foreach (CallbackMeasurement measurement in measurements)
        {
            long allocationCeiling = measurement.CallbackSamples switch
            {
                160 => 5_380,
                256 => 6_148,
                320 => 6_660,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(measurement.CallbackSamples),
                    measurement.CallbackSamples,
                    null),
            };
            Assert.True(
                measurement.AllocatedBytesPerCallback <= allocationCeiling,
                $"{measurement.CallbackSamples}-sample callback allocated " +
                $"{measurement.AllocatedBytesPerCallback} bytes versus the " +
                $"{allocationCeiling}-byte ceiling.");
            Assert.True(
                measurement.P99Milliseconds <= measurement.CallbackPeriodMilliseconds,
                $"{measurement.CallbackSamples}-sample p99 {measurement.P99Milliseconds:F3}ms exceeded " +
                $"its {measurement.CallbackPeriodMilliseconds:F3}ms callback period.");
            Assert.True(
                measurement.MaxMilliseconds <= measurement.CallbackPeriodMilliseconds * 2,
                $"{measurement.CallbackSamples}-sample max {measurement.MaxMilliseconds:F3}ms exceeded " +
                $"two callback periods.");
        }
    }

    internal static async Task<IReadOnlyList<CallbackMeasurement>> CaptureCurrentCallbackMeasurementsAsync()
    {
        const int sampleRate = 16_000;
        const int iterations = 200;
        var measurements = new List<CallbackMeasurement>();
        foreach (int callbackSamples in new[] { 160, 256, 320 })
        {
            var services = TestServices.Create();
            using var assistant = services.CreateAssistant();
            await assistant.StartAsync();
            float[] frame = new float[callbackSamples];
            services.Audio.Emit(frame); // JIT and subscription warm-up are outside the measurement set.

            var elapsedTicks = new long[iterations];
            long allocationsBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int index = 0; index < iterations; index++)
            {
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                services.Audio.Emit(frame);
                elapsedTicks[index] = System.Diagnostics.Stopwatch.GetTimestamp() - started;
            }
            long allocationsAfter = GC.GetAllocatedBytesForCurrentThread();
            await assistant.StopAsync();

            Array.Sort(elapsedTicks);
            double tickToMilliseconds = 1000d / System.Diagnostics.Stopwatch.Frequency;
            int p99Index = (int)Math.Ceiling(iterations * 0.99) - 1;
            measurements.Add(new CallbackMeasurement(
                callbackSamples,
                callbackSamples * 1000d / sampleRate,
                elapsedTicks[p99Index] * tickToMilliseconds,
                elapsedTicks[^1] * tickToMilliseconds,
                (allocationsAfter - allocationsBefore) / iterations));
        }

        return measurements;
    }

    private static async Task<AudioChunkingResult> RunAudioChunksAsync(
        float[] source,
        IReadOnlyList<int> chunkLengths,
        bool includeEmptyChunk)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, value) => replied.TrySetResult(value.Text);

        await assistant.StartAsync();
        Task listening = WaitForAssistantStateAsync(assistant, AssistantState.Listening);
        services.Vad.EmitSpeechStarted();
        await listening;
        int offset = 0;
        foreach (int length in chunkLengths)
        {
            services.Vad.EmitSpeaking(source.AsSpan(offset, length).ToArray());
            offset += length;
            if (includeEmptyChunk)
                services.Vad.EmitSpeaking([]);
        }
        Assert.Equal(source.Length, offset);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));

        string reply = await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        float[] fed = services.Asr.FedAudio["session-1"].SelectMany(static value => value).ToArray();
        return new AudioChunkingResult(reply, fed);
    }

    private static async Task<TextChunkingResult> RunTextChunksAsync(IReadOnlyList<LlmTextChunk> chunks)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Llm.StreamChunks = chunks;
        using var assistant = services.CreateAssistant();
        var visible = new List<string>();
        var firstReply = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var secondReply = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int replyCount = 0;
        assistant.AssistantText += (_, value) =>
        {
            if (!value.IsThinking)
                visible.Add(value.Delta);
        };
        assistant.AssistantReplied += (_, _) =>
        {
            if (Interlocked.Increment(ref replyCount) == 1)
                firstReply.TrySetResult();
            else
                secondReply.TrySetResult();
        };

        await assistant.StartAsync();
        Task firstListening = WaitForAssistantStateAsync(assistant, AssistantState.Listening);
        services.Vad.EmitSpeechStarted();
        await firstListening;
        services.Vad.EmitSpeaking([0.1f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await firstReply.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await WaitForAssistantStateAsync(assistant, AssistantState.Idle);

        Task secondListening = WaitForAssistantStateAsync(assistant, AssistantState.Listening);
        services.Vad.EmitSpeechStarted();
        await secondListening;
        services.Vad.EmitSpeaking([0.2f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await secondReply.Task.WaitAsync(TimeSpan.FromSeconds(2));

        ChatMessage[] secondPrompt = services.Llm.Prompts[^1];
        string history = string.Join(
            '|',
            secondPrompt.Select(static message => $"{message.Role}:{string.Concat(message.Content.OfType<TextContent>().Select(static content => content.Text))}"));
        return new TextChunkingResult(
            string.Concat(visible),
            string.Join('|', services.Tts.SynthesizedTexts),
            history);
    }

    private static async Task WaitForAssistantStateAsync(
        VoiceAssistant assistant,
        AssistantState expected)
    {
        if (assistant.State == expected)
            return;

        var reached = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.StateChanged += OnStateChanged;
        try
        {
            if (assistant.State == expected)
                reached.TrySetResult();
            await reached.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        finally
        {
            assistant.StateChanged -= OnStateChanged;
        }

        void OnStateChanged(object? sender, AssistantStateChangedEventArgs args)
        {
            if (args.To == expected)
                reached.TrySetResult();
        }
    }

    private sealed record AudioChunkingResult(string Reply, float[] FedSamples);

    private sealed record TextChunkingResult(string VisibleText, string TtsSentences, string History);

    internal sealed record CallbackMeasurement(
        int CallbackSamples,
        double CallbackPeriodMilliseconds,
        double P99Milliseconds,
        double MaxMilliseconds,
        long AllocatedBytesPerCallback);
}
