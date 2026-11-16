using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData("", true, false)]
    [InlineData("", true, true)]
    [InlineData("", false, true)]
    [InlineData("嗯", true, false)]
    public async Task PlaybackBargeIn_WithInvalidAsr_AllowsNextCapturedTurn(
        string recognizedText,
        bool continuousConversation,
        bool wakeAgain)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Asr.SetResult("session-2", recognizedText);
        TaskCompletionSource validationFinish = services.Asr.BlockFinish("session-2");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true,
            enableWakeWord: true,
            configure: options =>
            {
                options.Speaker.Confirmation.Enable = false;
                options.BargeIn.PlaybackStartupGuardMs = 0;
                options.Conversation.ContinuousConversationEnabled = continuousConversation;
            });
        var runtime = GetPrivateField<AssistantRuntime>(assistant, "_runtime");
        var controlLoop = RuntimeGraphProbe.RequireField<AssistantControlLoopHost>(runtime, "_loopHost");
        int wakeups = 0;
        assistant.WakeupDetected += (_, _) => Interlocked.Increment(ref wakeups);
        await assistant.StartAsync();

        services.Audio.Emit([0.1f]);
        services.Wakeup.EmitWakeup();
        services.Audio.Emit([0.2f]);
        await WaitUntilAsync(() => Volatile.Read(ref wakeups) == 1);
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

        services.Asr.CompleteStart("session-2");
        services.Vad.EmitSpeechStarted();
        EmitVadAudioWithDuplex(services, BargeInVoiceSamples());
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(500));
        await WaitUntilAsync(() => assistant.State == AssistantState.Listening && services.Audio.ClearQueueCount == 1);
        validationFinish.SetResult();
        // Idle is published before the control-loop cleanup finishes. Wait for the dispatch to
        // finish so the next capture tests the settled input route.
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle && !controlLoop.IsInMutation);
        Assert.Single(services.Llm.Prompts);
        Assert.Equal(continuousConversation
            ? AssistantWakeMode.ContinuousConversation
            : AssistantWakeMode.WaitingForWakeWord, assistant.WakeMode);

        if (wakeAgain)
        {
            services.Wakeup.EmitWakeup();
            services.Audio.Emit([0.4f]);
            await WaitUntilAsync(() => Volatile.Read(ref wakeups) == 2);
        }

        services.Tts.BlockAfterFirstChunk = null;
        services.Asr.CompleteStart("session-3");
        const string nextText = "今天天气怎么样";
        services.Asr.SetResult("session-3", nextText);
        var replied = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, _) => replied.TrySetResult();
        float[] nextAudio = [0.6f, 0.5f, 0.4f];
        // Emit VAD events only if actual captured audio reaches the VAD. Directly emitting them
        // here would bypass the input gate and let the broken recovery path pass this test.
        services.Vad.PredictAction = () =>
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(nextAudio);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        };
        int predictionsBefore = services.Vad.PredictedAudio.Count;
        services.Audio.Emit(nextAudio);
        Assert.True(services.Vad.PredictedAudio.Count > predictionsBefore,
            "The completed interruption must restore the capture route to VAD.");
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(2, services.Llm.Prompts.Count);
        Assert.Contains(services.Llm.Prompts[1], message => MessageText(message) == nextText);
        Assert.Contains(services.Asr.FedAudio["session-3"], chunk => chunk.SequenceEqual(nextAudio));
        Assert.Equal(1, services.Audio.ClearQueueCount);
    }
}
