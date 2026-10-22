using VoiceBox.Assistant.Api.Playback;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task EnqueueSpeechAsync_CancellationDuringDrainClearsAudioAndContinuesQueue()
    {
        var services = TestServices.Create();
        services.Audio.HoldPlayback();
        services.Audio.DrainPolled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant();
        using var cancellation = new CancellationTokenSource();
        await assistant.StartAsync();
        var active = await assistant.EnqueueSpeechAsync("当前播放", cancellation.Token);
        await services.Audio.DrainPolled.Task.WaitAsync(TimeSpan.FromSeconds(2));
        var pending = await assistant.EnqueueSpeechAsync("后续播放");

        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => active.Completion.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(1, services.Audio.ClearQueueCount);
        await pending.Completion.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(["当前播放", "后续播放"], services.Tts.SynthesizedTexts);
    }

    [Fact]
    public async Task EnqueueSpeechAsync_CompletesAfterPlaybackAndUsesGlobalTtsConfiguration()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant(
            configure: options =>
            {
                options.TtsVoice = "configured-voice";
                options.TtsSpeed = 1.25f;
            });

        await assistant.StartAsync();

        var submission = await assistant.EnqueueSpeechAsync("请直接播报这段文字");

        Assert.True(submission.RequestId > 0);
        await submission.Completion.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(["请直接播报这段文字"], services.Tts.SynthesizedTexts);
        Assert.NotNull(services.Tts.LastStreamOptions);
        Assert.Equal("configured-voice", services.Tts.LastStreamOptions!.Voice);
        Assert.Equal(1.25f, services.Tts.LastStreamOptions.Speed);
    }

    [Fact]
    public async Task EnqueueSpeechAsync_StopCancelsActiveAndQueuedRequests()
    {
        var services = TestServices.Create();
        services.Tts.BlockBeforeFirstChunk = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant();

        await assistant.StartAsync();
        var active = await assistant.EnqueueSpeechAsync("第一段");
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Count == 1);
        var queued = await assistant.EnqueueSpeechAsync("第二段");

        await assistant.StopAsync().WaitAsync(TimeSpan.FromSeconds(2));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await active.Completion);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await queued.Completion);
        Assert.Equal(["第一段"], services.Tts.SynthesizedTexts);
    }

    [Fact]
    public async Task EnqueueSpeechAsync_SharesTheQueueWithAnOrdinaryAssistantReply()
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        var firstSpeechGate = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockBeforeFirstChunk = firstSpeechGate;
        using var assistant = services.CreateAssistant();
        var replied = new TaskCompletionSource<string>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Count == 1);

        var direct = await assistant.EnqueueSpeechAsync("排在回复之后");
        await Task.Delay(50);
        Assert.Equal(["好的。"], services.Tts.SynthesizedTexts);

        firstSpeechGate.TrySetResult();
        await replied.Task.WaitAsync(TimeSpan.FromSeconds(2));
        await direct.Completion.WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal(["好的。", "排在回复之后"], services.Tts.SynthesizedTexts);
    }

    [Fact]
    public async Task EnqueueSpeechAsync_HeadIsPlayedBeforeWaitingTail()
    {
        var services = TestServices.Create();
        var firstGate = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockBeforeFirstChunk = firstGate;
        using var assistant = services.CreateAssistant();

        await assistant.StartAsync();
        var active = await assistant.EnqueueSpeechAsync("当前播放");
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Count == 1);
        var tail = await assistant.EnqueueSpeechAsync("队尾");
        var head = await assistant.EnqueueSpeechAsync(
            "队头",
            SpeechPlaybackQueuePosition.Head);

        Assert.Equal(["当前播放"], services.Tts.SynthesizedTexts);
        services.Tts.BlockBeforeFirstChunk = null;
        firstGate.TrySetResult();

        await Task.WhenAll(active.Completion, head.Completion, tail.Completion);
        Assert.Equal(["当前播放", "队头", "队尾"], services.Tts.SynthesizedTexts);
    }

    [Fact]
    public async Task StopCurrentSpeechPlayback_StopsOnlyCurrentAndContinuesPendingQueue()
    {
        var services = TestServices.Create();
        var firstGate = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        services.Tts.BlockBeforeFirstChunk = firstGate;
        using var assistant = services.CreateAssistant();

        await assistant.StartAsync();
        var active = await assistant.EnqueueSpeechAsync("当前播放");
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Count == 1);
        var pending = await assistant.EnqueueSpeechAsync("后续播放");

        services.Tts.BlockBeforeFirstChunk = null;
        assistant.StopCurrentSpeechPlayback();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await active.Completion);
        await pending.Completion.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(["当前播放", "后续播放"], services.Tts.SynthesizedTexts);
        Assert.Equal(1, services.Audio.ClearQueueCount);

        assistant.StopCurrentSpeechPlayback();
        Assert.Equal(1, services.Audio.ClearQueueCount);
    }
}
