using System.Runtime.CompilerServices;
using System.Threading.Channels;
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task ACompletelyCustomBusinessSessionBuildsWithConversationOnly()
    {
        var services = TestServices.Create();
        await using var session = new DelegateConversationSession
        {
            Script = (_, _) => SingleTurnAsync("业务答复"),
        };
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        var replied = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        assistant.AssistantReplied += (_, e) => replied.TrySetResult(e.Text);

        await assistant.StartAsync();
        RunTurn(services, "custom-session");
        Assert.Equal("业务答复", await replied.Task.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Contains("业务答复", services.Tts.SynthesizedTexts);
        await assistant.StopAsync();
    }

    [Fact]
    public async Task CustomSessionReceivesOnlyTheCurrentUtteranceAudio()
    {
        var services = TestServices.Create();
        var requests = Channel.CreateUnbounded<ConversationRequest>();
        await using var session = new DelegateConversationSession
        {
            Script = (request, _) =>
            {
                requests.Writer.TryWrite(request);
                return SingleTurnAsync("业务答复");
            },
        };
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        await assistant.StartAsync();

        services.Asr.SetResult("first-audio", "第一句");
        RunTurn(services, "first-audio");
        ConversationRequest first = await requests.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));
        ConversationAudio audio = Assert.IsType<ConversationAudio>(first.Audio);
        Assert.Equal("第一句", first.Text);
        Assert.Equal(16_000, audio.SampleRate);
        Assert.Equal(1, audio.Channels);
        Assert.Equal([0.1f, 0.2f, 0.3f], audio.Samples.ToArray());
        Assert.Equal(services.Asr.FedAudio["first-audio"].SelectMany(chunk => chunk), audio.Samples.ToArray());
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        services.Asr.SetResult("second-audio", "第二句");
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([-0.4f, -0.5f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        services.Asr.CompleteStart("second-audio");
        ConversationRequest second = await requests.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));

        Assert.Equal("第二句", second.Text);
        Assert.Equal([-0.4f, -0.5f], Assert.IsType<ConversationAudio>(second.Audio).Samples.ToArray());
        Assert.NotEqual(first.RequestId, second.RequestId);
        await assistant.StopAsync();
        Assert.Equal([0.1f, 0.2f, 0.3f], audio.Samples.ToArray());
    }

    [Fact]
    public async Task RecognitionWithoutSamplesDoesNotReusePreviousAudio()
    {
        var services = TestServices.Create();
        var requests = Channel.CreateUnbounded<ConversationRequest>();
        await using var session = new DelegateConversationSession
        {
            Script = (request, _) =>
            {
                requests.Writer.TryWrite(request);
                return SingleTurnAsync("业务答复");
            },
        };
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        await assistant.StartAsync();
        RunTurn(services, "with-audio");
        ConversationRequest first = await requests.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));
        Assert.NotNull(first.Audio);
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        services.Asr.CompleteStart("without-audio");
        ConversationRequest second = await requests.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Null(second.Audio);
        await assistant.StopAsync();
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("duplicate")]
    [InlineData("mismatch")]
    public async Task CustomSessionProtocolFaultsNeverBecomeASuccessfulReply(string mode)
    {
        var services = TestServices.Create();
        await using var session = new DelegateConversationSession
        {
            Script = (_, _) => mode switch
            {
                "missing" => TextOnlyAsync("部分回答"),
                "duplicate" => DuplicateCompletedAsync(),
                "mismatch" => MismatchedCompletedAsync(),
                _ => throw new ArgumentOutOfRangeException(nameof(mode)),
            },
        };
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        var replies = new List<string>();
        assistant.AssistantReplied += (_, e) =>
        {
            lock (replies) replies.Add(e.Text);
        };

        await assistant.StartAsync();
        RunTurn(services, "fault-session");
        await WaitUntilAsync(() => assistant.State == AssistantState.Idle);

        lock (replies)
            Assert.Empty(replies);

        // The assistant remains usable after the faulty turn.
        session.Script = (_, _) => SingleTurnAsync("正常回答");
        RunTurn(services, "recovered-session");
        await WaitUntilAsync(() =>
        {
            lock (replies) return replies.Contains("正常回答");
        });
        await assistant.StopAsync();
    }

    [Fact]
    public async Task AssistantStartsAndStopsTheSessionButNeverDisposesIt()
    {
        var services = TestServices.Create();
        var session = new LifecycleConversationSession();
        using VoiceAssistant assistant = CreateAssistantWith(services, session);

        await assistant.StartAsync();
        Assert.Equal(1, session.StartCount);
        await assistant.StopAsync();
        Assert.Equal(1, session.StopCount);

        assistant.Dispose();
        Assert.Equal(0, session.DisposeCount);
        await session.DisposeAsync();
    }

    [Fact]
    public async Task SessionStartFailureRollsBackTheStartedParticipants()
    {
        var services = TestServices.Create();
        var session = new LifecycleConversationSession { FailStart = true };
        using VoiceAssistant assistant = CreateAssistantWith(services, session);

        await Assert.ThrowsAnyAsync<Exception>(() => assistant.StartAsync());
        Assert.Equal(1, session.StartCount);
        // A session whose StartAsync throws is responsible for cleaning its own partial startup;
        // the Assistant only unwinds participants that reported success.
        Assert.Equal(0, session.StopCount);
        Assert.NotEqual(AssistantState.Idle, assistant.State);
    }

    [Fact]
    public async Task BackgroundSessionResultsArePublishedAfterTheForegroundBoundary()
    {
        var services = TestServices.Create();
        var session = new BackgroundConversationSession();
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        var replies = new List<string>();
        var assistantText = new List<string>();
        assistant.AssistantReplied += (_, e) =>
        {
            lock (replies) replies.Add(e.Text);
        };
        assistant.AssistantText += (_, e) =>
        {
            if (!e.IsThinking)
            {
                lock (assistantText) assistantText.Add(e.Delta);
            }
        };

        await assistant.StartAsync();
        RunTurn(services, "background-session");
        await WaitUntilAsync(() =>
        {
            lock (replies) return replies.Count == 1;
        });

        ConversationRequest request;
        lock (session.Requests)
            request = session.Requests.Single();
        session.Publish(new ConversationBackgroundMessage(
            Guid.NewGuid(),
            session.ConversationId,
            request.RequestId,
            "后台结果1"));
        session.Publish(new ConversationBackgroundMessage(
            Guid.NewGuid(),
            session.ConversationId,
            request.RequestId,
            "后台结果2"));

        await WaitUntilAsync(() =>
        {
            lock (replies) return replies.Count == 3;
        });
        lock (replies)
        {
            Assert.Equal("前台确认", replies[0]);
            Assert.Equal("后台结果1", replies[1]);
            Assert.Equal("后台结果2", replies[2]);
        }
        lock (assistantText)
            Assert.Contains("后台结果1", assistantText);

        await assistant.StopAsync();
        Assert.Equal(1, session.StopCount);
        Assert.Equal(0, session.DisposeCount);
        await session.DisposeAsync();
    }

    [Fact]
    public async Task BackgroundResultArrivingBeforeForegroundCompletionWaitsForItsBoundary()
    {
        var services = TestServices.Create();
        var session = new EarlyBackgroundConversationSession();
        using VoiceAssistant assistant = CreateAssistantWith(services, session);
        var replies = new List<string>();
        assistant.AssistantReplied += (_, e) =>
        {
            lock (replies) replies.Add(e.Text);
        };

        await assistant.StartAsync();
        RunTurn(services, "early-background");
        await WaitUntilAsync(() =>
        {
            lock (replies) return replies.Count == 2;
        });

        lock (replies)
        {
            Assert.Equal("前台确认", replies[0]);
            Assert.Equal("后台先到", replies[1]);
        }
        await assistant.StopAsync();
        await session.DisposeAsync();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task InterruptedBackgroundPlaybackKeepsListeningWithoutLoggingFailure(bool stopPlaybackOnly)
    {
        var services = TestServices.Create();
        await using var session = new BackgroundConversationSession();
        var logger = new BackgroundPlaybackLogger();
        using VoiceAssistant assistant = CreateAssistantWith(services, session, logger, enableBargeIn: true);
        var replies = new ConcurrentQueue<string>();
        assistant.AssistantReplied += (_, e) => replies.Enqueue(e.Text);

        await assistant.StartAsync();
        RunTurn(services, "foreground");
        await WaitUntilAsync(() => replies.Count == 1 && assistant.State == AssistantState.Idle);
        ConversationRequest request;
        lock (session.Requests)
            request = session.Requests.Single();

        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        session.Publish(new ConversationBackgroundMessage(Guid.NewGuid(), session.ConversationId, request.RequestId, "天气播报"));
        await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Tts.SynthesizedTexts.Contains("天气播报"));
        services.Tts.BlockAfterFirstChunk = null;
        if (stopPlaybackOnly)
        {
            // Stop the scheduler's playback token without canceling the response token.
            assistant.StopCurrentSpeechPlayback();
            await WaitUntilAsync(() => assistant.State == AssistantState.Idle);
        }
        else
        {
            services.Asr.CompleteStart("interruption");
            services.Asr.SetResult("interruption", "好了，我知道了，退下吧");
            EmitNearEndFrames(services, count: 10, renderRms: 0.08, captureRms: 0.12, emitMainAudio: true);
            await WaitUntilAsync(() => assistant.State == AssistantState.Listening && services.Audio.ClearQueueCount > 0);
            EmitSilenceFrames(services, count: 8, renderRms: 0);
            await WaitUntilAsync(() => replies.Count == 2 && assistant.State == AssistantState.Idle);
            ConversationRequest interruption;
            lock (session.Requests)
                interruption = session.Requests.Last();
            Assert.Equal("好了，我知道了，退下吧", interruption.Text);
            ConversationAudio audio = Assert.IsType<ConversationAudio>(interruption.Audio);
            Assert.Equal(services.Asr.FedAudio["interruption"].SelectMany(chunk => chunk), audio.Samples.ToArray());
            Assert.Equal(16_000, audio.SampleRate);
            Assert.Equal(1, audio.Channels);
        }

        // A later background result must still be consumed after the interrupted one.
        session.Publish(new ConversationBackgroundMessage(Guid.NewGuid(), session.ConversationId, request.RequestId, "后续结果"));
        await WaitUntilAsync(() => replies.Contains("后续结果"));
        await assistant.StopAsync();
        Assert.DoesNotContain("天气播报", replies);
        Assert.Contains(logger.Entries, entry => entry.Message.Contains("Background response interrupted"));
        Assert.DoesNotContain(logger.Entries, entry => entry.Level >= LogLevel.Error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BackgroundPlaybackStillReportsUnexpectedProviderFailures(bool cancellationException)
    {
        var services = TestServices.Create();
        await using var session = new BackgroundConversationSession();
        var logger = new BackgroundPlaybackLogger();
        using VoiceAssistant assistant = CreateAssistantWith(services, session, logger);
        await assistant.StartAsync();
        RunTurn(services, "foreground");
        await WaitUntilAsync(() => services.Tts.SynthesizedTexts.Count == 1 && assistant.State == AssistantState.Idle);
        ConversationRequest request;
        lock (session.Requests)
            request = session.Requests.Single();
        services.Tts.StreamException = cancellationException
            ? new OperationCanceledException("provider timeout")
            : new InvalidOperationException("provider failed");
        session.Publish(new ConversationBackgroundMessage(Guid.NewGuid(), session.ConversationId, request.RequestId, "后台结果"));
        await WaitUntilAsync(() => logger.Entries.Any(entry => entry.Level == LogLevel.Error &&
            entry.Message.Contains("Background conversation playback failed")));
        await assistant.StopAsync();
    }

    private sealed class BackgroundPlaybackLogger : ILogger<VoiceAssistant>
    {
        public ConcurrentQueue<(LogLevel Level, string Message)> Entries { get; } = new();
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
            => Entries.Enqueue((logLevel, formatter(state, exception)));
    }

    private static VoiceAssistant CreateAssistantWith(TestServices services, IConversationSession session,
        ILogger<VoiceAssistant>? logger = null, bool enableBargeIn = false)
    {
        var options = new VoiceAssistantOptions
        {
            MinSentenceChars = 1,
            PlaybackPrerollMs = 0,
            PlaybackTailBufferMs = 0,
            Listening = { MinDurationMs = 0, MinRms = 0 },
            BargeIn = { Enable = enableBargeIn, MinDurationMs = 120, PlaybackStartupGuardMs = 0, ActiveNearEndEndSilenceMs = 60 },
        };
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseConversation(session)
            .UseSpeechSynthesis(services.Tts);
        if (enableBargeIn)
            builder.AddBargeIn(options.BargeIn, TestAcousticPathDelayEstimatorFactory.Instance);
        return builder.BuildForBenchmark(TimeProvider.System, new VoiceAssistantBenchmarkProbe(), logger,
            playbackBargeInDetectorFactory: new FakePlaybackBargeInDetectorFactory(services.PlaybackDetector));
    }

    private static void RunTurn(TestServices services, string asrSessionId)
    {
        services.Vad.EmitSpeechStarted();
        services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
        services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
        services.Asr.CompleteStart(asrSessionId);
    }

    private static async IAsyncEnumerable<ConversationEvent> SingleTurnAsync(string text)
    {
        await Task.Yield();
        yield return new ConversationTextDelta(text);
        yield return new ConversationCompleted(text);
    }

    private static async IAsyncEnumerable<ConversationEvent> TextOnlyAsync(string text)
    {
        await Task.Yield();
        yield return new ConversationTextDelta(text);
    }

    private static async IAsyncEnumerable<ConversationEvent> DuplicateCompletedAsync()
    {
        await Task.Yield();
        yield return new ConversationTextDelta("第一次");
        yield return new ConversationCompleted("第一次");
        yield return new ConversationCompleted("第二次");
    }

    private static async IAsyncEnumerable<ConversationEvent> MismatchedCompletedAsync()
    {
        await Task.Yield();
        yield return new ConversationTextDelta("可见文字");
        yield return new ConversationCompleted("不同的完成文字");
    }

    private sealed class DelegateConversationSession : IConversationSession
    {
        public string ConversationId => "delegate-session";

        public Func<ConversationRequest, CancellationToken, IAsyncEnumerable<ConversationEvent>> Script
            { get; set; } = (_, _) => SingleTurnAsync("默认");

        public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
            => Script(request, cancellationToken);

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class LifecycleConversationSession : IConversationSession, IConversationLifecycle
    {
        public int StartCount { get; private set; }

        public int StopCount { get; private set; }

        public int DisposeCount { get; private set; }

        public bool FailStart { get; init; }

        public string ConversationId => "lifecycle-session";

        public Task StartAsync(CancellationToken cancellationToken = default)
        {
            StartCount++;
            return FailStart
                ? Task.FromException(new InvalidOperationException("start failed"))
                : Task.CompletedTask;
        }

        public Task StopAsync()
        {
            StopCount++;
            return Task.CompletedTask;
        }

        public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
            => SingleTurnAsync("生命周期回答");

        public ValueTask DisposeAsync()
        {
            DisposeCount++;
            return ValueTask.CompletedTask;
        }
    }

    private sealed class EarlyBackgroundConversationSession :
        IConversationSession,
        IConversationBackgroundOutput
    {
        private readonly object _lock = new();
        private Channel<ConversationBackgroundMessage>? _channel;

        public string ConversationId => "early-background-session";

        public Task StartAsync(CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                _channel = Channel.CreateUnbounded<ConversationBackgroundMessage>();
            }
            return Task.CompletedTask;
        }

        public Task StopAsync()
        {
            lock (_lock)
                _channel?.Writer.TryComplete();
            return Task.CompletedTask;
        }

        public async IAsyncEnumerable<ConversationBackgroundMessage> ReadBackgroundMessagesAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Channel<ConversationBackgroundMessage> channel;
            lock (_lock)
                channel = _channel ?? throw new InvalidOperationException("not started");

            await foreach (ConversationBackgroundMessage message in
                channel.Reader.ReadAllAsync(cancellationToken))
            {
                yield return message;
            }
        }

        public async IAsyncEnumerable<ConversationEvent> StreamAsync(
            ConversationRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                Assert.NotNull(_channel);
                _channel!.Writer.TryWrite(new ConversationBackgroundMessage(
                    Guid.NewGuid(),
                    ConversationId,
                    request.RequestId,
                    "后台先到"));
            }
            await Task.Delay(150, cancellationToken);
            yield return new ConversationTextDelta("前台确认");
            yield return new ConversationCompleted("前台确认");
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class BackgroundConversationSession :
        IConversationSession,
        IConversationBackgroundOutput
    {
        private readonly object _lock = new();
        private Channel<ConversationBackgroundMessage>? _channel;

        public List<ConversationRequest> Requests { get; } = [];

        public int StopCount { get; private set; }

        public int DisposeCount { get; private set; }

        public string ConversationId => "background-session";

        public Task StartAsync(CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                _channel = Channel.CreateUnbounded<ConversationBackgroundMessage>(
                    new UnboundedChannelOptions
                    {
                        SingleReader = true,
                        SingleWriter = false,
                        AllowSynchronousContinuations = false,
                    });
            }
            return Task.CompletedTask;
        }

        public Task StopAsync()
        {
            StopCount++;
            lock (_lock)
                _channel?.Writer.TryComplete();
            return Task.CompletedTask;
        }

        public void Publish(ConversationBackgroundMessage message)
        {
            lock (_lock)
            {
                Assert.NotNull(_channel);
                Assert.True(_channel!.Writer.TryWrite(message));
            }
        }

        public async IAsyncEnumerable<ConversationBackgroundMessage> ReadBackgroundMessagesAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Channel<ConversationBackgroundMessage> channel;
            lock (_lock)
                channel = _channel ?? throw new InvalidOperationException("not started");

            await foreach (ConversationBackgroundMessage message in
                channel.Reader.ReadAllAsync(cancellationToken))
            {
                yield return message;
            }
        }

        public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
        {
            lock (_lock)
                Requests.Add(request);
            return SingleTurnAsync("前台确认");
        }

        public ValueTask DisposeAsync()
        {
            DisposeCount++;
            return ValueTask.CompletedTask;
        }
    }
}
