using System.Runtime.CompilerServices;
using System.Threading.Channels;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Conversation.Abstractions;
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

    private static VoiceAssistant CreateAssistantWith(TestServices services, IConversationSession session)
    {
        var options = new VoiceAssistantOptions
        {
            MinSentenceChars = 1,
            PlaybackPrerollMs = 0,
            PlaybackTailBufferMs = 0,
            Listening = { MinDurationMs = 0, MinRms = 0 },
        };
        return VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseConversation(session)
            .UseSpeechSynthesis(services.Tts)
            .Build();
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
