using System.Runtime.CompilerServices;
using System.Threading.Channels;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Internal;
using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 带结构化工具能力的 LLM 会话。它拥有本地工具和外部工具来源, 通过可选生命周期建立连接,
/// 并把请求结束之后的后台结果发布到独立通道。
/// </summary>
/// <remarks>
/// <see cref="StartAsync"/> 完成后才能发起请求。停止或释放会取消尚未完成的启动,
/// 等待启动清理和运行任务退出后断开连接。停止保留业务历史, 完成后允许再次启动。
/// DisposeAsync 释放本会话拥有的工具与来源, 不释放借用的 LLM 服务。
/// </remarks>
public sealed class LlmToolConversationSession :
    IConversationSession,
    IConversationBackgroundOutput,
    IBackgroundResultPublisher
{
    private const int BackgroundChannelCapacity = 64;

    private readonly object _lifecycleLock = new();
    private readonly ConversationSessionCore _core;
    private readonly ToolTurnExecutor _executor;
    private SessionRun? _run;
    private int _disposed;

    internal LlmToolConversationSession(
        string conversationId,
        LlmConversationOptions options,
        IReadOnlyList<IConversationContextProvider> contextProviders,
        IToolCallingStreamLLMService service,
        ToolConfiguration tools)
    {
        var history = new ConversationHistoryStore(
            options.SystemPrompt,
            options.MaxHistoryTurns,
            tools.Options.ToolTranscriptRetentionTurns,
            tools.Options.MaxPersistedToolResultBytes);
        var runtime = new ConversationToolRuntime(
            tools.Tools,
            tools.Providers,
            tools.ApprovalPolicy,
            tools.Options);
        _executor = new ToolTurnExecutor(
            new StructuredLlmTurnClient(service),
            runtime,
            history,
            this,
            tools.Options);
        _core = new ConversationSessionCore(
            conversationId,
            history,
            options,
            contextProviders,
            _executor);
    }

    public string ConversationId => _core.ConversationId;

    public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
    {
        lock (_lifecycleLock)
        {
            GetStartedRun();
            return _core.StreamAsync(request, cancellationToken);
        }
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        SessionRun run;
        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            if (_run is not null)
                throw new InvalidOperationException("The conversation session is already starting, started or stopping.");
            cancellationToken.ThrowIfCancellationRequested();
            run = new SessionRun();
            _run = run;
        }

        return StartCoreAsync(run, cancellationToken);
    }

    private async Task StartCoreAsync(SessionRun run, CancellationToken cancellationToken)
    {
        try
        {
            using var startup = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, run.Cancellation.Token);
            _core.BeginRun();
            await _executor.StartAsync(startup.Token).ConfigureAwait(false);
            startup.Token.ThrowIfCancellationRequested();
            lock (_lifecycleLock)
                run.IsStarted = true;
        }
        catch
        {
            // Cleanup joins the same stop operation whether startup failed or Stop canceled it.
            run.StartupFinished.TrySetResult();
            await StopRunAsync(run).ConfigureAwait(false);
            throw;
        }
        finally
        {
            run.StartupFinished.TrySetResult();
        }
    }

    public Task StopAsync()
    {
        SessionRun? run;
        lock (_lifecycleLock)
            run = _run;
        return run is null ? Task.CompletedTask : StopRunAsync(run);
    }

    private Task StopRunAsync(SessionRun run)
    {
        TaskCompletionSource completion;
        lock (_lifecycleLock)
        {
            if (run.StopCompletion is not null)
                return run.StopCompletion.Task;
            completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            run.StopCompletion = completion;
            run.Background.Writer.TryComplete();
        }

        _ = StopCoreAsync(run, completion);
        return completion.Task;
    }

    private async Task StopCoreAsync(SessionRun run, TaskCompletionSource completion)
    {
        Exception? failure = null;
        try
        {
            try { await run.Cancellation.CancelAsync().ConfigureAwait(false); } catch { }
            await run.StartupFinished.Task.ConfigureAwait(false);
            try
            {
                // No new requests are admitted once stopping starts. Drain the active turn
                // before closing providers, after startup can no longer open another connection.
                await _core.EndRunAsync().ConfigureAwait(false);
            }
            finally
            {
                await _executor.StopAsync().ConfigureAwait(false);
            }
        }
        catch (Exception exception)
        {
            failure = exception;
        }
        finally
        {
            lock (_lifecycleLock)
            {
                run.Cancellation.Dispose();
                if (ReferenceEquals(_run, run))
                    _run = null;
            }
        }

        if (failure is null)
            completion.TrySetResult();
        else
            completion.TrySetException(failure);
    }

    private SessionRun GetStartedRun()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_run is not { IsStarted: true, StopCompletion: null } run)
            throw new InvalidOperationException("The conversation session is not started.");
        return run;
    }

    public async IAsyncEnumerable<ConversationBackgroundMessage> ReadBackgroundMessagesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Channel<ConversationBackgroundMessage> background;
        lock (_lifecycleLock)
        {
            background = GetStartedRun().Background;
        }

        await foreach (ConversationBackgroundMessage message in
            background.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return message;
        }
    }

    /// <summary>
    /// Enqueues a background message with wait backpressure. Results produced after Stop are
    /// dropped: the next start creates a new channel and never replays them.
    /// </summary>
    async ValueTask IBackgroundResultPublisher.PublishAsync(ConversationBackgroundMessage message, CancellationToken cancellationToken)
    {
        Channel<ConversationBackgroundMessage>? background;
        lock (_lifecycleLock)
            background = _run is { IsStarted: true, StopCompletion: null } run
                ? run.Background
                : null;

        if (background is null)
            return;

        try
        {
            await background.Writer.WriteAsync(message, cancellationToken).ConfigureAwait(false);
        }
        catch (ChannelClosedException)
        {
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        try
        {
            await StopAsync().ConfigureAwait(false);
        }
        finally
        {
            await _core.DisposeAsync().ConfigureAwait(false);
        }
    }

    private sealed class SessionRun
    {
        public CancellationTokenSource Cancellation { get; } = new();
        public TaskCompletionSource StartupFinished { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource? StopCompletion { get; set; }
        public bool IsStarted { get; set; }
        public Channel<ConversationBackgroundMessage> Background { get; } =
            Channel.CreateBounded<ConversationBackgroundMessage>(
                new BoundedChannelOptions(BackgroundChannelCapacity)
                {
                    FullMode = BoundedChannelFullMode.Wait,
                    SingleReader = true,
                    SingleWriter = false,
                    AllowSynchronousContinuations = false,
                });
    }
}
