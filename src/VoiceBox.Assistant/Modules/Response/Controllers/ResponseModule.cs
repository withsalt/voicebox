using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Formatting;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Contracts;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Response.Controllers;

/// <summary>
/// Owns one semantic response operation and coordinates protocol-neutral generation with playback.
/// </summary>
internal sealed class ResponseModule : IDisposable, IActiveResponseCancellationPort
{
    private readonly VoiceAssistantOptions _options;
    private readonly bool _bargeInInstalled;
    private readonly IResponseGenerator _responseGenerator;
    private readonly IResponsePlaybackSessionFactory _playbackSessions;
    private readonly IConversationPromptPort _history;
    private readonly IResponseControlSignalPort _control;
    private readonly IAssistantTextStreamSink _textStream;
    private readonly PlaybackStartupGuard _playbackStartupGuard;
    private readonly ILogger _logger;
    private readonly IBargeInDecisionResetPort _bargeInDecisionReset;
    private readonly IPlaybackTextEchoResetPort _playbackTextEchoGuard;
    private readonly object _operationLock = new();
    private readonly Dictionary<long, ResponseOperation> _pendingResourceReleases = [];
    private ResponseOperation? _currentOperation;
    private int _disposed;

    public ResponseModule(
        VoiceAssistantOptions options,
        bool bargeInInstalled,
        IResponseGenerator responseGenerator,
        IResponsePlaybackSessionFactory playbackSessions,
        IConversationPromptPort history,
        IResponseControlSignalPort control,
        IAssistantTextStreamSink textStream,
        PlaybackStartupGuard playbackStartupGuard,
        ILogger logger,
        IBargeInDecisionResetPort bargeInDecisionReset,
        IPlaybackTextEchoResetPort playbackTextEchoGuard)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _bargeInInstalled = bargeInInstalled;
        _responseGenerator = responseGenerator ?? throw new ArgumentNullException(nameof(responseGenerator));
        _playbackSessions = playbackSessions ?? throw new ArgumentNullException(nameof(playbackSessions));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _control = control ?? throw new ArgumentNullException(nameof(control));
        _textStream = textStream ?? throw new ArgumentNullException(nameof(textStream));
        _playbackStartupGuard = playbackStartupGuard ?? throw new ArgumentNullException(nameof(playbackStartupGuard));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _bargeInDecisionReset = bargeInDecisionReset
            ?? throw new ArgumentNullException(nameof(bargeInDecisionReset));
        _playbackTextEchoGuard = playbackTextEchoGuard
            ?? throw new ArgumentNullException(nameof(playbackTextEchoGuard));
    }

    public async Task<ResponseTurnResult> RunRecognizedTextTurnAsync(
        long turnId,
        string userText,
        string? rawText,
        TimeSpan speechDuration,
        CancellationToken cancellationToken,
        long operationId,
        bool canEndContinuousConversation = false,
        bool recognitionAlreadyPublished = false,
        CancellationToken runtimeCancellationToken = default,
        IResponseTurnResourcePort? turnResources = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (operationId <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationId));
        ResponseOperation? operation = turnResources is null
            ? null
            : BeginOperation(
                turnId,
                operationId,
                cancellationToken,
                runtimeCancellationToken,
                turnResources);
        try
        {
            if (string.IsNullOrEmpty(userText) || !AsrTextNormalizer.HasMeaningfulCharacters(userText))
            {
                if (_logger.IsEnabled(LogLevel.Debug))
                    _logger.LogDebug("ASR returned unusable text; skipping response generation.");
                if (operation is not null)
                    EndOperation(operation, turnResources);
                return new ResponseTurnResult(null, false);
            }

            if (!recognitionAlreadyPublished)
            {
                await _control.PublishRecognitionCompletedAsync(
                    turnId,
                    userText,
                    speechDuration,
                    cancellationToken).ConfigureAwait(false);
            }

            if (canEndContinuousConversation &&
                AsrTextNormalizer.ContainsAnyPhrase(userText, _options.Conversation.EndConversationKeywords))
            {
                _bargeInDecisionReset.ResetDecisionState();
                _logger.LogInformation("Detected continuous-conversation termination phrase; returning to wake-word mode.");
                if (operation is not null)
                    EndOperation(operation, turnResources);
                return new ResponseTurnResult(null, true);
            }
        }
        catch
        {
            if (operation is not null)
                EndOperation(operation, turnResources);
            throw;
        }

        operation ??= BeginOperation(
            turnId,
            operationId,
            cancellationToken,
            runtimeCancellationToken,
            turnResources);

        ChatMessage userMessage;
        ChatMessage[] prompt;
        try
        {
            userMessage = _history.AddUserMessage(turnId, userText, out prompt);
        }
        catch
        {
            EndOperation(operation, turnResources);
            throw;
        }

        CancellationToken ct = operation.Cancellation.Token;
        try
        {
            ResponseControlTransition thinking = await _control.StartResponseAsync(
                turnId,
                operation.OperationId,
                ct).ConfigureAwait(false);
            if (!thinking.Accepted)
                throw new OperationCanceledException(ct);

            _playbackTextEchoGuard.BeginResponse(turnId);
            _playbackStartupGuard.Clear();
            OpenInputForBargeInIfEnabled(turnId);

            var request = new ModelTurnRequest(
                prompt,
                new ModelGenerationOptions(
                    MaxTokens: _options.MaxResponseTokens,
                    EnableThinking: _options.Conversation.EnableThinking),
                ToolCatalogSnapshot.Empty,
                turnId);
            var roundSw = Stopwatch.StartNew();
            var visible = new StringBuilder();
            bool transitionedToSpeaking = false;
            bool responseGenerated = false;

            async ValueTask NotifyResponseGeneratedAsync()
            {
                if (responseGenerated)
                    return;
                responseGenerated = true;
                string generated = visible.ToString();
                if (IsCurrentOperation(operation))
                {
                    await _control.PublishResponseCompletedAsync(
                        turnId,
                        operation.OperationId,
                        generated,
                        ct).ConfigureAwait(false);
                }
            }

            async IAsyncEnumerable<SpeakableTextDelta> ReadSpeakableTextAsync(
                [EnumeratorCancellation] CancellationToken streamCancellation = default)
            {
                await foreach (ModelTurnEvent modelEvent in _responseGenerator
                    .GenerateAsync(request, streamCancellation)
                    .WithCancellation(streamCancellation)
                    .ConfigureAwait(false))
                {
                    switch (modelEvent)
                    {
                        case ThinkingDelta thinkingDelta when
                            _options.Conversation.EnableThinking &&
                            !string.IsNullOrEmpty(thinkingDelta.Text):
                            _textStream.RaiseAssistantText(new AssistantTextEventArgs
                            {
                                Delta = thinkingDelta.Text,
                                IsThinking = true,
                            });
                            break;
                        case ThinkingDelta:
                            break;
                        case VisibleTextDelta visibleDelta when !string.IsNullOrEmpty(visibleDelta.Text):
                            if (!transitionedToSpeaking)
                            {
                                transitionedToSpeaking = true;
                                if (!await TryTransitionToSpeakingAsync(operation, streamCancellation)
                                    .ConfigureAwait(false))
                                {
                                    throw new OperationCanceledException(streamCancellation);
                                }
                            }

                            visible.Append(visibleDelta.Text);
                            _textStream.RaiseAssistantText(new AssistantTextEventArgs
                            {
                                Delta = visibleDelta.Text,
                                IsThinking = false,
                            });
                            yield return new SpeakableTextDelta(
                                visibleDelta.Text,
                                visibleDelta.ForceSpeechBoundary,
                                visibleDelta.SpeechBoundaryCompletion);
                            break;
                        case VisibleTextDelta:
                            break;
                        case ModelTurnCompleted:
                            await NotifyResponseGeneratedAsync().ConfigureAwait(false);
                            break;
                        case ToolCallRequested toolCall:
                            throw new InvalidOperationException(
                                $"Structured tool call '{toolCall.CallId}' was received without tool capability.");
                        default:
                            throw new InvalidOperationException(
                                $"Unsupported model turn event '{modelEvent.GetType().Name}'.");
                    }
                }

                await NotifyResponseGeneratedAsync().ConfigureAwait(false);
            }

            string assistantText = await _playbackSessions.PlayAsync(
                turnId,
                ReadSpeakableTextAsync(ct),
                ct).ConfigureAwait(false);
            _bargeInDecisionReset.ResetDecisionState();

            if (!ct.IsCancellationRequested && IsCurrentOperation(operation))
            {
                await _control.PublishPlaybackCompletedAsync(
                    turnId,
                    operation.OperationId,
                    assistantText,
                    roundSw.Elapsed,
                    ct).ConfigureAwait(false);
            }

            return new ResponseTurnResult(userMessage, false);
        }
        catch (OperationCanceledException)
        {
            await TryPublishTerminalOutcomeAsync(
                operation,
                ResponseOperationOutcomeKind.Cancelled,
                failure: null).ConfigureAwait(false);
            throw;
        }
        catch (Exception exception)
        {
            await TryPublishTerminalOutcomeAsync(
                operation,
                ResponseOperationOutcomeKind.Failed,
                exception.GetType().Name).ConfigureAwait(false);
            throw;
        }
        finally
        {
            EndOperation(operation, turnResources);
        }
    }

    internal async Task RunBackgroundTextTurnAsync(
        long turnId,
        long operationId,
        string text,
        CancellationToken cancellationToken,
        IResponseTurnResourcePort turnResources)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ResponseOperation operation = BeginOperation(
            turnId, operationId, cancellationToken, turnResources.RunToken, turnResources);
        CancellationToken ct = operation.Cancellation.Token;
        try
        {
            _playbackTextEchoGuard.BeginResponse(turnId);
            _playbackStartupGuard.Clear();
            OpenInputForBargeInIfEnabled(turnId);

            var roundSw = Stopwatch.StartNew();
            async IAsyncEnumerable<SpeakableTextDelta> ReadTextAsync(
                [EnumeratorCancellation] CancellationToken streamCancellation = default)
            {
                if (!await TryTransitionToSpeakingAsync(operation, streamCancellation).ConfigureAwait(false))
                    throw new OperationCanceledException(streamCancellation);
                yield return new SpeakableTextDelta(text);
            }

            string spokenText = await _playbackSessions.PlayAsync(
                turnId, ReadTextAsync(ct), ct).ConfigureAwait(false);
            _bargeInDecisionReset.ResetDecisionState();
            if (!ct.IsCancellationRequested && IsCurrentOperation(operation))
            {
                await _control.PublishPlaybackCompletedAsync(
                    turnId, operationId, spokenText, roundSw.Elapsed, ct).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            await TryPublishTerminalOutcomeAsync(
                operation, ResponseOperationOutcomeKind.Cancelled, failure: null).ConfigureAwait(false);
            throw;
        }
        catch (Exception exception)
        {
            await TryPublishTerminalOutcomeAsync(
                operation, ResponseOperationOutcomeKind.Failed, exception.GetType().Name).ConfigureAwait(false);
            throw;
        }
        finally
        {
            EndOperation(operation, turnResources);
        }
    }

    internal void ReleaseOperationContext(long turnId)
    {
        ResponseOperation? operation;
        lock (_operationLock)
        {
            if (!_pendingResourceReleases.Remove(turnId, out operation))
            {
                return;
            }
        }

        operation.Resources!.ReleaseResponseContext(turnId, operation.Cancellation);
    }

    public bool Cancel(long turnId, long operationId = 0)
    {
        ResponseOperation? operation;
        lock (_operationLock)
        {
            operation = _currentOperation;
            if (operation is null || operation.TurnId != turnId ||
                (operationId > 0 && operation.OperationId != operationId))
            {
                return false;
            }
        }

        try { _ = operation.Cancellation.CancelAsync(); }
        catch (ObjectDisposedException) { return false; }
        return true;
    }

    /// <summary>
    /// The response module is the only owner allowed to cancel its operation CTS. Runtime
    /// lifecycle/effect collaborators receive this narrow port instead of reaching into
    /// TurnResourceSlot or ResponseTurnResources.
    /// </summary>
    void IActiveResponseCancellationPort.CancelActiveResponse()
    {
        ResponseOperation? operation;
        lock (_operationLock)
            operation = _currentOperation;

        if (operation is not null)
            Cancel(operation.TurnId, operation.OperationId);
    }

    internal void CancelBackgroundOperations()
    {
        if (_responseGenerator is IResponseGeneratorLifecycle lifecycle)
            lifecycle.CancelBackgroundOperations();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        ResponseOperation? operation;
        ResponseOperation[] pending;
        lock (_operationLock)
        {
            operation = _currentOperation;
            _currentOperation = null;
            pending = _pendingResourceReleases.Values.ToArray();
            _pendingResourceReleases.Clear();
        }
        try { operation?.Cancellation.Cancel(); } catch { }
        try { ReleaseOperation(operation); } catch { }
        foreach (ResponseOperation resource in pending)
        {
            try { ReleaseOperation(resource); } catch { }
        }
        if (_responseGenerator is IDisposable disposable)
            disposable.Dispose();
    }

    private ResponseOperation BeginOperation(
        long turnId,
        long requestedOperationId,
        CancellationToken cancellationToken,
        CancellationToken runtimeCancellationToken,
        IResponseTurnResourcePort? turnResources)
    {
        CancellationTokenSource operationCancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            runtimeCancellationToken);
        if (turnResources is not null &&
            !turnResources.TryAttachResponseContext(turnId, operationCancellation))
        {
            turnResources.ReleaseResponseContext(turnId, operationCancellation);
            throw new ResponseOperationAdmissionRejectedException();
        }

        var operation = new ResponseOperation(
            turnId,
            requestedOperationId,
            operationCancellation)
        {
            Resources = turnResources,
        };
        lock (_operationLock)
            _currentOperation = operation;
        return operation;
    }

    private void EndOperation(
        ResponseOperation operation,
        IResponseTurnResourcePort? turnResources)
    {
        lock (_operationLock)
        {
            if (ReferenceEquals(_currentOperation, operation))
                _currentOperation = null;
        }

        if (turnResources is null)
        {
            operation.Cancellation.Dispose();
            return;
        }

        lock (_operationLock)
        {
            _pendingResourceReleases.Add(operation.TurnId, operation);
        }
    }

    private static void ReleaseOperation(ResponseOperation? operation)
    {
        if (operation is null)
            return;

        if (operation.Resources is null)
            operation.Cancellation.Dispose();
        else
            operation.Resources.ReleaseResponseContext(operation.TurnId, operation.Cancellation);
    }

    private bool IsCurrentOperation(ResponseOperation operation)
    {
        lock (_operationLock)
            return ReferenceEquals(_currentOperation, operation) && _control.IsCurrentTurn(operation.TurnId);
    }

    private async ValueTask<bool> TryTransitionToSpeakingAsync(
        ResponseOperation operation,
        CancellationToken cancellationToken)
    {
        if (!IsCurrentOperation(operation))
            return false;

        ResponseControlTransition transition = await _control.StartSpeakableResponseAsync(
            operation.TurnId,
            operation.OperationId,
            cancellationToken).ConfigureAwait(false);
        if (!transition.Accepted)
            return false;

        if (transition.Changed)
        {
            await _textStream.WaitUntilDeliveredAsync(
                transition.PublicationSequence,
                cancellationToken).ConfigureAwait(false);
        }

        return true;
    }

    private void OpenInputForBargeInIfEnabled(long turnId)
    {
        if (!_bargeInInstalled)
            return;

        _control.OpenInputForBargeIn(turnId);
    }

    private async ValueTask TryPublishTerminalOutcomeAsync(
        ResponseOperation operation,
        ResponseOperationOutcomeKind outcome,
        string? failure)
    {
        if (!IsCurrentOperation(operation))
            return;

        try
        {
            await _control.PublishResponseTerminatedAsync(
                operation.TurnId,
                operation.OperationId,
                outcome,
                failure,
                CancellationToken.None).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
        }
        catch (InvalidOperationException)
        {
        }
    }

    private sealed class ResponseOperation(
        long turnId,
        long operationId,
        CancellationTokenSource cancellation)
    {
        public long TurnId { get; } = turnId;

        public long OperationId { get; } = operationId;

        public CancellationTokenSource Cancellation { get; } = cancellation;

        public IResponseTurnResourcePort? Resources { get; set; }
    }

    internal sealed class ResponseOperationAdmissionRejectedException : Exception
    {
    }
}
