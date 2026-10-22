using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Formatting;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Controllers;

internal sealed class RecognitionModule :
    IRecognitionAudioPort,
    IRecognitionCompletionPort,
    IRecognitionCaptureOwner
{
    private static readonly TimeSpan CleanupTimeout = TimeSpan.FromSeconds(5);
    private const int ActiveAudioCapacity = 256;
    private const int TerminalOperationRetentionCapacity = 256;

    private readonly IRecognitionProviderAdapter _provider;
    private readonly IRecognitionControlSignalPort _control;
    private readonly ILogger _logger;
    private readonly object _lock = new();
    private readonly Dictionary<long, RecognitionSession> _sessionsByTurn = [];
    private readonly Dictionary<long, Queue<float[]>> _pendingAudioByTurn = [];
    // Capture ownership for the production turn boundary. _pendingAudioByTurn is the
    // low-level IRecognitionAudioPort admission buffer before a session exists; this ledger
    // owns session identity and endpoint-before-ready state used by the capture router.
    private readonly Dictionary<long, CaptureTurnState> _captureByTurn = [];
    private readonly Dictionary<string, RecognitionSession> _sessionsByProviderId =
        new(StringComparer.Ordinal);
    private readonly Dictionary<TerminalOperationKey, Task<RecognitionOutcome>> _terminalOperations = [];
    private readonly Queue<TerminalOperationKey> _completedTerminalOperations = [];
    private readonly SemaphoreSlim _audioWakeup = new(0);
    private int _audioWakeupPending;
    private int _audioPumpStarted;

    public RecognitionModule(
        IRecognitionProviderAdapter provider,
        IRecognitionControlSignalPort control,
        ILogger logger)
    {
        _provider = provider ?? throw new ArgumentNullException(nameof(provider));
        _control = control ?? throw new ArgumentNullException(nameof(control));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task PrepareAsync(
        long turnId,
        long operationId,
        CancellationToken cancellationToken)
    {
        if (operationId <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationId));
        RecognitionSession session;
        lock (_lock)
        {
            if (_sessionsByTurn.ContainsKey(turnId))
                return;

            session = new RecognitionSession(turnId, operationId);
            _sessionsByTurn.Add(turnId, session);
            if (_pendingAudioByTurn.Remove(turnId, out Queue<float[]>? pending))
            {
                while (pending.TryDequeue(out float[]? audio))
                    session.TryEnqueue(audio);
            }
        }

        string? providerSessionId = null;
        bool attached = false;
        try
        {
            providerSessionId = await _provider.StartSessionAsync(cancellationToken).ConfigureAwait(false);
            attached = AttachProviderSession(session, providerSessionId);
            if (!attached)
            {
                await CleanupProviderSessionAsync(session, providerSessionId).ConfigureAwait(false);
                return;
            }

            if (cancellationToken.IsCancellationRequested)
            {
                await CleanupProviderSessionAsync(session, providerSessionId).ConfigureAwait(false);
                return;
            }

            bool accepted = await _control.PublishRecognitionStartedAsync(
                turnId,
                session.OperationId,
                providerSessionId,
                cancellationToken).ConfigureAwait(false);
            if (accepted)
            {
                // RecognitionStarted is the control-plane ownership boundary. Bootstrap
                // audio is supplied there and must precede capture frames queued while
                // the provider session was opening.
                session.OpenAudioAdmission();
                DrainSessionAudio(session);
            }
            else
            {
                await CleanupAsync(turnId, providerSessionId).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            if (providerSessionId is not null)
                await CleanupProviderSessionAsync(session, providerSessionId).ConfigureAwait(false);
            session.MarkTerminal();
        }
        catch (Exception ex)
        {
            if (providerSessionId is not null)
                await CleanupProviderSessionAsync(session, providerSessionId).ConfigureAwait(false);

            _logger.LogError(ex, "启动 ASR 会话失败。");
            var failure = new AssistantFailure(
                "VBX-CTRL-011",
                "operation",
                "recognition",
                "session-start-failed");
            session.MarkTerminal();
            await _control.PublishRecognitionFailedAsync(
                new RecognitionOutcome(
                    turnId,
                    session.OperationId,
                    RecognitionOutcomeKind.Failed,
                    null,
                    string.Empty,
                    TimeSpan.Zero,
                    failure),
                cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            // Start cancellation/failure may happen before a provider id exists, so it cannot
            // always flow through the terminal-operation path. Retire any terminal session here
            // as a final ownership backstop; the operation path performs the same idempotent
            // removal after Finish/Cleanup.
            if (session.IsTerminal)
                RetireSession(session, providerSessionId);
        }
    }

    public bool TryFeedAudio(long turnId, ReadOnlySpan<float> audio)
    {
        if (audio.IsEmpty)
            return true;

        RecognitionSession? session;
        lock (_lock)
        {
            _sessionsByTurn.TryGetValue(turnId, out session);
            if (session is null)
            {
                if (!_pendingAudioByTurn.TryGetValue(turnId, out Queue<float[]>? pending))
                {
                    pending = new Queue<float[]>();
                    _pendingAudioByTurn.Add(turnId, pending);
                }
                if (pending.Count >= ActiveAudioCapacity)
                    return false;

                pending.Enqueue(audio.ToArray());
                return true;
            }
        }

        if (session.TryBeginDirectFeed(out string sessionId))
        {
            try
            {
                _provider.FeedAudio(sessionId, audio);
            }
            catch (Exception ex)
            {
                session.MarkAudioFeedFailed(ex);
                throw;
            }
            finally
            {
                session.EndDirectFeed();
                if (session.HasPendingAudio)
                    SignalAudioPump();
            }

            return true;
        }

        if (!session.TryEnqueue(
                audio,
                ActiveAudioCapacity))
        {
            // A capture callback may race the control-loop cleanup of a failed/stale session.
            // Such PCM no longer belongs to a live turn and must be ignored, not escalated as a
            // capacity fault; genuine bounded-queue exhaustion remains a hard failure.
            return session.IsClosing;
        }

        SignalAudioPump();
        return true;
    }

    bool IRecognitionAudioPort.HasAttachedSession(long turnId)
    {
        lock (_lock)
        {
            return _captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture) &&
                capture.SessionId is not null;
        }
    }

    public ValueTask DrainAudioAsync()
    {
        RecognitionSession[] sessions;
        lock (_lock)
            sessions = _sessionsByTurn.Values.ToArray();

        foreach (RecognitionSession session in sessions)
            DrainSessionAudio(session);
        return ValueTask.CompletedTask;
    }

    public Task StartAudioPump(CancellationToken cancellationToken)
    {
        if (Interlocked.Exchange(ref _audioPumpStarted, 1) != 0)
            throw new InvalidOperationException("Recognition audio pump can only be started once.");

        return RunAudioPumpAsync(cancellationToken);
    }

    public void ResetAudio(long turnId)
    {
        lock (_lock)
        {
            _pendingAudioByTurn.Remove(turnId);
            if (_sessionsByTurn.TryGetValue(turnId, out RecognitionSession? session))
                session.ClearPendingAudio();
        }
    }

    /// <summary>Returns the provider session identity owned by this module for a turn.</summary>
    public bool TryGetProviderSession(
        long turnId,
        out long operationId,
        out string? sessionId)
    {
        lock (_lock)
        {
            operationId = 0;
            sessionId = null;
            if (!_sessionsByTurn.TryGetValue(turnId, out RecognitionSession? session) ||
                session.ProviderSessionId is not { } providerSessionId)
            {
                return false;
            }

            operationId = session.OperationId;
            sessionId = providerSessionId;
            return true;
        }
    }

    /// <summary>
    /// Classifies capture audio for the active or validation turn. Provider session identity and
    /// all PCM admitted before that session is ready are kept in this module; the caller supplies
    /// only committed snapshot facts and the candidate pre-roll port.
    /// </summary>
    public SpeakingDecision ObserveCaptureAudio(
        long activeTurnId,
        AssistantState conversationState,
        ReadOnlySpan<float> audio,
        BargeInValidationCaptureSnapshot validation)
    {
        if (audio.IsEmpty)
            return default;

        lock (_lock)
        {
            long validationTurnId = validation.TurnId;
            long turnId = validationTurnId != 0 ? validationTurnId : activeTurnId;
            if (turnId <= 0)
                return default;

            if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
            {
                // Late VAD audio after Finish retired this turn has no capture owner. Do not
                // resurrect an empty ledger entry for every completed conversation turn.
                if (validationTurnId == 0 && conversationState != AssistantState.Listening)
                    return default;
                capture = GetOrCreateCaptureLocked(turnId);
            }
            if (capture.SessionId is not null)
            {
                bool isValidation = capture.IsValidation;
                return new SpeakingDecision(
                    SpeakingDecisionKind.FeedAsr,
                    capture.SessionId,
                    ResponseToCancel: isValidation ? validation.ResponseCancellation : null,
                    InterruptedTurnId: isValidation ? validation.InterruptedTurnId : 0,
                    OldState: conversationState,
                    TurnId: isValidation ? validationTurnId : activeTurnId,
                    ConfirmedCandidateSamples: 0,
                    ConfirmedCandidateRms: 0,
                    ConfirmedConfirmationSamples: 0,
                    ConfirmedConfirmationRms: 0,
                    ConfirmationMinDurationMs: 0,
                    ConfirmationMinRms: 0,
                    ConfirmationInitialSuppressionMs: 0,
                    ConfirmedBargeInDiagnostics: default);
            }

            // Once an endpoint has been observed before ASR admission, later capture belongs to
            // the next utterance and must not be appended to the completed turn. Validation is
            // the exception: its endpoint is allowed to race session startup and the buffered
            // tail remains part of that validation turn.
            if (!capture.SpeechEndedBeforeAsrReady &&
                (validationTurnId != 0 || conversationState == AssistantState.Listening))
            {
                if (capture.PendingAudio.Count >= ActiveAudioCapacity)
                {
                    return new SpeakingDecision(
                        SpeakingDecisionKind.RecognitionQueueOverflow,
                        AsrSessionId: null,
                        ResponseToCancel: null,
                        InterruptedTurnId: validation.InterruptedTurnId,
                        OldState: conversationState,
                        TurnId: turnId,
                        ConfirmedCandidateSamples: 0,
                        ConfirmedCandidateRms: 0,
                        ConfirmedConfirmationSamples: 0,
                        ConfirmedConfirmationRms: 0,
                        ConfirmationMinDurationMs: 0,
                        ConfirmationMinRms: 0,
                        ConfirmationInitialSuppressionMs: 0,
                        ConfirmedBargeInDiagnostics: default);
                }

                capture.PendingAudio.Enqueue(audio.ToArray());
            }

            return default;
        }
    }

    public SpeechEndedDecision ObserveCaptureSpeechEnded(
        long activeTurnId,
        long validationTurnId,
        AssistantState conversationState,
        TimeSpan duration)
    {
        lock (_lock)
        {
            long turnId = validationTurnId != 0 ? validationTurnId : activeTurnId;
            if (turnId <= 0)
                return default;

            if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
            {
                // A late endpoint has no owner once Finish retired the capture ledger. Only
                // listening or validation can establish a new pre-session capture record.
                if (conversationState != AssistantState.Listening && validationTurnId == 0)
                    return default;
                capture = GetOrCreateCaptureLocked(turnId);
            }
            if (capture.SessionId is not null)
            {
                return new SpeechEndedDecision(
                    SpeechEndedDecisionKind.ReadyForRecognition,
                    capture.SessionId,
                    turnId,
                    duration,
                    0,
                    0,
                    0,
                    0,
                    default);
            }

            if (conversationState != AssistantState.Listening && validationTurnId == 0)
                return default;

            capture.IsValidation = validationTurnId != 0;
            capture.SpeechEndedBeforeAsrReady = true;
            capture.PendingSpeechDuration = duration;
            return new SpeechEndedDecision(
                SpeechEndedDecisionKind.WaitingForAsrStart,
                AsrSessionId: null,
                turnId,
                duration,
                0,
                0,
                0,
                0,
                default);
        }
    }

    /// <summary>Attaches a provider session after its RecognitionStarted signal is admitted.</summary>
    public bool TryAttachCaptureSession(
        long turnId,
        string sessionId,
        bool isValidationTurn,
        IReadOnlyList<float[]> bootstrapAudio,
        out bool speechAlreadyEnded,
        out TimeSpan speechDuration)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        ArgumentNullException.ThrowIfNull(bootstrapAudio);
        speechAlreadyEnded = false;
        speechDuration = TimeSpan.Zero;
        if (bootstrapAudio.Count > ActiveAudioCapacity)
            return false;

        RecognitionSession session;
        List<float[]> ownedBootstrap = new(bootstrapAudio.Count + ActiveAudioCapacity);
        ownedBootstrap.AddRange(bootstrapAudio);
        lock (_lock)
        {
            if (!_sessionsByTurn.TryGetValue(turnId, out session!) ||
                !StringComparer.Ordinal.Equals(session.ProviderSessionId, sessionId))
                return false;

            CaptureTurnState capture = GetOrCreateCaptureLocked(turnId);
            capture.IsValidation = isValidationTurn;
            while (capture.PendingAudio.TryDequeue(out float[]? audio))
                ownedBootstrap.Add(audio);
        }

        while (true)
        {
            if (!session.TryEnqueueOwnedBootstrap(ownedBootstrap, ActiveAudioCapacity))
                return false;

            ownedBootstrap = [];
            bool attached = false;
            lock (_lock)
            {
                if (!_sessionsByTurn.TryGetValue(turnId, out RecognitionSession? current) ||
                    !ReferenceEquals(current, session) ||
                    !StringComparer.Ordinal.Equals(session.ProviderSessionId, sessionId))
                {
                    return false;
                }

                CaptureTurnState capture = GetOrCreateCaptureLocked(turnId);
                while (capture.PendingAudio.TryDequeue(out float[]? audio))
                    ownedBootstrap.Add(audio);

                if (ownedBootstrap.Count == 0)
                {
                    // Session identity is published only after every frame that raced the owned
                    // handoff was installed in the priority bootstrap queue. Frames arriving
                    // after this point use the ordinary queue and therefore cannot overtake it.
                    capture.SessionId = sessionId;
                    speechAlreadyEnded = capture.SpeechEndedBeforeAsrReady;
                    speechDuration = capture.PendingSpeechDuration;
                    capture.SpeechEndedBeforeAsrReady = false;
                    capture.PendingSpeechDuration = TimeSpan.Zero;
                    attached = true;
                }
            }

            if (attached)
                break;
        }

        // The session may have accumulated PCM while the provider was starting. Open admission
        // only after the owned pre-roll batch is installed, then drain before direct feed begins.
        session.OpenAudioAdmission();
        DrainSessionAudio(session);
        return true;
    }

    public bool CanEnterCaptureRecognizing(
        long turnId,
        string asrSessionId,
        AssistantState conversationState,
        out bool isValidationTurn)
    {
        lock (_lock)
        {
            isValidationTurn = false;
            if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture) ||
                !StringComparer.Ordinal.Equals(capture.SessionId, asrSessionId))
                return false;

            if (capture.IsValidation)
            {
                isValidationTurn = true;
                return true;
            }

            if (conversationState is not (AssistantState.Listening or AssistantState.Recognizing))
                return false;

            return true;
        }
    }

    public bool CommitEnterCaptureRecognizing(
        long turnId,
        string asrSessionId,
        bool isValidationTurn)
    {
        lock (_lock)
        {
            if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture) ||
                !StringComparer.Ordinal.Equals(capture.SessionId, asrSessionId) ||
                capture.IsValidation != isValidationTurn)
            {
                return false;
            }

            capture.SessionId = null;
            capture.PendingAudio.Clear();
            capture.SpeechEndedBeforeAsrReady = false;
            capture.PendingSpeechDuration = TimeSpan.Zero;

            return true;
        }
    }

    public string? GetCaptureSessionId(long activeTurnId, long validationTurnId)
    {
        lock (_lock)
        {
            if (activeTurnId != 0 &&
                _captureByTurn.TryGetValue(activeTurnId, out CaptureTurnState? active) &&
                active.SessionId is not null)
            {
                return active.SessionId;
            }

            return validationTurnId != 0 &&
                _captureByTurn.TryGetValue(validationTurnId, out CaptureTurnState? validation)
                ? validation.SessionId
                : null;
        }
    }

    public bool CanObserveCaptureStartFailure(long turnId, AssistantState conversationState)
        => turnId > 0 && conversationState is AssistantState.Listening or AssistantState.Idle;

    public bool CommitCaptureStartFailure(long turnId)
    {
        lock (_lock)
        {
            if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
                return false;
            capture.SessionId = null;
            capture.PendingAudio.Clear();
            capture.SpeechEndedBeforeAsrReady = false;
            capture.PendingSpeechDuration = TimeSpan.Zero;
            return true;
        }
    }

    public void PrepareCaptureTurn(long turnId)
    {
        if (turnId <= 0)
            return;
        lock (_lock)
        {
            CaptureTurnState capture = GetOrCreateCaptureLocked(turnId);
            capture.SessionId = null;
            capture.PendingAudio.Clear();
            capture.SpeechEndedBeforeAsrReady = false;
            capture.PendingSpeechDuration = TimeSpan.Zero;
        }
    }

    public void ClearCaptureSessionId(long turnId)
    {
        lock (_lock)
        {
            if (_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
                capture.SessionId = null;
        }
    }

    public string? TakeAbandonedCaptureSession(long activeTurnId, long validationTurnId)
    {
        lock (_lock)
        {
            // The capture ledger intentionally clears SessionId when recognition begins. Stop and
            // wakeup cleanup must still be able to find a provider session in the small interval
            // before (or without) a terminal operation, so fall back to the module's authoritative
            // turn/session registry. A terminal operation remains exactly-once through
            // _terminalOperations even if cleanup races FinishAsync.
            string? active = TakeAbandonedTurnSessionLocked(activeTurnId);
            if (active is not null || validationTurnId == activeTurnId)
                return active;

            return TakeAbandonedTurnSessionLocked(validationTurnId);
        }
    }

    private string? TakeAbandonedTurnSessionLocked(long turnId)
    {
        if (turnId <= 0)
            return null;

        string? sessionId = null;
        if (_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
        {
            sessionId = capture.SessionId;
            capture.SessionId = null;
            capture.PendingAudio.Clear();
            capture.SpeechEndedBeforeAsrReady = false;
            capture.PendingSpeechDuration = TimeSpan.Zero;
        }

        if (sessionId is not null)
            return sessionId;

        if (!_sessionsByTurn.TryGetValue(turnId, out RecognitionSession? session) ||
            session.IsTerminal)
        {
            return null;
        }

        return session.ProviderSessionId;
    }

    public void ReleaseCaptureTurn(long turnId)
    {
        lock (_lock)
            _captureByTurn.Remove(turnId);
    }

    private CaptureTurnState GetOrCreateCaptureLocked(long turnId)
    {
        if (!_captureByTurn.TryGetValue(turnId, out CaptureTurnState? capture))
        {
            capture = new CaptureTurnState();
            _captureByTurn.Add(turnId, capture);
        }

        return capture;
    }

    private sealed class CaptureTurnState
    {
        public Queue<float[]> PendingAudio { get; } = new();
        public string? SessionId { get; set; }
        public bool IsValidation { get; set; }
        public bool SpeechEndedBeforeAsrReady { get; set; }
        public TimeSpan PendingSpeechDuration { get; set; }
    }

    public Task<RecognitionOutcome> FinishAsync(
        long turnId,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        var operationKey = new TerminalOperationKey(turnId, sessionId);
        RecognitionSession session;
        TaskCompletionSource<RecognitionOutcome> owner;
        lock (_lock)
        {
            if (_terminalOperations.TryGetValue(operationKey, out Task<RecognitionOutcome>? existing))
                return existing;

            // Ownership lookup and terminal-operation registration are one atomic claim. Without
            // this, a concurrent finisher can retire the session after the deduplication lookup
            // but before RequireSessionLocked, causing a valid duplicate Finish to fail spuriously.
            session = RequireSessionLocked(turnId, sessionId);
            owner = RegisterTerminalOperationLocked(operationKey);
        }

        return StartTerminalOperation(
            owner,
            session,
            operationKey,
            sessionId,
            speechDuration,
            cancellationToken);
    }

    private Task<RecognitionOutcome> GetOrStartTerminalOperation(
        RecognitionSession session,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        var operationKey = new TerminalOperationKey(session.TurnId, sessionId);
        TaskCompletionSource<RecognitionOutcome> owner;
        lock (_lock)
        {
            if (_terminalOperations.TryGetValue(operationKey, out Task<RecognitionOutcome>? existing))
                return existing;

            owner = RegisterTerminalOperationLocked(operationKey);
        }

        return StartTerminalOperation(
            owner,
            session,
            operationKey,
            sessionId,
            speechDuration,
            cancellationToken);
    }

    private TaskCompletionSource<RecognitionOutcome> RegisterTerminalOperationLocked(
        TerminalOperationKey operationKey)
    {
        var owner = new TaskCompletionSource<RecognitionOutcome>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        _terminalOperations.Add(operationKey, owner.Task);
        return owner;
    }

    private Task<RecognitionOutcome> StartTerminalOperation(
        TaskCompletionSource<RecognitionOutcome> owner,
        RecognitionSession session,
        TerminalOperationKey operationKey,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        // Starting an async method executes synchronously until its first incomplete await. Start
        // it only after releasing the ownership lock so provider Feed/Finish can never run under
        // that lock, even when the provider completes synchronously.
        _ = CompleteTerminalOperationAsync(
            owner,
            session,
            operationKey,
            sessionId,
            speechDuration,
            cancellationToken);
        return owner.Task;
    }

    private async Task CompleteTerminalOperationAsync(
        TaskCompletionSource<RecognitionOutcome> completion,
        RecognitionSession session,
        TerminalOperationKey operationKey,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        RecognitionOutcome outcome = default;
        Exception? failure = null;
        try
        {
            outcome = await FinishCoreAsync(
                session,
                sessionId,
                speechDuration,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        finally
        {
            RetireSession(session, sessionId);
            RetainCompletedTerminalOperation(operationKey);
        }

        // Publish completion only after every live ownership registry has released the session.
        // Awaiters therefore cannot observe a completed operation while its per-turn resources
        // are still retained.
        if (failure is null)
            completion.TrySetResult(outcome);
        else
            completion.TrySetException(failure);
    }

    public async Task CleanupAsync(long turnId, string sessionId)
    {
        RecognitionSession session;
        lock (_lock)
        {
            if (!_sessionsByProviderId.TryGetValue(sessionId, out session!))
                return;
        }

        await CleanupProviderSessionAsync(session, sessionId).ConfigureAwait(false);
    }

    public async Task CleanupAbandonedAsrSessionAsync(string sessionId)
    {
        RecognitionSession session;
        lock (_lock)
        {
            if (!_sessionsByProviderId.TryGetValue(sessionId, out session!))
                return;
        }

        await CleanupProviderSessionAsync(session, sessionId).ConfigureAwait(false);
    }

    private bool AttachProviderSession(RecognitionSession session, string providerSessionId)
    {
        lock (_lock)
        {
            if (!session.TryAttach(providerSessionId))
                return false;

            _sessionsByProviderId.Add(providerSessionId, session);
            return true;
        }
    }

    private RecognitionSession RequireSessionLocked(long turnId, string sessionId)
    {
        if (!_sessionsByTurn.TryGetValue(turnId, out RecognitionSession? session) ||
            !StringComparer.Ordinal.Equals(session.ProviderSessionId, sessionId))
        {
            throw new InvalidOperationException(
                $"Recognition session '{sessionId}' is not owned by turn '{turnId}'.");
        }

        return session;
    }

    private async Task<RecognitionOutcome> FinishCoreAsync(
        RecognitionSession session,
        string sessionId,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        bool providerFinishStarted = false;
        try
        {
            session.BeginTerminalDrain();
            try
            {
                await DrainSessionAudioFullyAsync(session, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // DrainSessionAudio records the failed feed and closes PCM admission. Finish
                // must still release the provider before returning the failed turn outcome.
                session.MarkAudioFeedFailed(ex);
            }
            providerFinishStarted = true;
            string? rawText = await _provider.FinishSessionAsync(
                sessionId,
                cancellationToken).ConfigureAwait(false);
            if (session.AudioFeedFailure is { } feedFailure)
                throw new InvalidOperationException("ASR audio feed failed.", feedFailure);
            string text = AsrTextNormalizer.Sanitize(rawText);
            return new RecognitionOutcome(
                session.TurnId,
                session.OperationId,
                string.IsNullOrEmpty(text)
                    ? RecognitionOutcomeKind.Empty
                    : RecognitionOutcomeKind.Completed,
                rawText,
                text,
                speechDuration,
                default);
        }
        catch (OperationCanceledException)
        {
            bool needsProviderCancellation = providerFinishStarted;
            if (!providerFinishStarted)
            {
                // Cancellation during the final drain must not retire the Assistant's only
                // reference before the provider has received its terminal call. Do not race
                // Finish with an in-flight synchronous Feed, or retry an already-started Finish.
                session.ClearPendingAudio();
                try
                {
                    using var cleanupCts = new CancellationTokenSource(CleanupTimeout);
                    await DrainSessionAudioFullyAsync(session, cleanupCts.Token).ConfigureAwait(false);
                    needsProviderCancellation = true;
                    await _provider.FinishSessionAsync(sessionId, cleanupCts.Token).ConfigureAwait(false);
                    needsProviderCancellation = false;
                }
                catch (Exception exception)
                {
                    _logger.LogWarning(exception, "清理取消的 ASR 会话失败: {SessionId}", sessionId);
                }
            }
            if (needsProviderCancellation)
            {
                // Some backends retain a cancelled Finish waiter for retry. The Assistant is
                // retiring this session, so use optional explicit teardown instead of a second
                // Finish, which other backends cannot safely accept.
                try
                {
                    using var cleanupCts = new CancellationTokenSource(CleanupTimeout);
                    await _provider.TryCancelSessionAsync(sessionId, cleanupCts.Token).ConfigureAwait(false);
                }
                catch (Exception exception)
                {
                    _logger.LogWarning(exception, "取消 ASR 后端会话失败: {SessionId}", sessionId);
                }
            }
            return new RecognitionOutcome(
                session.TurnId,
                session.OperationId,
                RecognitionOutcomeKind.Cancelled,
                null,
                string.Empty,
                speechDuration,
                new AssistantFailure(
                    "VBX-CTRL-012",
                    "cancellation",
                    "recognition",
                    "session-cancelled"));
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "ASR recognition failed: turnId={TurnId}, sessionId={SessionId}.",
                session.TurnId,
                sessionId);
            return new RecognitionOutcome(
                session.TurnId,
                session.OperationId,
                RecognitionOutcomeKind.Failed,
                null,
                string.Empty,
                speechDuration,
                new AssistantFailure(
                    "VBX-CTRL-013",
                    "operation",
                    "recognition",
                    "session-failed"));
        }
        finally
        {
            session.MarkTerminal();
        }
    }

    private async Task CleanupProviderSessionAsync(
        RecognitionSession session,
        string sessionId)
    {
        try
        {
            // A rejected start never crosses the control-plane ownership boundary, but
            // it still needs a finite provider teardown path.
            session.OpenAudioAdmission();
            using var cleanupCts = new CancellationTokenSource(CleanupTimeout);
            Task<RecognitionOutcome> terminal = GetOrStartTerminalOperation(
                session,
                sessionId,
                TimeSpan.Zero,
                cleanupCts.Token);

            RecognitionOutcome outcome = await terminal.ConfigureAwait(false);
            if (outcome.Kind is RecognitionOutcomeKind.Failed or RecognitionOutcomeKind.Cancelled)
                throw new InvalidOperationException(outcome.Failure.MessageKey);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "清理未使用的 ASR 会话失败: {SessionId}", sessionId);
        }
    }

    private void RetireSession(RecognitionSession session, string? sessionId)
    {
        lock (_lock)
        {
            if (_sessionsByTurn.TryGetValue(session.TurnId, out RecognitionSession? byTurn) &&
                ReferenceEquals(byTurn, session))
            {
                _sessionsByTurn.Remove(session.TurnId);
            }

            _pendingAudioByTurn.Remove(session.TurnId);
            _captureByTurn.Remove(session.TurnId);
            if (sessionId is not null &&
                _sessionsByProviderId.TryGetValue(sessionId, out RecognitionSession? byProvider) &&
                ReferenceEquals(byProvider, session))
            {
                _sessionsByProviderId.Remove(sessionId);
            }
        }
    }

    private void RetainCompletedTerminalOperation(TerminalOperationKey operationKey)
    {
        lock (_lock)
        {
            _completedTerminalOperations.Enqueue(operationKey);
            while (_completedTerminalOperations.Count > TerminalOperationRetentionCapacity)
            {
                TerminalOperationKey expired = _completedTerminalOperations.Dequeue();
                _terminalOperations.Remove(expired);
            }
        }
    }

    private void DrainSessionAudio(RecognitionSession session)
    {
        if (!session.TryBeginDrain(out string sessionId))
            return;

        try
        {
            while (session.TryDequeue(out float[] audio))
                _provider.FeedAudio(sessionId, audio);
        }
        catch (Exception ex)
        {
            // A result from a session missing PCM must not be accepted as a complete utterance.
            // Keep the cause until the shared terminal operation releases the provider session.
            session.MarkAudioFeedFailed(ex);
            throw;
        }
        finally
        {
            session.EndDrain();
            if (session.HasPendingAudio)
                SignalAudioPump();
        }
    }

    private async Task DrainSessionAudioFullyAsync(
        RecognitionSession session,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            DrainSessionAudio(session);
            if (!session.HasPendingAudio && !session.IsFeedActive)
                return;
            await Task.Yield();
        }
    }

    private void SignalAudioPump()
    {
        if (Volatile.Read(ref _audioPumpStarted) == 0 ||
            Interlocked.Exchange(ref _audioWakeupPending, 1) != 0)
        {
            return;
        }

        _audioWakeup.Release();
    }

    private async Task RunAudioPumpAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                await _audioWakeup.WaitAsync(cancellationToken).ConfigureAwait(false);
                Volatile.Write(ref _audioWakeupPending, 0);
                RecognitionSession[] sessions;
                lock (_lock)
                    sessions = _sessionsByTurn.Values.ToArray();
                foreach (RecognitionSession session in sessions)
                {
                    try
                    {
                        DrainSessionAudio(session);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "ASR audio pump feed failed for turn {TurnId}.", session.TurnId);
                        // Feed failures belong to this session. Release it through the same
                        // deduplicated Finish used by endpoint/cleanup, then keep the pump alive
                        // for later turns instead of leaving all queued feeds without a consumer.
                        if (session.ProviderSessionId is { } sessionId)
                            await CleanupProviderSessionAsync(session, sessionId).ConfigureAwait(false);
                        await _control.PublishRecognitionFailedAsync(
                            new RecognitionOutcome(
                                session.TurnId, session.OperationId, RecognitionOutcomeKind.Failed,
                                null, string.Empty, TimeSpan.Zero,
                                new AssistantFailure("VBX-CTRL-013", "operation", "recognition", "audio-feed-failed")),
                            cancellationToken).ConfigureAwait(false);
                    }
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
    }

    private readonly record struct TerminalOperationKey(long TurnId, string SessionId);

}
