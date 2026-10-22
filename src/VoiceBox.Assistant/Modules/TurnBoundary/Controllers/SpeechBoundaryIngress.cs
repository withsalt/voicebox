using System.Diagnostics;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using VoiceBox.Assistant.Runtime.Control.State;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Controllers;

/// <summary>
/// The audio-callback side of the turn boundary. VAD facts are normalised, stamped with a source
/// timestamp and admitted to the bounded ingress; nothing here reduces, publishes or awaits.
/// Capture PCM that no session owns yet waits in a bounded queue until the control consumer has
/// established its semantic ownership.
/// </summary>
internal sealed class SpeechBoundaryIngress : ICommittedTurnObserver, IWakeupReplayAuthorizationPort
{
    private const int DeferredSpeakingAudioCapacity = 256;

    private readonly AssistantControlIngress _controlIngress;
    private readonly TurnStateProjection _projection;
    private readonly ITurnStateView _turnState;
    private readonly IRecognitionAudioPort _recognitionAudio;
    private readonly INearEndEndpointPort _nearEnd;
    private readonly ISpeakerGateCapturePort _speakerGate;
    private readonly VoiceAssistantListeningOptions _listeningOptions;
    private readonly BoundedMpscQueue<DeferredSpeakingAudio> _deferredSpeakingAudio =
        new(DeferredSpeakingAudioCapacity);
    private long _speechAdmissionEpoch;
    private long _pendingSpeechStartProjection;
    private int _speechAudioDrainScheduled;
    private readonly object _wakeupReplayLock = new();
    private long _authorizedWakeupReplayEpoch = -1;

    public SpeechBoundaryIngress(
        AssistantControlIngress controlIngress,
        TurnStateProjection projection,
        ITurnStateView turnState,
        IRecognitionAudioPort recognitionAudio,
        VoiceAssistantListeningOptions listeningOptions,
        INearEndEndpointPort nearEnd,
        ISpeakerGateCapturePort speakerGate)
    {
        _controlIngress = controlIngress ?? throw new ArgumentNullException(nameof(controlIngress));
        _projection = projection ?? throw new ArgumentNullException(nameof(projection));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _recognitionAudio = recognitionAudio ?? throw new ArgumentNullException(nameof(recognitionAudio));
        _listeningOptions = listeningOptions ?? throw new ArgumentNullException(nameof(listeningOptions));
        _nearEnd = nearEnd ?? throw new ArgumentNullException(nameof(nearEnd));
        _speakerGate = speakerGate ?? throw new ArgumentNullException(nameof(speakerGate));
    }

    public long AdmissionEpoch => Volatile.Read(ref _speechAdmissionEpoch);

    /// <summary>
    /// True while a speech start has been admitted but not yet reduced. The public State property
    /// reports Listening during that window so callers never observe a gap.
    /// </summary>
    public bool HasPendingListeningProjection
        => Volatile.Read(ref _pendingSpeechStartProjection) == AdmissionEpoch + 1;

    public void BeginWakeupReplay()
    {
        lock (_wakeupReplayLock)
            // VAD may raise SpeechStarted after Observe returns because the boundary's event is
            // admitted to the control loop. Publish the one-shot authorization immediately and
            // consume it when that queued fact is reduced.
            _authorizedWakeupReplayEpoch = AdmissionEpoch;
    }

    public void EndWakeupReplay()
    {
        // The authorization intentionally survives the synchronous replay call until the queued
        // SpeechStarted observation is reduced by the control loop.
    }

    public bool TryConsumeWakeupReplayAuthorization(long admissionEpoch)
    {
        lock (_wakeupReplayLock)
        {
            if (_authorizedWakeupReplayEpoch != admissionEpoch)
                return false;

            _authorizedWakeupReplayEpoch = -1;
            return true;
        }
    }

    public void OnSpeechStarted(object? sender, EventArgs e)
    {
        // Endpoint VAD observations are data-plane facts, so they must retain callback order with
        // the following PCM/DTD frames. The turn transition remains queued.
        long admissionEpoch = AdmissionEpoch;
        AssistantState projectedState = _projection.Current;
        bool activeCapture = _nearEnd.GetProcessingState().HasActiveCapture;
        if (projectedState == AssistantState.Speaking || activeCapture)
            _nearEnd.ObserveEndpointSpeechStarted();

        bool accepted = _controlIngress.TryEnqueueCritical(
            AssistantSignal.SpeechStartedWithEndpointObservation(activeCapture, admissionEpoch),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.CaptureCallback,
            out _);
        if (accepted &&
            projectedState == AssistantState.Idle &&
            _listeningOptions.MinDurationMs == 0 &&
            _listeningOptions.MinRms == 0)
        {
            MarkPendingSpeechStart(admissionEpoch);
        }
    }

    public void OnSpeaking(object? sender, TurnBoundarySpeechEventArgs e)
    {
        if (e.Audio.Length == 0)
            return;

        long admissionEpoch = AdmissionEpoch;
        bool activeNearEndCapture = _nearEnd.GetProcessingState().HasActiveCapture;
        if (_projection.Current == AssistantState.Speaking || activeNearEndCapture)
        {
            // Preserve the VAD-aligned timeline for a DTD confirmation that may be committed
            // later, without allowing this data-plane observation to bypass the queued VAD
            // state-machine transition.
            _nearEnd.ObserveEndpointSpeechAudio(e.Audio);
        }

        // Active near-end capture owns the ASR feed from the primary AEC timeline. The VAD callback
        // can contain a larger accumulated chunk and must not admit that same PCM through the
        // ordinary recognition path a second time.
        if (activeNearEndCapture)
            return;

        if (TryFeedAdmittedRecognitionAudio(e.Audio))
            return;

        if (!_deferredSpeakingAudio.TryWrite(new DeferredSpeakingAudio(admissionEpoch, e.Audio)))
        {
            _controlIngress.TryTerminate(
                new AssistantFailure(
                    "VBX-CTRL-007",
                    "capacity",
                    "speech-audio-data-plane",
                    "deferred-speaking-overflow"),
                Stopwatch.GetTimestamp(),
                out _);
        }

        ScheduleDeferredSpeakingDrain();
    }

    public void OnSpeechEnded(object? sender, TurnBoundarySpeechEventArgs e)
    {
        long admissionEpoch = AdmissionEpoch;
        bool endpointObservationApplied = _nearEnd.GetProcessingState().HasActiveCapture;
        if (endpointObservationApplied)
        {
            _nearEnd.ObserveEndpointSpeechEnded();
            _speakerGate.Reset();
            return;
        }

        _nearEnd.ObserveUnownedEndpointSpeechEnded();

        _controlIngress.TryEnqueueCritical(
            AssistantSignal.SpeechEnded(
                0,
                e.Duration,
                endpointObservationApplied: false,
                admissionEpoch),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.CaptureCallback,
            out _);
    }

    /// <summary>
    /// Replays PCM the control consumer has taken ownership of. Facts admitted under a superseded
    /// admission epoch belong to an interrupted speech segment and are dropped.
    /// </summary>
    public bool TryReadAdmittedAudio(out float[] audio)
    {
        while (_deferredSpeakingAudio.TryRead(out DeferredSpeakingAudio pending))
        {
            if (pending.AdmissionEpoch == AdmissionEpoch)
            {
                audio = pending.Audio;
                return true;
            }
        }

        audio = [];
        return false;
    }

    public bool HasDeferredSpeakingAudio => _deferredSpeakingAudio.TryPeek(out _);

    public void ClearDeferredSpeakingAudio()
    {
        while (_deferredSpeakingAudio.TryRead(out _))
        {
        }
    }

    public void ReleaseAudioDrainSchedule() => Volatile.Write(ref _speechAudioDrainScheduled, 0);

    public void ScheduleDeferredSpeakingDrain()
    {
        if (Interlocked.Exchange(ref _speechAudioDrainScheduled, 1) != 0)
            return;

        if (_controlIngress.TryEnqueueCritical(
                AssistantSignal.SpeechAudioAvailable(),
                Stopwatch.GetTimestamp(),
                AssistantSignalSource.CaptureCallback,
                out _))
        {
            return;
        }

        Volatile.Write(ref _speechAudioDrainScheduled, 0);
        _controlIngress.TryTerminate(
            new AssistantFailure(
                "VBX-CTRL-014",
                "capacity",
                "speech-audio-data-plane",
                "audio-availability-admission-failed"),
            Stopwatch.GetTimestamp(),
            out _);
    }

    public bool IsCurrentAdmissionEpoch(long admissionEpoch) => admissionEpoch == AdmissionEpoch;

    public void SettlePendingSpeechStart(long admissionEpoch)
        => Interlocked.CompareExchange(ref _pendingSpeechStartProjection, 0, admissionEpoch + 1);

    void ICommittedTurnObserver.OnTurnCommitted(TurnSignal signal, AssistantState state)
    {
        if (signal.Kind == TurnSignalKind.CancelRequested &&
            string.Equals(signal.Reason, "wakeup-interrupt", StringComparison.Ordinal))
        {
            // Clear facts admitted by the interrupted speech epoch before cancellation or playback
            // effects make the wakeup commit externally visible. Capture callbacks submitted after
            // that boundary belong to the fresh wakeup turn.
            ResetAfterWakeupInterrupt();
        }
    }

    private void ResetAfterWakeupInterrupt()
    {
        Volatile.Write(ref _pendingSpeechStartProjection, 0);
        ClearDeferredSpeakingAudio();
        // Publish the new epoch only after the old queue has been drained. Producers that raced
        // with the drain retain the old epoch and are rejected by the consumer; facts admitted
        // after this write belong to the fresh wakeup turn and are never cleared.
        Interlocked.Increment(ref _speechAdmissionEpoch);
    }

    private void MarkPendingSpeechStart(long admissionEpoch)
    {
        long token = admissionEpoch + 1;
        long observed = Volatile.Read(ref _pendingSpeechStartProjection);
        while (observed < token)
        {
            long replaced = Interlocked.CompareExchange(
                ref _pendingSpeechStartProjection,
                token,
                observed);
            if (replaced == observed)
                return;
            observed = replaced;
        }
    }

    private bool TryFeedAdmittedRecognitionAudio(float[] audio)
    {
        // PCM already admitted to the runtime queue predates this callback. It must reach the
        // recognition session first; otherwise a newly admitted direct feed can overtake it while
        // the control loop is attaching the provider session.
        if (_deferredSpeakingAudio.TryPeek(out _))
            return false;

        // A pending single-talk candidate aligns and scores every VAD chunk on the control loop;
        // the direct feed would let audio bypass the identity query while it is still open.
        if (_speakerGate.HasActiveCandidate)
            return false;

        Kernel.Models.TurnSnapshot snapshot = _turnState.Snapshot;
        if (snapshot.ConversationState != AssistantState.Listening ||
            snapshot.TurnId <= 0 ||
            !_recognitionAudio.TryGetProviderSession(snapshot.TurnId, out _, out _))
        {
            return false;
        }

        return _recognitionAudio.TryFeedAudio(snapshot.TurnId, audio);
    }

    private readonly record struct DeferredSpeakingAudio(long AdmissionEpoch, float[] Audio);
}
