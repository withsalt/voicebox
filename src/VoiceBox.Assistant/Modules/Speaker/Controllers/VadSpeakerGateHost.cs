using System.Diagnostics;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Controllers;

/// <summary>
/// Admits VAD speaker-gate verdicts to the control plane on behalf of the speaker module. It owns
/// the bounded observation queue that carries the verdict payload alongside its correlated signal,
/// so the gate never blocks a verification thread on the control loop.
/// </summary>
internal sealed class VadSpeakerGateHost : IVadSpeakerGateHost
{
    private const int ObservationCapacity = 8;

    private readonly AssistantControlIngress _ingress;
    private readonly ControlOperationRegistry _operations;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly RuntimeRunScope _runScope;
    private readonly CorrelatedObservationQueue<VadSpeakerGateVerdict> _observations;
    private ITurnBoundaryEndpointSink? _speechEnded;

    public VadSpeakerGateHost(
        AssistantControlIngress ingress,
        ControlOperationRegistry operations,
        AssistantTaskSupervisor tasks,
        RuntimeRunScope runScope)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _observations = new CorrelatedObservationQueue<VadSpeakerGateVerdict>(
            ingress,
            ObservationCapacity,
            "vad-speaker-data-plane",
            overflowCode: "VBX-CTRL-015",
            mismatchCode: "VBX-CTRL-017",
            mismatchMessageKey: "speaker-gate-observation-mismatch");
    }

    /// <summary>Composition binds the turn boundary that owns the no-speaker fallback path.</summary>
    public void Bind(ITurnBoundaryEndpointSink speechEnded)
        => _speechEnded = speechEnded ?? throw new ArgumentNullException(nameof(speechEnded));

    public CancellationToken RunToken => _runScope.Token;

    public bool IsRunCancelled => _runScope.IsCancelled;

    public long NextOperationId() => _operations.NextOperationId();

    public void Track(string taskName, Task task)
        => _tasks.Track(task, taskName, ownerTurnId: null, AssistantTaskFaultPolicy.ObserveAndLog);

    public SpeechBoundaryObservation SpeechEndedWithoutSpeakerGate(TimeSpan duration)
        => _speechEnded!.HandleSpeechEnded(duration);

    public void SubmitGateResolved(
        long operationId,
        SpeakerVerificationPurpose purpose,
        BargeInSpeakerGateStatus status,
        TimeSpan? pendingSpeechEnd)
    {
        if (!_observations.TryWrite(
                operationId,
                new VadSpeakerGateVerdict(purpose, status, pendingSpeechEnd),
                "speaker-gate-observation-overflow"))
        {
            return;
        }

        if (!_ingress.TryEnqueueCritical(
                AssistantSignal.SpeakerGateResolved(operationId),
                Stopwatch.GetTimestamp(),
                AssistantSignalSource.Validation,
                out _))
        {
            _ingress.TryTerminate(
                new AssistantFailure(
                    "VBX-CTRL-016",
                    "capacity",
                    "vad-speaker-data-plane",
                    "speaker-gate-admission-failed"),
                Stopwatch.GetTimestamp(),
                out _);
        }
    }

    /// <summary>
    /// Takes the payload of an admitted verdict on the control consumer. A mismatch means the
    /// queue and the signal stream diverged, which is a terminal ordering fault.
    /// </summary>
    public bool TryTakeObservation(
        long operationId,
        out SpeakerVerificationPurpose purpose,
        out BargeInSpeakerGateStatus status,
        out TimeSpan? pendingSpeechEnd)
    {
        purpose = default;
        status = default;
        pendingSpeechEnd = null;
        if (!_observations.TryTake(operationId, out VadSpeakerGateVerdict verdict))
            return false;

        purpose = verdict.Purpose;
        status = verdict.Status;
        pendingSpeechEnd = verdict.PendingSpeechEnd;
        return true;
    }

    private readonly record struct VadSpeakerGateVerdict(
        SpeakerVerificationPurpose Purpose,
        BargeInSpeakerGateStatus Status,
        TimeSpan? PendingSpeechEnd);
}
