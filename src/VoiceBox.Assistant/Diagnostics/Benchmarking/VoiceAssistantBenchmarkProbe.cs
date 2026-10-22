using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Diagnostics.Benchmarking;

/// <summary>
/// Passive observation point for offline benchmarks. It records the exact production DTD
/// decision and never changes runtime state.
/// </summary>
internal sealed class VoiceAssistantBenchmarkProbe
{
    private readonly object _lock = new();
    private IInterruptionAdmissionProgressView? _interruptionAdmissions;
    private readonly List<KernelDecisionObservation> _kernelDecisions = [];
    private readonly List<ControlTraceEntry> _controlTrace = [];
    private ICapturePositionView? _capturePosition;
    private long _controlTraceSequence;
    private BargeInDecision _highestDecision;
    private float _peakNearEndProbability;
    private float _peakSmoothedNearEndProbability;
    private RenderAwareBargeInSnapshot _detector;
    private ActiveNearEndDiagnosticSnapshot _coordinator;
    private bool _playbackActive;
    private AssistantState _assistantState;
    private bool _isSpeakingWithoutPendingCandidate;

    /// <summary>
    /// True when every admitted real-time interruption decision has already been reduced by
    /// the single control consumer. Offline evaluators use it to avoid advancing virtual
    /// audio time beyond the 5.4.1 callback-period budget; it never creates,
    /// changes or acknowledges a decision.
    /// </summary>
    public bool InterruptionAdmissionsSettled
        => _interruptionAdmissions?.HasSettledAllInterruptionAdmissions ?? true;

    public void AttachInterruptionAdmissionView(IInterruptionAdmissionProgressView admissions)
        => _interruptionAdmissions = admissions ?? throw new ArgumentNullException(nameof(admissions));

    public void AttachCapturePosition(ICapturePositionView capturePosition)
        => _capturePosition = capturePosition ?? throw new ArgumentNullException(nameof(capturePosition));

    public void ObserveControlTrace(
        string stage,
        long turnId = 0,
        long operationId = 0,
        string? detail = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(stage);
        long capturePosition;
        try { capturePosition = _capturePosition?.CurrentCaptureSamplePosition ?? -1; }
        catch (ObjectDisposedException) { capturePosition = -1; }
        lock (_lock)
        {
            _controlTrace.Add(new ControlTraceEntry(
                Interlocked.Increment(ref _controlTraceSequence),
                stage,
                capturePosition,
                turnId,
                operationId,
                detail));
        }
    }

    public ControlTraceEntry[] GetControlTrace()
    {
        lock (_lock)
            return _controlTrace.ToArray();
    }

    public void ObserveKernelDecision(
        TurnSignal signal,
        TurnSnapshot previous,
        TurnDecision decision)
    {
        lock (_lock)
        {
            _kernelDecisions.Add(new KernelDecisionObservation(
                signal,
                previous,
                decision.Next,
                decision.Effects.ToArray(),
                decision.Accepted));
        }
    }

    public KernelDecisionObservation[] GetKernelDecisions()
    {
        lock (_lock)
            return _kernelDecisions.ToArray();
    }

    public void Observe(
        BargeInDecision decision,
        RenderAwareBargeInSnapshot detector,
        bool startupGuardActive,
        bool playbackActive,
        ActiveNearEndDiagnosticSnapshot coordinator,
        AssistantState assistantState,
        bool isSpeakingWithoutPendingCandidate)
    {
        lock (_lock)
        {
            if (detector.HasObservation)
            {
                _peakNearEndProbability = Math.Max(
                    _peakNearEndProbability,
                    detector.LatestProbability);
                _peakSmoothedNearEndProbability = Math.Max(
                    _peakSmoothedNearEndProbability,
                    detector.LatestSmoothedProbability);
            }

            if (startupGuardActive ||
                decision.PeakScoreDurationMs < _highestDecision.PeakScoreDurationMs)
            {
                return;
            }

            _highestDecision = decision;
            _detector = detector;
            _playbackActive = playbackActive;
            _coordinator = coordinator;
            _assistantState = assistantState;
            _isSpeakingWithoutPendingCandidate = isSpeakingWithoutPendingCandidate;
        }
    }

    public VoiceAssistantBenchmarkSnapshot GetSnapshot()
    {
        lock (_lock)
        {
            return new VoiceAssistantBenchmarkSnapshot(
                _highestDecision,
                _peakNearEndProbability,
                _peakSmoothedNearEndProbability,
                _detector,
                _playbackActive,
                _detector.RenderStatus == RenderTimelineStatus.Valid,
                _coordinator,
                _assistantState,
                _isSpeakingWithoutPendingCandidate);
        }
    }

    // Benchmark-only progress observation. It records a control-plane fact but never
    // creates, changes, or acknowledges an Assistant decision.
    public void ObserveControlPlaneProgress(
        ActiveNearEndDiagnosticSnapshot coordinator,
        AssistantState assistantState,
        bool isSpeakingWithoutPendingCandidate)
    {
        lock (_lock)
        {
            _coordinator = coordinator;
            _assistantState = assistantState;
            _isSpeakingWithoutPendingCandidate = isSpeakingWithoutPendingCandidate;
        }
    }
}
