using System.Diagnostics;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Owns the barge-in data plane's reset and fault convergence: the playback detector's decision
/// state, the near-end module's buffers and the decision journal. A detector fault converges
/// exactly once as a terminal control-ingress fault instead of mutating any turn.
/// </summary>
internal sealed class BargeInDataPlaneController :
    IBargeInDataPlaneResetPort,
    IRenderAwareFaultSink,
    IRuntimeFaultPublicationGate
{
    private readonly IPlaybackBargeInDetector? _detector;
    private readonly BargeInDecisionJournal _journal;
    private readonly AssistantControlIngress _ingress;
    private readonly RuntimeRunScope _runScope;
    private Exception? _fatalFault;
    private int _faultEventPublished;

    public BargeInDataPlaneController(
        IPlaybackBargeInDetector? detector,
        BargeInDecisionJournal journal,
        AssistantControlIngress ingress,
        RuntimeRunScope runScope)
    {
        _detector = detector;
        _journal = journal ?? throw new ArgumentNullException(nameof(journal));
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
    }

    public bool TryClaim(out Exception? fatalException)
    {
        fatalException = Volatile.Read(ref _fatalFault);
        return Interlocked.Exchange(ref _faultEventPublished, 1) == 0;
    }

    public Exception? FatalFault => Volatile.Read(ref _fatalFault);

    public void ResetAll()
    {
        ResetDecisionState();
        _journal.Clear();
    }

    public void ResetDecisionState()
    {
        if (_detector is null)
            return;

        try
        {
            _detector.ResetDecisionState();
        }
        catch (ObjectDisposedException) when (_runScope.IsStopping)
        {
        }
        catch (Exception ex)
        {
            HandleFatalDetectorFault(ex);
        }
    }

    public void HandleFatalDetectorFault(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        if (Interlocked.CompareExchange(ref _fatalFault, exception, null) is not null)
            return;

        _ingress.TryTerminate(
            new AssistantFailure(
                "VBX-CTRL-006",
                "runtime",
                "render-aware-dtd",
                "playback-detector-fault"),
            Stopwatch.GetTimestamp(),
            out _);
    }

    void IRenderAwareFaultSink.ReportFatalDetectorFault(Exception exception)
        => HandleFatalDetectorFault(exception);
}
