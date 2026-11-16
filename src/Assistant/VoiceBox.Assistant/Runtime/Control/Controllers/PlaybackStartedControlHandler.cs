using System.Diagnostics;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>Applies playback-start reset semantics in control-loop order.</summary>
internal sealed class PlaybackStartedControlHandler
{
    private readonly ITurnStateView _turnState;
    private readonly IBargeInDecisionResetPort _bargeIn;
    private readonly ITurnBoundarySource _turnBoundary;
    private readonly INearEndResetPort _nearEnd;
    private readonly PlaybackStartupGuard _startupGuard;
    private readonly int _startupGuardMs;

    public PlaybackStartedControlHandler(
        ITurnStateView turnState,
        IBargeInDecisionResetPort bargeIn,
        ITurnBoundarySource turnBoundary,
        INearEndResetPort nearEnd,
        PlaybackStartupGuard startupGuard,
        int startupGuardMs)
    {
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _bargeIn = bargeIn ?? throw new ArgumentNullException(nameof(bargeIn));
        _turnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        _nearEnd = nearEnd ?? throw new ArgumentNullException(nameof(nearEnd));
        _startupGuard = startupGuard ?? throw new ArgumentNullException(nameof(startupGuard));
        _startupGuardMs = startupGuardMs;
    }

    public void Handle(long turnId)
    {
        if (!_turnState.IsCurrentTurn(turnId))
            return;

        _bargeIn.ResetDecisionState();
        _turnBoundary.Reset();
        _nearEnd.ClearAudioReservoirs();
        if (_startupGuardMs > 0)
            _startupGuard.Start(_startupGuardMs);
    }
}
