using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Execution;
using VoiceBox.Assistant.Runtime.Telemetry;

namespace VoiceBox.Assistant.Runtime.TurnContext;

/// <summary>
/// Retires a turn once the Kernel has moved it back to Idle: it releases the per-turn observation
/// context, applies the continuous-conversation wake policy and emits the resulting state
/// publication. It never decides the transition itself.
/// </summary>
internal sealed class TurnCompletionCoordinator : ITurnCompletionPort
{
    private readonly TurnObservationRelease _release;
    private readonly PlaybackStartupGuard _startupGuard;
    private AssistantEffectRunner? _effects;
    private readonly AssistantControlSignalGateway _controlSignals;
    private readonly IWakeupModule? _wakeup;
    private readonly VoiceAssistantConversationOptions _conversationOptions;
    private readonly TurnContextStore _turnContexts;
    private readonly bool _wakeupInstalled;
    private readonly ILogger _logger;

    public TurnCompletionCoordinator(
        TurnObservationRelease release,
        PlaybackStartupGuard startupGuard,
        AssistantControlSignalGateway controlSignals,
        IWakeupModule? wakeup,
        VoiceAssistantConversationOptions conversationOptions,
        TurnContextStore turnContexts,
        ILogger logger)
    {
        _release = release ?? throw new ArgumentNullException(nameof(release));
        _startupGuard = startupGuard ?? throw new ArgumentNullException(nameof(startupGuard));
        _controlSignals = controlSignals ?? throw new ArgumentNullException(nameof(controlSignals));
        _wakeup = wakeup;
        _conversationOptions = conversationOptions ?? throw new ArgumentNullException(nameof(conversationOptions));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _wakeupInstalled = wakeup is not null;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>Composition binds the closed effect runner once its typed handlers exist.</summary>
    public void BindEffects(AssistantEffectRunner effects)
        => _effects = effects ?? throw new ArgumentNullException(nameof(effects));

    void ITurnCompletionPort.CompleteTurn(long turnId)
        => TryCompleteTurn(turnId, endConversationRequested: false, publishStateTransition: false);

    public bool TryCompleteTurn(
        long turnId,
        bool endConversationRequested = false,
        bool publishStateTransition = true)
    {
        if (!_release.TryReleaseCompletedTurn(turnId, out AssistantState old, out bool changed, out bool completedInterruptedTurn))
            return false;

        _turnContexts.ReleaseIfNoResponseCts(turnId);
        _startupGuard.Clear();
        WakeupTimeoutDecision wakeModeDecision = default;
        if (_wakeupInstalled)
        {
            if (endConversationRequested && _conversationOptions.ContinuousConversationEnabled)
            {
                wakeModeDecision = _wakeup!.ReturnToWakeWord();
            }
            else if (_conversationOptions.ContinuousConversationEnabled)
            {
                // 回合完成视为一次交互: 连续对话窗口从"上一轮交互结束"起算,
                // 否则长回复 (LLM+TTS 超过窗口时长) 一结束窗口立即过期。
                _wakeup!.MarkUserActivity();
            }
            else if (!completedInterruptedTurn)
            {
                // The completion snapshot above already resolved whether a validation is still
                // outstanding. Re-reading it here would race with validation acceptance and could
                // send the newly accepted turn back to wake-word mode.
                wakeModeDecision = _wakeup!.ReturnToWakeWord();
            }
        }

        if (changed && publishStateTransition)
            PublishStateChanged(turnId, old, AssistantState.Idle);
        if (wakeModeDecision.ModeChanged)
            _controlSignals.SubmitWakeModeChanged(wakeModeDecision.OldMode, wakeModeDecision.NewMode);
        if (changed && _logger.IsEnabled(LogLevel.Debug))
        {
            _logger.LogDebug(
                AssistantRuntimeEventIds.TurnCompleted,
                "Turn {TurnId} completed.",
                turnId);
        }

        return true;
    }

    public bool TryObserveWakeupInterrupt(long operationOwnerTurnId, out WakeupInterruptDecision decision)
        => _release.TryObserveWakeupInterrupt(operationOwnerTurnId, out decision);

    /// <summary>Releases every observation context the accepted wakeup interruption abandons.</summary>
    public bool CommitWakeupInterrupt(WakeupInterruptDecision decision)
        => _release.CommitWakeupInterrupt(decision);

    /// <summary>
    /// Executes the typed StateChanged publish effect through the closed effect runner so every
    /// stateful public event stays traceable to a Kernel effect.
    /// </summary>
    public void PublishStateChanged(long turnId, AssistantState from, AssistantState to)
        => _effects!.ExecuteAsync(new PublishStateChangedEffect(turnId, from, to))
            .GetAwaiter()
            .GetResult();

}
