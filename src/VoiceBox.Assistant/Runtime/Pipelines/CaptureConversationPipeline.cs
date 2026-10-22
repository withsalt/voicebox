using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Runtime.Pipelines;

/// <summary>
/// Wakeup-gated capture conversation path. Speaker evidence is a separate router sink so this
/// pipeline never fans out independently past a wake-word short-circuit; VAD only sees frames
/// the gate routes to conversation.
/// </summary>
internal sealed class CaptureConversationPipeline : IProcessedCaptureFrameSink
{
    private readonly IWakeupModule? _wakeup;
    private readonly InputGate _gate;
    private readonly ITurnBoundarySource _turnBoundary;
    private readonly AecMetricsSampler _aecMetrics;
    private readonly BargeInTurnCoordinator _turns;
    private readonly IRuntimeLivenessView _runtime;
    private readonly IPlaybackStartupGuardView _playbackStartupGuard;
    private readonly IWakeupDecisionSink _wakeupDecisions;
    private readonly IWakeupReplayAuthorizationPort _wakeupReplayAuthorization;
    private readonly INearEndCapturePort _nearEnd;
    private readonly ILogger _logger;

    public CaptureConversationPipeline(
        IWakeupModule? wakeup,
        InputGate gate,
        ITurnBoundarySource turnBoundary,
        AecMetricsSampler aecMetrics,
        BargeInTurnCoordinator turns,
        IRuntimeLivenessView runtime,
        IPlaybackStartupGuardView playbackStartupGuard,
        IWakeupDecisionSink wakeupDecisions,
        IWakeupReplayAuthorizationPort wakeupReplayAuthorization,
        INearEndCapturePort nearEnd,
        ILogger logger)
    {
        _wakeup = wakeup;
        _gate = gate ?? throw new ArgumentNullException(nameof(gate));
        _turnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        _aecMetrics = aecMetrics ?? throw new ArgumentNullException(nameof(aecMetrics));
        _turns = turns ?? throw new ArgumentNullException(nameof(turns));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _playbackStartupGuard = playbackStartupGuard
            ?? throw new ArgumentNullException(nameof(playbackStartupGuard));
        _wakeupDecisions = wakeupDecisions ?? throw new ArgumentNullException(nameof(wakeupDecisions));
        _wakeupReplayAuthorization = wakeupReplayAuthorization
            ?? throw new ArgumentNullException(nameof(wakeupReplayAuthorization));
        _nearEnd = nearEnd ?? throw new ArgumentNullException(nameof(nearEnd));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public void OnProcessedCapture(in ProcessedCaptureFrame frame)
    {
        ReadOnlySpan<float> raw = frame.Audio.Span;
        if (_runtime.IsStopping) return;
        if (raw.IsEmpty) return;

        try
        {
            _aecMetrics.SampleIfDue();
            if (_wakeup is not null)
            {
                // 超时只在"真正空闲" (Idle 且无语音候选) 时判定: 回合进行中或候选累积期间
                // 切回待唤醒会把用户语音掐断在半截 (模式切换后音频只进 KWS, VAD 冻在 mid-speech)。
                if (_turns.IsIdleWithoutCandidate)
                {
                    WakeupTimeoutDecision timeout = _wakeup.CheckIdleTimeout();
                    if (timeout.ModeChanged)
                    {
                        _wakeupDecisions.TryEnqueueObservation(new WakeupAudioDecision(
                            WakeupAudioRoute.WakeDetectorOnly,
                            null,
                            null,
                            WakeupTriggerKind.None,
                            timeout.OldMode,
                            timeout.NewMode,
                            true));
                    }
                }

                WakeupAudioDecision decision = _wakeup.OnAudio(in frame);
                if (decision.ModeChanged || decision.Wakeup is not null)
                {
                    if (!_wakeupDecisions.TryEnqueueObservation(decision))
                        return;
                    if (decision.TriggerKind == WakeupTriggerKind.InSessionWakeup)
                        return;
                }

                if (decision.Route != WakeupAudioRoute.Conversation)
                    return;

                if (!_gate.IsOpen)
                    return;

                if (decision.TriggerKind == WakeupTriggerKind.InitialWakeup)
                {
                    _turnBoundary.Reset();
                    if (decision.ReplayAudio is { Length: > 0 } replay)
                    {
                        _wakeupReplayAuthorization.BeginWakeupReplay();
                        try
                        {
                            _turnBoundary.Observe(replay);
                        }
                        finally
                        {
                            _wakeupReplayAuthorization.EndWakeupReplay();
                        }
                    }
                    return;
                }
            }
            else if (!_gate.IsOpen)
            {
                return;
            }

            if (_playbackStartupGuard.IsActive)
            {
                // The startup guard suppresses authorization only. Keep the main-AEC
                // timeline continuous so speech that starts during the guard remains
                // available as recognition pre-roll once barge-in is confirmed.
                _ = _nearEnd.ObserveRecognitionAudio(in frame);
                return;
            }

            MainAudioObservation observation = _nearEnd.ObserveRecognitionAudio(in frame);
            // VAD can synchronously emit an endpoint while processing this frame. Admit the
            // frame to an already-confirmed near-end ASR capture first so the queued endpoint
            // cannot finalize the session ahead of its final PCM block.
            _nearEnd.FeedFromRecognitionAudio(raw, observation);
            // Render-aware DTD owns playback authorization. The main VAD contributes only
            // onset/endpoint timing on the final primary-AEC recognition stream.
            _turnBoundary.Observe(raw);
        }
        catch (ObjectDisposedException ex) when (_runtime.IsStopping)
        {
            _logger.LogDebug(ex, "忽略停止过程中的输入音频回调。");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "输入音频处理失败, 已跳过当前音频块。");
        }
    }
}
