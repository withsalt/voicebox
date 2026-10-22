using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers
{
    /// <summary>
    /// Records the acoustic decision that authorized a barge-in and emits compact diagnostics.
    /// The detector owns acoustic diagnostics; this coordinator only associates them with a turn.
    /// </summary>
    internal sealed class BargeInDiagnosticsCoordinator : Contracts.IBargeInCandidateDiagnosticsPort
    {
        private readonly VoiceAssistantListeningOptions _listeningOptions;
        private readonly AssistantAudioFormat _audioFormat;
        private readonly BargeInDecisionJournal _journal;
        private readonly ILogger _logger;

        public BargeInDiagnosticsCoordinator(
            VoiceAssistantListeningOptions listeningOptions,
            AssistantAudioFormat audioFormat,
            BargeInDecisionJournal journal,
            ILogger logger)
        {
            _listeningOptions = listeningOptions ?? throw new ArgumentNullException(nameof(listeningOptions));
            _audioFormat = audioFormat ?? throw new ArgumentNullException(nameof(audioFormat));
            _journal = journal ?? throw new ArgumentNullException(nameof(journal));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public BargeInDecisionDiagnostics RecordConfirmed(
            SpeakingDecision decision,
            BargeInTriggerSource source)
        {
            BargeInDecisionDiagnostics diagnostics = decision.ConfirmedBargeInDiagnostics;
            _journal.RememberTriggerSource(decision.TurnId, source);
            _journal.RememberContext(decision.TurnId, diagnostics);

            if (_logger.IsEnabled(LogLevel.Debug))
            {
                _logger.LogDebug(
                    "Barge-in acoustic authorization: source={Source}, turnId={TurnId}, " +
                    "frames={Frames}, accepted={AcceptedFrames}, echoLike={EchoLikeFrames}, " +
                    "acceptedMs={AcceptedMs:F0}, scoreMs={ScoreMs:F0}, peakScoreMs={PeakScoreMs:F0}, " +
                    "reason={Reason}.",
                    source,
                    decision.TurnId,
                    diagnostics.FrameCount,
                    diagnostics.AcceptedFrameCount,
                    diagnostics.EchoLikeFrameCount,
                    diagnostics.AcceptedDurationMs,
                    diagnostics.ScoreDurationMs,
                    diagnostics.PeakScoreDurationMs,
                    diagnostics.LastRejectReason ?? "none");
            }

            return diagnostics;
        }

        public void LogDiscardedListeningCandidate(SpeechEndedDecision decision)
        {
            if (!_logger.IsEnabled(LogLevel.Debug))
                return;

            _logger.LogDebug(
                "Discarded listening candidate: turnId={TurnId}, total={Samples} samples " +
                "({DurationMs:F0}ms), rms={Rms:F4}; confirmation={ConfirmSamples} samples " +
                "({ConfirmDurationMs:F0}ms), rms={ConfirmRms:F4}; threshold={MinDurationMs}ms/{MinRms:F4}.",
                decision.TurnId,
                decision.DiscardedCandidateSamples,
                SamplesToMilliseconds(decision.DiscardedCandidateSamples),
                decision.DiscardedCandidateRms,
                decision.DiscardedConfirmationSamples,
                SamplesToMilliseconds(decision.DiscardedConfirmationSamples),
                decision.DiscardedConfirmationRms,
                _listeningOptions.MinDurationMs,
                _listeningOptions.MinRms);
        }

        public void RecordDiscardedPlaybackCandidate(
            SpeechEndedDecision decision,
            AssistantState currentState)
        {
            if (!_logger.IsEnabled(LogLevel.Debug))
                return;

            BargeInDecisionDiagnostics diagnostics = decision.DiscardedBargeInDiagnostics;
            _logger.LogDebug(
                "Discarded playback candidate: state={State}, turnId={TurnId}, frames={Frames}, " +
                "acceptedMs={AcceptedMs:F0}, scoreMs={ScoreMs:F0}, peakScoreMs={PeakScoreMs:F0}, reason={Reason}.",
                currentState,
                decision.TurnId,
                diagnostics.FrameCount,
                diagnostics.AcceptedDurationMs,
                diagnostics.ScoreDurationMs,
                diagnostics.PeakScoreDurationMs,
                diagnostics.LastRejectReason ?? "none");
        }

        private double SamplesToMilliseconds(long samples)
            => _audioFormat.SamplesToMilliseconds(samples);
    }
}
