using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Formatting;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers
{
    /// <summary>
    /// Finishes ASR after an interruption, filters filler and playback echo, then starts a new turn
    /// only when the recognized utterance is usable. Playback is not controlled here.
    /// </summary>
    internal sealed class BargeInRecognitionOutcomeCoordinator : IBargeInRecognitionOutcomePort
    {
        private readonly IRecognitionCompletionPort _recognition;
        private readonly IBargeInValidationControlSignalPort _control;
        private readonly ILogger _logger;
        private readonly PlaybackTextEchoGuard _playbackTextEchoGuard;
        private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;

        public BargeInRecognitionOutcomeCoordinator(
            IRecognitionCompletionPort recognition,
            IBargeInValidationControlSignalPort control,
            ILogger logger,
            PlaybackTextEchoGuard playbackTextEchoGuard,
            VoiceAssistantBenchmarkProbe? benchmarkProbe = null)
        {
            _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
            _control = control ?? throw new ArgumentNullException(nameof(control));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _playbackTextEchoGuard = playbackTextEchoGuard
                ?? throw new ArgumentNullException(nameof(playbackTextEchoGuard));
            _benchmarkProbe = benchmarkProbe;
        }

        public async Task RunRecognitionOutcomeAsync(
            long turnId,
            string asrSessionId,
            TimeSpan speechDuration,
            CancellationToken runToken)
        {
            bool completeWithoutResponse = false;

            try
            {
                _benchmarkProbe?.ObserveControlTrace("recognition-outcome.begin", turnId);
                RecognitionOutcome outcome = await _recognition.FinishAsync(
                    turnId,
                    asrSessionId,
                    speechDuration,
                    runToken).ConfigureAwait(false);
                ThrowIfRecognitionFailed(outcome, runToken);
                string userText = outcome.Text;
                bool meaningful = AsrTextNormalizer.IsMeaningfulPlaybackBargeInText(userText);
                PlaybackTextEchoMatch echoMatch = meaningful
                    ? _playbackTextEchoGuard.Match(userText)
                    : default;
                bool validUserText = meaningful && !echoMatch.IsMatch;
                string resolutionReason = validUserText
                    ? "recognized"
                    : echoMatch.IsMatch
                        ? "playback-echo-text"
                        : string.IsNullOrEmpty(userText)
                            ? "empty"
                            : "filler";
                // The turn is already accepted; this only releases the recognition-side context of
                // the interruption. It can neither reject nor restore anything.
                await _control.ResolveValidationAsync(
                    turnId,
                    userText,
                    resolutionReason,
                    validUserText,
                    runToken).ConfigureAwait(false);

                if (!validUserText)
                {
                    _logger.LogDebug(
                        "播放期打断整句文本无效, 不生成新回复: turnId={TurnId}, reason={Reason}.",
                        turnId,
                        resolutionReason);
                    completeWithoutResponse = true;
                    return;
                }

                await _control.PublishValidatedRecognitionCompletedAsync(
                    turnId,
                    userText,
                    outcome.RawText,
                    speechDuration,
                    runToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (runToken.IsCancellationRequested)
            {
            }
            catch (OperationCanceledException)
            {
                _logger.LogDebug("Pipeline 已被新一轮对话取代。");
                await _control.ResolveValidationAsync(
                    turnId,
                    text: null,
                    reason: "asr-canceled",
                    meaningful: false,
                    CancellationToken.None).ConfigureAwait(false);
                completeWithoutResponse = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "播放期打断整句识别失败, 不启动新的语义回合。");
                await _control.ResolveValidationAsync(
                    turnId,
                    text: null,
                    reason: "asr-error",
                    meaningful: false,
                    CancellationToken.None).ConfigureAwait(false);
                completeWithoutResponse = true;
            }
            finally
            {
                if (completeWithoutResponse)
                {
                    await _control.PublishValidationCancelledAsync(
                        turnId,
                        CancellationToken.None).ConfigureAwait(false);
                }
            }
        }

        private static void ThrowIfRecognitionFailed(
            RecognitionOutcome outcome,
            CancellationToken cancellationToken)
        {
            if (outcome.Kind == RecognitionOutcomeKind.Cancelled)
                throw new OperationCanceledException(cancellationToken);
            if (outcome.Kind == RecognitionOutcomeKind.Failed)
            {
                throw new InvalidOperationException(
                    $"{outcome.Failure.Code}:{outcome.Failure.Component}:{outcome.Failure.MessageKey}");
            }
        }
    }
}
