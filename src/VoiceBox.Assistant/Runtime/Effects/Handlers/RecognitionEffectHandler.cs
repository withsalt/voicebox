using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class RecognitionEffectHandler : IRecognitionEffectHandler
{
    private readonly RecognitionModule _recognition;
    private readonly TurnContextStore _turnContexts;
    private readonly IAssistantControlSignalSink _signals;
    private readonly IEffectBackgroundScheduler _scheduler;
    private readonly ControlOperationRegistry _operations;
    private readonly IRuntimeCancellationView _runtime;

    public RecognitionEffectHandler(
        RecognitionModule recognition,
        TurnContextStore turnContexts,
        IAssistantControlSignalSink signals,
        IEffectBackgroundScheduler scheduler,
        ControlOperationRegistry operations,
        IRuntimeCancellationView runtime)
    {
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _signals = signals ?? throw new ArgumentNullException(nameof(signals));
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
    }

    public ValueTask PrepareRecognitionAsync(
        PrepareRecognitionEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        CancellationToken runToken = _runtime.RunToken;
        runToken.ThrowIfCancellationRequested();
        _turnContexts.TryCreate(effect.TurnId);
        _scheduler.Schedule(
            "recognition.start",
            effect.TurnId,
            () => _recognition.PrepareAsync(
                effect.TurnId,
                _operations.NextOperationId(),
                runToken));
        return ValueTask.CompletedTask;
    }

    public ValueTask FinishRecognitionAsync(
        FinishRecognitionEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        CancellationToken runToken = _runtime.RunToken;
        runToken.ThrowIfCancellationRequested();
        _scheduler.Schedule(
            "recognition.finish",
            effect.TurnId,
            () => FinishAsync(effect, runToken));
        return ValueTask.CompletedTask;
    }

    private async Task FinishAsync(
        FinishRecognitionEffect effect,
        CancellationToken cancellationToken)
    {
        try
        {
            RecognitionOutcome outcome = await _recognition.FinishAsync(
                effect.TurnId,
                effect.SessionId,
                _turnContexts.TryGetRecognitionMetadata(
                    effect.TurnId,
                    out _,
                    out TimeSpan speechDuration)
                    ? speechDuration
                    : TimeSpan.Zero,
                cancellationToken).ConfigureAwait(false);

            switch (outcome.Kind)
            {
                case RecognitionOutcomeKind.Completed:
                case RecognitionOutcomeKind.Empty:
                    _turnContexts.TrySetRecognitionMetadata(
                        effect.TurnId,
                        outcome.RawText,
                        outcome.SpeechDuration);
                    await _signals.SubmitAsync(
                        AssistantSignal.RecognitionCompleted(
                            effect.TurnId,
                            effect.OperationId,
                            outcome.Text,
                            outcome.SpeechDuration),
                        AssistantSignalSource.Recognition,
                        cancellationToken).ConfigureAwait(false);
                    return;
                case RecognitionOutcomeKind.Cancelled:
                    await _signals.SubmitAsync(
                        AssistantSignal.OperationCancelled(effect.TurnId, effect.OperationId),
                        AssistantSignalSource.Recognition,
                        cancellationToken).ConfigureAwait(false);
                    return;
                case RecognitionOutcomeKind.Failed:
                    await _signals.SubmitAsync(
                        AssistantSignal.OperationFailed(
                            AssistantSignalKind.RecognitionFailed,
                            effect.TurnId,
                            effect.OperationId,
                            outcome.Failure),
                        AssistantSignalSource.Recognition,
                        cancellationToken).ConfigureAwait(false);
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(outcome), outcome.Kind, null);
            }
        }
        finally
        {
            _recognition.ReleaseCaptureTurn(effect.TurnId);
        }
    }
}
