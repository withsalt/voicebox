using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Execution;

internal sealed class AssistantEffectRunner
{
    private readonly AssistantEffectHandlers _handlers;

    public AssistantEffectRunner(AssistantEffectHandlers handlers)
        => _handlers = handlers ?? throw new ArgumentNullException(nameof(handlers));

    public ValueTask ExecuteAsync(AssistantEffect effect, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(effect);
        return effect switch
        {
            SetInputGateEffect value => _handlers.InputGate.SetInputGateAsync(value, cancellationToken),
            PublishWakeModeChangedEffect value => _handlers.Events.PublishWakeModeChangedAsync(value, cancellationToken),
            PublishWakeupDetectedEffect value => _handlers.Events.PublishWakeupDetectedAsync(value, cancellationToken),
            PrepareRecognitionEffect value => _handlers.Recognition.PrepareRecognitionAsync(value, cancellationToken),
            FinishRecognitionEffect value => _handlers.Recognition.FinishRecognitionAsync(value, cancellationToken),
            GenerateResponseEffect value => _handlers.ResponseGeneration.GenerateResponseAsync(value, cancellationToken),
            CancelResponseEffect value => _handlers.ResponseGeneration.CancelResponseAsync(value, cancellationToken),
            ClearPlaybackQueueEffect value => _handlers.Playback.ClearPlaybackQueueAsync(value, cancellationToken),
            StopPlaybackEffect value => _handlers.Playback.StopPlaybackAsync(value, cancellationToken),
            CommitHistoryEffect value => _handlers.Conversation.CommitHistoryAsync(value, cancellationToken),
            RollbackHistoryEffect value => _handlers.Conversation.RollbackHistoryAsync(value, cancellationToken),
            PublishStateChangedEffect value => _handlers.Events.PublishStateChangedAsync(value, cancellationToken),
            PublishBargeInValidationEffect value => _handlers.Events.PublishBargeInValidationAsync(value, cancellationToken),
            PublishUserSpokeEffect value => _handlers.Events.PublishUserSpokeAsync(value, cancellationToken),
            PublishAssistantResponseGeneratedEffect value =>
                _handlers.Events.PublishAssistantResponseGeneratedAsync(value, cancellationToken),
            PublishAssistantRepliedEffect value =>
                _handlers.Events.PublishAssistantRepliedAsync(value, cancellationToken),
            PublishFaultedEffect value => _handlers.Events.PublishFaultedAsync(value, cancellationToken),
            CancelOperationEffect value => _handlers.RuntimeEffects.CancelOperationAsync(value, cancellationToken),
            CompleteTurnEffect value => _handlers.RuntimeEffects.CompleteTurnAsync(value, cancellationToken),
            _ => throw new ArgumentOutOfRangeException(nameof(effect), effect.GetType().Name, null),
        };
    }
}
