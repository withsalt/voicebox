using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Runtime.Effects.Contracts;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class ConversationEffectHandler : IConversationEffectHandler
{
    private readonly ConversationHistoryStore _history;

    public ConversationEffectHandler(ConversationHistoryStore history)
        => _history = history ?? throw new ArgumentNullException(nameof(history));

    public ValueTask CommitHistoryAsync(CommitHistoryEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!_history.CommitGeneratedResponse(effect.TurnId, effect.Text))
        {
            throw new InvalidOperationException(
                $"Response history for turn {effect.TurnId} was not pending or was already committed.");
        }
        return ValueTask.CompletedTask;
    }

    public ValueTask RollbackHistoryAsync(RollbackHistoryEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _history.RemoveUnansweredUserMessage(effect.TurnId);
        return ValueTask.CompletedTask;
    }
}
