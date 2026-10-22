using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

internal interface IConversationEffectHandler
{
    ValueTask CommitHistoryAsync(CommitHistoryEffect effect, CancellationToken cancellationToken);
    ValueTask RollbackHistoryAsync(RollbackHistoryEffect effect, CancellationToken cancellationToken);
}
