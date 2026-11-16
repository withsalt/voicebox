using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

internal interface IRuntimeEffectHandler
{
    ValueTask CancelOperationAsync(CancelOperationEffect effect, CancellationToken cancellationToken);
    ValueTask CompleteTurnAsync(CompleteTurnEffect effect, CancellationToken cancellationToken);
}
