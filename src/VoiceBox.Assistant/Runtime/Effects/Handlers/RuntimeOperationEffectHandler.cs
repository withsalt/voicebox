using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Runtime.Effects.Contracts;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class RuntimeOperationEffectHandler : IRuntimeEffectHandler
{
    private readonly IActiveResponseCancellationPort _cancellation;
    private readonly ITurnCompletionPort _completion;

    public RuntimeOperationEffectHandler(
        IActiveResponseCancellationPort cancellation,
        ITurnCompletionPort completion)
    {
        _cancellation = cancellation ?? throw new ArgumentNullException(nameof(cancellation));
        _completion = completion ?? throw new ArgumentNullException(nameof(completion));
    }

    public ValueTask CancelOperationAsync(CancelOperationEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _cancellation.CancelActiveResponse();
        return ValueTask.CompletedTask;
    }

    public ValueTask CompleteTurnAsync(CompleteTurnEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _completion.CompleteTurn(effect.TurnId);
        return ValueTask.CompletedTask;
    }
}
