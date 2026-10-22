using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

internal interface IResponseEffectHandler
{
    ValueTask GenerateResponseAsync(GenerateResponseEffect effect, CancellationToken cancellationToken);
    ValueTask CancelResponseAsync(CancelResponseEffect effect, CancellationToken cancellationToken);
}
