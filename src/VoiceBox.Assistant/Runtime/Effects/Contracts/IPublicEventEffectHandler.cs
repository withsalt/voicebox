using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

internal interface IPublicEventEffectHandler
{
    ValueTask PublishWakeModeChangedAsync(PublishWakeModeChangedEffect effect, CancellationToken cancellationToken);
    ValueTask PublishWakeupDetectedAsync(PublishWakeupDetectedEffect effect, CancellationToken cancellationToken);
    ValueTask PublishStateChangedAsync(PublishStateChangedEffect effect, CancellationToken cancellationToken);
    ValueTask PublishBargeInValidationAsync(PublishBargeInValidationEffect effect, CancellationToken cancellationToken);
    ValueTask PublishUserSpokeAsync(PublishUserSpokeEffect effect, CancellationToken cancellationToken);
    ValueTask PublishAssistantResponseGeneratedAsync(
        PublishAssistantResponseGeneratedEffect effect,
        CancellationToken cancellationToken);
    ValueTask PublishAssistantRepliedAsync(
        PublishAssistantRepliedEffect effect,
        CancellationToken cancellationToken);
    ValueTask PublishFaultedAsync(PublishFaultedEffect effect, CancellationToken cancellationToken);
}
