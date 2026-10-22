using VoiceBox.Assistant.Runtime.Effects.Contracts;

namespace VoiceBox.Assistant.Runtime.Effects.Execution;

/// <summary>
/// Immutable build-time bundle of one handler per capability family. It is deliberately a set of
/// named typed slots rather than a single aggregate interface, so no object can answer for more
/// than the families it actually owns.
/// </summary>
internal sealed class AssistantEffectHandlers
{
    public AssistantEffectHandlers(
        IInputGateEffectHandler inputGate,
        IRecognitionEffectHandler recognition,
        IResponseEffectHandler responseGeneration,
        IPlaybackEffectHandler playback,
        IConversationEffectHandler conversation,
        IPublicEventEffectHandler events,
        IRuntimeEffectHandler runtimeEffects)
    {
        InputGate = inputGate ?? throw new ArgumentNullException(nameof(inputGate));
        Recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        ResponseGeneration = responseGeneration ?? throw new ArgumentNullException(nameof(responseGeneration));
        Playback = playback ?? throw new ArgumentNullException(nameof(playback));
        Conversation = conversation ?? throw new ArgumentNullException(nameof(conversation));
        Events = events ?? throw new ArgumentNullException(nameof(events));
        RuntimeEffects = runtimeEffects ?? throw new ArgumentNullException(nameof(runtimeEffects));
    }

    public IInputGateEffectHandler InputGate { get; }
    public IRecognitionEffectHandler Recognition { get; }
    public IResponseEffectHandler ResponseGeneration { get; }
    public IPlaybackEffectHandler Playback { get; }
    public IConversationEffectHandler Conversation { get; }
    public IPublicEventEffectHandler Events { get; }
    public IRuntimeEffectHandler RuntimeEffects { get; }
}
