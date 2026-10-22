using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

internal interface IRecognitionEffectHandler
{
    ValueTask PrepareRecognitionAsync(PrepareRecognitionEffect effect, CancellationToken cancellationToken);
    ValueTask FinishRecognitionAsync(FinishRecognitionEffect effect, CancellationToken cancellationToken);
}
