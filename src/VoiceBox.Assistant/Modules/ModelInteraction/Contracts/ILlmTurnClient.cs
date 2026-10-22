using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Contracts;

internal interface ILlmTurnClient
{
    IAsyncEnumerable<ModelTurnEvent> RunAsync(
        ModelTurnRequest request,
        CancellationToken cancellationToken);
}
