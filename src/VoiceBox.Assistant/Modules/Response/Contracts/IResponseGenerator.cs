using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IResponseGenerator
{
    IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
        ModelTurnRequest request,
        CancellationToken cancellationToken);
}
