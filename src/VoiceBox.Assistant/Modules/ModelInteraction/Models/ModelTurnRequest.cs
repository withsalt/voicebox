using System.Collections.ObjectModel;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal sealed class ModelTurnRequest
{
    public ModelTurnRequest(
        IEnumerable<ChatMessage> transcript,
        ModelGenerationOptions options,
        ToolCatalogSnapshot toolCatalog,
        long turnId = 0)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        Transcript = Array.AsReadOnly(transcript.ToArray());
        if (Transcript.Any(static message => message is null))
            throw new ArgumentException("The model transcript cannot contain null messages.", nameof(transcript));
        Options = options;
        ToolCatalog = toolCatalog ?? throw new ArgumentNullException(nameof(toolCatalog));
        TurnId = turnId;
    }

    public ReadOnlyCollection<ChatMessage> Transcript { get; }

    public ModelGenerationOptions Options { get; }

    public ToolCatalogSnapshot ToolCatalog { get; }

    /// <summary>
    /// Correlates tool execution and deferred assistant results to the originating conversation
    /// turn. Zero is reserved for callers that do not participate in conversation persistence.
    /// </summary>
    public long TurnId { get; }
}
