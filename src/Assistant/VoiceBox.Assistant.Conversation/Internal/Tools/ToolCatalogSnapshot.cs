using System.Diagnostics.CodeAnalysis;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>The immutable tool catalog of one run, in a stable order.</summary>
internal sealed class ToolCatalogSnapshot
{
    private readonly Dictionary<string, ToolCatalogEntry> _byExposedName = new(StringComparer.Ordinal);

    public ToolCatalogSnapshot(IEnumerable<ConversationToolDefinition> definitions)
    {
        ArgumentNullException.ThrowIfNull(definitions);
        ConversationToolDefinition[] sorted = definitions.ToArray();
        if (sorted.Any(static definition => definition is null))
            throw new ArgumentException("Tool definitions cannot contain null values.", nameof(definitions));
        Array.Sort(sorted, static (left, right) =>
        {
            int alias = StringComparer.Ordinal.Compare(left.Id.ServerAlias, right.Id.ServerAlias);
            return alias != 0 ? alias : StringComparer.Ordinal.Compare(left.Id.ToolName, right.Id.ToolName);
        });

        var tools = new List<ToolCatalogEntry>(sorted.Length);
        foreach (ConversationToolDefinition definition in sorted)
        {
            string exposedName = CreateProviderSafeName(definition.Id);
            if (_byExposedName.TryGetValue(exposedName, out ToolCatalogEntry? existing))
            {
                throw new InvalidOperationException(
                    $"Tool identities '{existing.Id}' and '{definition.Id}' map to the same exposed name '{exposedName}'.");
            }

            var entry = new ToolCatalogEntry(definition, exposedName);
            _byExposedName.Add(exposedName, entry);
            tools.Add(entry);
        }

        Tools = tools.AsReadOnly();
        ModelDefinitions = tools.Select(static tool => tool.ModelDefinition).ToArray();
    }

    public IReadOnlyList<ToolCatalogEntry> Tools { get; }

    /// <summary>Function definitions for the provider request, in catalog order.</summary>
    public IReadOnlyList<LlmToolDefinition> ModelDefinitions { get; }

    public bool IsEmpty => Tools.Count == 0;

    public bool TryGetByExposedName(string exposedName, [NotNullWhen(true)] out ToolCatalogEntry? tool)
        => _byExposedName.TryGetValue(exposedName, out tool);

    private static string CreateProviderSafeName(ConversationToolId id)
    {
        const int maxProviderNameLength = 64;
        string value = $"{Sanitize(id.ServerAlias)}__{Sanitize(id.ToolName)}";
        if (value.Length <= maxProviderNameLength)
            return value;

        string suffix = ModelJson.StableHash($"{id.ServerAlias}\0{id.ToolName}");
        int prefixLength = maxProviderNameLength - suffix.Length - 2;
        return $"{value[..prefixLength]}__{suffix}";
    }

    private static string Sanitize(string value)
    {
        var result = new StringBuilder(value.Length);
        foreach (char character in value)
            result.Append(char.IsAsciiLetterOrDigit(character) || character == '_' ? character : '_');
        return result.ToString();
    }
}
