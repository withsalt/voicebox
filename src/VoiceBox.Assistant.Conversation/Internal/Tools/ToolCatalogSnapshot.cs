using VoiceBox.Assistant.Conversation.Internal.Execution;
using System.Collections.ObjectModel;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

internal sealed class ToolCatalogSnapshot
{
    private static readonly IComparer<ToolDefinition> DefinitionComparer =
        Comparer<ToolDefinition>.Create(static (left, right) =>
        {
            int alias = StringComparer.Ordinal.Compare(left.Id.ServerAlias, right.Id.ServerAlias);
            return alias != 0 ? alias : StringComparer.Ordinal.Compare(left.Id.ToolName, right.Id.ToolName);
        });

    public static ToolCatalogSnapshot Empty { get; } = new(Array.Empty<ToolDefinition>());

    public ToolCatalogSnapshot(IEnumerable<ToolDefinition> definitions)
    {
        ArgumentNullException.ThrowIfNull(definitions);
        ToolDefinition[] sorted = definitions.ToArray();
        Array.Sort(sorted, DefinitionComparer);
        Definitions = Array.AsReadOnly(sorted);
        if (Definitions.Any(static definition => definition is null))
            throw new ArgumentException("Tool definitions cannot contain null values.", nameof(definitions));

        var exposedNames = new Dictionary<ConversationToolId, string>();
        var identitiesByExposedName = new Dictionary<string, ConversationToolId>(StringComparer.Ordinal);
        foreach (ToolDefinition definition in Definitions)
        {
            string exposedName = CreateProviderSafeName(definition.Id);
            if (identitiesByExposedName.TryGetValue(exposedName, out ConversationToolId existing))
            {
                throw new InvalidOperationException(
                    $"Tool identities '{existing.ServerAlias}/{existing.ToolName}' and " +
                    $"'{definition.Id.ServerAlias}/{definition.Id.ToolName}' map to the same exposed name '{exposedName}'.");
            }
            identitiesByExposedName.Add(exposedName, definition.Id);
            exposedNames.Add(definition.Id, exposedName);
        }

        ExposedNames = new ReadOnlyDictionary<ConversationToolId, string>(exposedNames);
        Hash = ComputeHash(Definitions, ExposedNames);
    }

    public ReadOnlyCollection<ToolDefinition> Definitions { get; }

    public ReadOnlyDictionary<ConversationToolId, string> ExposedNames { get; }

    public string Hash { get; }

    public bool IsEmpty => Definitions.Count == 0;

    private static string CreateProviderSafeName(ConversationToolId id)
    {
        const int maxProviderNameLength = 64;
        string value = $"{Sanitize(id.ServerAlias)}__{Sanitize(id.ToolName)}";
        if (value.Length <= maxProviderNameLength)
            return value;

        string suffix = OwnedJsonValue.ComputeStableHash($"{id.ServerAlias}\0{id.ToolName}");
        int prefixLength = maxProviderNameLength - suffix.Length - 2;
        return $"{value[..prefixLength]}__{suffix}";
    }

    private static string Sanitize(string value)
    {
        var result = new StringBuilder(value.Length);
        foreach (char character in value)
        {
            result.Append(char.IsAsciiLetterOrDigit(character) || character == '_' ? character : '_');
        }
        return result.ToString();
    }

    private static string ComputeHash(
        ReadOnlyCollection<ToolDefinition> definitions,
        ReadOnlyDictionary<ConversationToolId, string> exposedNames)
    {
        var canonical = new StringBuilder();
        foreach (ToolDefinition definition in definitions)
        {
            canonical
                .Append(definition.Id.ServerAlias).Append('\0')
                .Append(definition.Id.ToolName).Append('\0')
                .Append(exposedNames[definition.Id]).Append('\0')
                .Append(definition.Title).Append('\0')
                .Append(definition.Description).Append('\0')
                .Append((int)definition.Risk).Append('\0')
                .Append((int)definition.ExecutionPolicy).Append('\0')
                .Append(definition.ExecutionPrompt ?? string.Empty).Append('\0')
                .Append(definition.Timeout.Ticks).Append('\0')
                .Append(definition.InputSchema.CanonicalHash).Append('\0')
                .Append(definition.OutputSchema?.CanonicalHash ?? string.Empty).Append('\n');
        }
        return OwnedJsonValue.ComputeStableHash(canonical.ToString());
    }
}
