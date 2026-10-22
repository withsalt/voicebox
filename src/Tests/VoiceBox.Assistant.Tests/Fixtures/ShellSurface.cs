using System.Reflection;
using VoiceBox.Assistant.Runtime;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// The declared dependency surface of the AssistantRuntime Shell. The Shell is a set of
/// collaborators rather than one class, so "what the Shell depends on" is the closure of field
/// types reachable from the facade, not the facade's own field list.
/// </summary>
internal static class ShellSurface
{
    private const BindingFlags Declared =
        BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly;

    private const int MaxDepth = 6;

    /// <summary>
    /// Every field declared by the facade or by a collaborator it can reach, excluding adapters.
    /// An adapter exists precisely to face a concrete provider, so counting its fields would make
    /// "the Shell depends only on ports" unprovable by construction.
    /// </summary>
    public static FieldInfo[] Fields { get; } = CollectFields();

    /// <summary>Every collaborator type in the closure, including the facade itself.</summary>
    public static Type[] Types { get; } = CollectTypes();

    private static FieldInfo[] CollectFields()
        => [.. CollectTypes()
            .Where(static type => !type.Name.EndsWith("Adapter", StringComparison.Ordinal))
            .SelectMany(static type => type.GetFields(Declared))];

    private static Type[] CollectTypes()
    {
        var seen = new HashSet<Type>();
        // The Shell is the facade plus every Runtime collaborator, then whatever those declare.
        // Seeding the Runtime namespace matters because collaborators reach each other through
        // narrow ports, so a pure walk from the facade would miss the objects behind them.
        foreach (Type seed in typeof(AssistantRuntime).Assembly.GetTypes())
        {
            if (seed.Namespace?.StartsWith("VoiceBox.Assistant.Runtime", StringComparison.Ordinal) == true)
                Walk(seed, seen, 0);
        }

        Walk(typeof(AssistantRuntime), seen, 0);
        return [.. seen];
    }

    private static void Walk(Type type, HashSet<Type> seen, int depth)
    {
        if (depth > MaxDepth || !IsAssistantType(type) || !seen.Add(type))
            return;

        foreach (FieldInfo field in type.GetFields(Declared))
        {
            Walk(field.FieldType, seen, depth + 1);
            foreach (Type argument in field.FieldType.GenericTypeArguments)
                Walk(argument, seen, depth + 1);
        }
    }

    private static bool IsAssistantType(Type type)
        => !type.IsPrimitive &&
            !type.IsEnum &&
            type != typeof(string) &&
            type.Assembly == typeof(AssistantRuntime).Assembly;
}
