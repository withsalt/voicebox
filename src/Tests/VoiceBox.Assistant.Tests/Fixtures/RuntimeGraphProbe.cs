using System.Reflection;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Finds a field or method inside the composed Runtime Shell. The shell is a set of collaborators
/// rather than one object, so a probe that used to read a facade field now has to say what it
/// wants and let the graph say who owns it.
/// </summary>
internal static class RuntimeGraphProbe
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
    private const int MaxDepth = 5;

    public static object RequireField(object root, string fieldName)
        => FindField(root, fieldName)
            ?? throw new Xunit.Sdk.XunitException($"Field '{fieldName}' was not found in the runtime graph.");

    public static T RequireField<T>(object root, string fieldName)
        => (T)RequireField(root, fieldName);

    public static object? FindField(object root, string fieldName)
    {
        Visit(root, new HashSet<object>(ReferenceEqualityComparer.Instance), 0, node =>
        {
            FieldInfo? field = node.GetType().GetField(fieldName, Instance);
            return field is null ? null : new object?[] { field.GetValue(node) };
        }, out object?[]? hit);
        return hit?[0];
    }

    public static object? InvokeMethod(object root, string methodName, params object?[] args)
    {
        if (!Visit(root, new HashSet<object>(ReferenceEqualityComparer.Instance), 0, node =>
            {
                MethodInfo? method = node.GetType().GetMethod(methodName, Instance);
                return method is null || method.GetParameters().Length != args.Length
                    ? null
                    : new object?[] { method.Invoke(node, args) };
            }, out object?[]? result))
        {
            throw new Xunit.Sdk.XunitException($"Method '{methodName}' was not found in the runtime graph.");
        }

        return result![0];
    }

    private static bool Visit(
        object node,
        HashSet<object> visited,
        int depth,
        Func<object, object?[]?> probe,
        out object?[]? result)
    {
        result = null;
        if (depth > MaxDepth || !visited.Add(node))
            return false;

        result = probe(node);
        if (result is not null)
            return true;

        foreach (FieldInfo field in node.GetType().GetFields(Instance))
        {
            if (!IsWorthWalking(field.FieldType))
                continue;
            if (field.GetValue(node) is not object child)
                continue;
            if (Visit(child, visited, depth + 1, probe, out result))
                return true;
        }

        return false;
    }

    private static bool IsWorthWalking(Type type)
        => !type.IsPrimitive &&
            type != typeof(string) &&
            type.Namespace?.StartsWith("VoiceBox", StringComparison.Ordinal) == true;
}
