using Microsoft.ML.OnnxRuntime;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace VoiceBox.Assistant.Composition.Configuration;

/// <summary>
/// Serializer settings for copying assistant options. Session factories are delegates and cannot
/// be serialized, so the caller copies them back after the JSON round-trip without changing the
/// original options object.
/// </summary>
internal static class AssistantOptionsCloneSerializer
{
    public static JsonSerializerOptions Json { get; } = CreateJsonOptions();

    private static JsonSerializerOptions CreateJsonOptions()
    {
        var options = new JsonSerializerOptions
        {
            TypeInfoResolver = new DefaultJsonTypeInfoResolver
            {
                Modifiers = { SkipSessionOptionsFactory },
            },
        };
        options.MakeReadOnly();
        return options;
    }

    private static void SkipSessionOptionsFactory(JsonTypeInfo typeInfo)
    {
        foreach (JsonPropertyInfo property in typeInfo.Properties)
        {
            if (property.PropertyType == typeof(Func<SessionOptions>))
                property.ShouldSerialize = static (_, _) => false;
        }
    }
}
