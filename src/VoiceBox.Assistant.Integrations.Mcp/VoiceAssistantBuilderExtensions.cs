using VoiceBox.Assistant.Composition.Builder;

namespace VoiceBox.Assistant.Integrations.Mcp;

public static class VoiceAssistantBuilderExtensions
{
    public static VoiceAssistantBuilder UseMcpTools(
        this VoiceAssistantBuilder builder,
        Action<McpToolsBuilder> configure)
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentNullException.ThrowIfNull(configure);
        var tools = new McpToolsBuilder();
        configure(tools);
        return builder.UseToolRuntime(tools.BuildFactory());
    }
}
