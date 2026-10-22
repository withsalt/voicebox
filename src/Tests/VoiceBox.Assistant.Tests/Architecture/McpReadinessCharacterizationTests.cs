using System.Reflection;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class McpReadinessCharacterizationTests
{
    [Fact]
    public void CurrentAssistantAssembly_HasNoDirectMcpSdkDependency()
    {
        Assembly assistantAssembly = typeof(VoiceAssistant).Assembly;

        Assert.DoesNotContain(
            assistantAssembly.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("ModelContextProtocol.", StringComparison.Ordinal) == true);
    }

    [Fact]
    public void AssistantProjectAssets_DoNotResolveTheMcpSdk()
    {
        string repository = FindRepositoryRoot();
        string assistantAssets = Path.Combine(
            repository,
            "src",
            "VoiceBox.Assistant",
            "obj",
            "project.assets.json");
        string assistantProject = Path.Combine(
            repository,
            "src",
            "VoiceBox.Assistant",
            "VoiceBox.Assistant.csproj");

        Assert.DoesNotContain("ModelContextProtocol.", File.ReadAllText(assistantAssets), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("ModelContextProtocol.", File.ReadAllText(assistantProject), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Tool argument validation needs a JSON Schema implementation, which belongs in the integration
    /// assembly that already accepts third-party dependencies. The core must keep depending only on
    /// <c>Microsoft.Extensions.*</c> so a host can take the assistant without the schema stack.
    /// </summary>
    [Fact]
    public void AssistantProjectAssets_DoNotResolveTheJsonSchemaStack()
    {
        string repository = FindRepositoryRoot();
        string assistantAssets = Path.Combine(
            repository,
            "src",
            "VoiceBox.Assistant",
            "obj",
            "project.assets.json");
        string assistantProject = Path.Combine(
            repository,
            "src",
            "VoiceBox.Assistant",
            "VoiceBox.Assistant.csproj");

        Assert.DoesNotContain("JsonSchema.Net", File.ReadAllText(assistantAssets), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("JsonSchema.Net", File.ReadAllText(assistantProject), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(
            typeof(VoiceAssistant).Assembly.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("JsonSchema", StringComparison.Ordinal) == true);
    }

    private static string FindRepositoryRoot()
    {
        for (DirectoryInfo? current = new(AppContext.BaseDirectory); current is not null; current = current.Parent)
        {
            if (File.Exists(Path.Combine(current.FullName, "src", "VoiceBox.slnx")))
                return current.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate the VoiceBox repository root.");
    }
}
