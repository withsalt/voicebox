using System.Reflection;
using System.Text.RegularExpressions;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tests;

public sealed class DependencyBoundaryTests
{
    [Fact]
    public void AbstractionsReferencesOnlyTheBcl()
    {
        Assembly abstractions = typeof(IConversationSession).Assembly;
        Assert.DoesNotContain(
            abstractions.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("VoiceBox.", StringComparison.Ordinal) == true);
        Assert.DoesNotContain(
            abstractions.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("ModelContextProtocol", StringComparison.Ordinal) == true);
    }

    [Fact]
    public void ToolContractsAreDefinedInAbstractions()
    {
        Assembly abstractions = typeof(IConversationSession).Assembly;
        Assert.Same(abstractions, typeof(IConversationTool).Assembly);
        Assert.Same(abstractions, typeof(IConversationToolProvider).Assembly);
        Assert.Same(abstractions, typeof(IConversationToolApprovalPolicy).Assembly);
        Assert.Equal("VoiceBox.Assistant.Conversation.Abstractions.Tools", typeof(ConversationToolDefinition).Namespace);
    }

    [Fact]
    public void ThisProjectDoesNotReferenceTheDefaultConversationOrAssistant()
    {
        Assert.Null(Type.GetType("VoiceBox.Assistant.Conversation.ConversationBuilder, VoiceBox.Assistant.Conversation"));
        Assert.Null(Type.GetType("VoiceBox.Assistant.VoiceAssistant, VoiceBox.Assistant"));
        Assert.DoesNotContain(
            AppDomain.CurrentDomain.GetAssemblies(),
            assembly => assembly.GetName().Name is
                "VoiceBox.Assistant.Conversation" or
                "VoiceBox.Assistant.Conversation.Mcp" or
                "VoiceBox.Assistant");
    }

    [Fact]
    public void ProjectFileHasOnlyTheAbstractionsProductReference()
    {
        string project = ReadProjectFile();
        string[] references = Regex.Matches(project, "ProjectReference Include=\"([^\"]+)\"")
            .Select(match => match.Groups[1].Value.Replace('\\', '/'))
            .ToArray();

        string reference = Assert.Single(references);
        Assert.EndsWith(
            "VoiceBox.Assistant.Conversation.Abstractions/VoiceBox.Assistant.Conversation.Abstractions.csproj",
            reference,
            StringComparison.Ordinal);
    }

    private static string ReadProjectFile()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(
                current.FullName,
                "src",
                "Tests",
                "VoiceBox.Assistant.Conversation.Abstractions.Tests",
                "VoiceBox.Assistant.Conversation.Abstractions.Tests.csproj");
            if (File.Exists(candidate))
                return File.ReadAllText(candidate);
            current = current.Parent;
        }

        throw new DirectoryNotFoundException("The test project file was not found.");
    }
}
