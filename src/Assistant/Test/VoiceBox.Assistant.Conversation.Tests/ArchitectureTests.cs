using System.Reflection;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Tests;

public sealed class ArchitectureTests
{
    [Fact]
    public void AbstractionsDependsOnlyOnTheBcl()
    {
        Assembly abstractions = typeof(IConversationSession).Assembly;
        Assert.DoesNotContain(
            abstractions.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("VoiceBox.", StringComparison.Ordinal) == true);
        Assert.DoesNotContain(
            abstractions.GetReferencedAssemblies(),
            reference => reference.Name?.StartsWith("ModelContextProtocol", StringComparison.Ordinal) == true);

        string project = ReadProject(
            "src",
            "Assistant",
            "VoiceBox.Assistant.Conversation.Abstractions",
            "VoiceBox.Assistant.Conversation.Abstractions.csproj");
        Assert.DoesNotContain("ProjectReference", project, StringComparison.Ordinal);
        Assert.DoesNotContain("PackageReference", project, StringComparison.Ordinal);
    }

    [Fact]
    public void AllPublicToolContractsLiveInAbstractions()
    {
        Assembly abstractions = typeof(IConversationSession).Assembly;
        Type[] toolContracts =
        [
            typeof(IConversationTool),
            typeof(IConversationToolProvider),
            typeof(IConversationToolProviderMetadata),
            typeof(IConversationToolApprovalPolicy),
            typeof(ConversationToolApprovalContext),
            typeof(ConversationToolApproval),
            typeof(ConversationToolId),
            typeof(ConversationToolDescriptor),
            typeof(ConversationToolDefinition),
            typeof(ConversationToolCall),
            typeof(ConversationToolResult),
            typeof(ConversationToolResultKind),
            typeof(ConversationToolRisk),
            typeof(ConversationToolExecutionPolicy),
            typeof(ConversationToolExecutionPreference),
        ];

        Assert.All(toolContracts, type => Assert.Same(abstractions, type.Assembly));
        Assert.Equal("VoiceBox.Assistant.Conversation.Abstractions.Tools", typeof(IConversationTool).Namespace);
    }

    [Fact]
    public void LocalToolsToolProvidersAndApprovalShareOneCallAndOneResultType()
    {
        MethodInfo execute = typeof(IConversationTool).GetMethod(nameof(IConversationTool.ExecuteAsync))!;
        MethodInfo invoke = typeof(IConversationToolProvider).GetMethod(nameof(IConversationToolProvider.InvokeAsync))!;

        Assert.Equal(typeof(ConversationToolCall), execute.GetParameters()[0].ParameterType);
        Assert.Equal(typeof(ConversationToolCall), invoke.GetParameters()[0].ParameterType);
        Assert.Equal(typeof(ValueTask<ConversationToolResult>), execute.ReturnType);
        Assert.Equal(typeof(ValueTask<ConversationToolResult>), invoke.ReturnType);
        PropertyInfo approvedCall = typeof(ConversationToolApprovalContext).GetProperty(nameof(ConversationToolApprovalContext.Call))!;
        Assert.Equal(typeof(ConversationToolCall), approvedCall.PropertyType);

        string[] toolTypeNames = typeof(IConversationTool).Assembly.GetExportedTypes().Select(static type => type.Name).ToArray();
        Assert.DoesNotContain("ConversationToolInvocation", toolTypeNames);
        Assert.DoesNotContain("ConversationToolExecutionResult", toolTypeNames);
    }

    [Fact]
    public void AbstractionsPublicSignaturesDoNotLeakForbiddenDependencies()
    {
        Assembly abstractions = typeof(IConversationSession).Assembly;
        foreach (Type type in abstractions.GetExportedTypes())
        {
            foreach (Type dependency in EnumerateSignatureTypes(type))
            {
                string? ns = dependency.Namespace;
                Assert.False(
                    ns?.StartsWith("VoiceBox.Core", StringComparison.Ordinal) == true ||
                    ns?.StartsWith("ModelContextProtocol", StringComparison.Ordinal) == true,
                    $"{type.FullName} leaks '{dependency.FullName}'.");
            }
        }
    }

    [Fact]
    public void DefaultConversationDoesNotReferenceTheAssistantOrMcpSdk()
    {
        Assembly conversation = typeof(ConversationBuilder).Assembly;
        Assert.DoesNotContain(
            conversation.GetReferencedAssemblies(),
            reference => reference.Name is
                "VoiceBox.Assistant" or
                "ModelContextProtocol.Core" or
                "VoiceBox.Assistant.Conversation.Mcp");
        Assert.Contains(
            conversation.GetReferencedAssemblies(),
            reference => reference.Name == "VoiceBox.Assistant.Conversation.Abstractions");
    }

    private static IEnumerable<Type> EnumerateSignatureTypes(Type type)
    {
        const BindingFlags flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        foreach (MethodBase method in type.GetMethods(flags).Cast<MethodBase>()
            .Concat(type.GetConstructors(flags)))
        {
            yield return method is MethodInfo info ? info.ReturnType : typeof(void);
            foreach (ParameterInfo parameter in method.GetParameters())
                yield return parameter.ParameterType;
        }

        foreach (PropertyInfo property in type.GetProperties(flags))
            yield return property.PropertyType;

        foreach (FieldInfo field in type.GetFields(flags))
            yield return field.FieldType;
    }

    private static string ReadProject(params string[] segments)
    {
        string directory = AppContext.BaseDirectory;
        DirectoryInfo? current = new(directory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, segments[0]);
            if (Directory.Exists(candidate))
                return File.ReadAllText(Path.Combine([candidate, .. segments[1..]]));
            current = current.Parent;
        }

        throw new DirectoryNotFoundException($"Repository root containing '{segments[0]}' was not found.");
    }
}
